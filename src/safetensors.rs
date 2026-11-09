//! Minimal safetensors reader/writer (F32 / F16 / BF16), streaming one tensor at a
//! time so a multi-GB checkpoint is never fully resident.

use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct SafeTensors {
    pub path: PathBuf,
    pub tensors: HashMap<String, TensorInfo>,
    pub metadata: HashMap<String, String>,
    data_offset: u64,
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

impl SafeTensors {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut f = File::open(&path)?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len);
        if n > 100 << 20 {
            return Err(bad("safetensors header too large"));
        }
        let mut hdr = vec![0u8; n as usize];
        f.read_exact(&mut hdr)?;
        let v: Value = serde_json::from_slice(&hdr).map_err(|e| bad(e.to_string()))?;
        let obj = v.as_object().ok_or_else(|| bad("header is not an object"))?;
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        for (k, val) in obj {
            if k == "__metadata__" {
                if let Some(m) = val.as_object() {
                    for (mk, mv) in m {
                        metadata.insert(mk.clone(), mv.as_str().unwrap_or_default().to_string());
                    }
                }
                continue;
            }
            let dtype = val["dtype"].as_str().ok_or_else(|| bad("dtype"))?.to_string();
            let shape = val["shape"].as_array().ok_or_else(|| bad("shape"))?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
            let off = val["data_offsets"].as_array().ok_or_else(|| bad("data_offsets"))?;
            let start = off[0].as_u64().ok_or_else(|| bad("offset"))?;
            let end = off[1].as_u64().ok_or_else(|| bad("offset"))?;
            tensors.insert(k.clone(), TensorInfo { dtype, shape, start, end });
        }
        Ok(SafeTensors { path, tensors, metadata, data_offset: 8 + n })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Read one tensor converted to f32, with its shape.
    pub fn read_f32(&self, name: &str) -> io::Result<(Vec<f32>, Vec<usize>)> {
        let info = self.tensors.get(name).ok_or_else(|| bad(format!("missing tensor {name}")))?;
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(self.data_offset + info.start))?;
        let mut buf = vec![0u8; (info.end - info.start) as usize];
        f.read_exact(&mut buf)?;
        let n = info.numel();
        let out: Vec<f32> = match info.dtype.as_str() {
            "F32" => buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "F16" => buf.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "BF16" => buf.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            other => return Err(bad(format!("{name}: unsupported dtype {other}"))),
        };
        if out.len() != n {
            return Err(bad(format!("{name}: size mismatch")));
        }
        Ok((out, info.shape.clone()))
    }
}

/// Write f32 tensors (sorted by name) plus string metadata. Written to a temp
/// file and renamed, so a crash never leaves a torn file behind.
pub fn write_f32(path: impl AsRef<Path>, tensors: &[(String, Vec<usize>, &[f32])], metadata: &BTreeMap<String, String>) -> io::Result<()> {
    let path = path.as_ref();
    let mut hdr = Map::new();
    if !metadata.is_empty() {
        let m: Map<String, Value> = metadata.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect();
        hdr.insert("__metadata__".into(), Value::Object(m));
    }
    let mut sorted: Vec<&(String, Vec<usize>, &[f32])> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut off = 0u64;
    for (name, shape, data) in &sorted {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}: shape mismatch");
        let n = data.len() as u64 * 4;
        hdr.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [off, off + n]}));
        off += n;
    }
    let mut h = serde_json::to_vec(&Value::Object(hdr))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = io::BufWriter::new(File::create(&tmp)?);
        f.write_all(&(h.len() as u64).to_le_bytes())?;
        f.write_all(&h)?;
        for (_, _, data) in &sorted {
            for v in data.iter() {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        f.flush()?;
        f.get_ref().sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_and_halfs() {
        let dir = std::env::temp_dir().join(format!("st-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.safetensors");
        let a = vec![1.0f32, -2.5, 3.25, 0.0, 7.0, 8.0];
        let mut meta = BTreeMap::new();
        meta.insert("k".to_string(), "v".to_string());
        write_f32(&p, &[("a".into(), vec![2, 3], &a)], &meta).unwrap();
        let st = SafeTensors::open(&p).unwrap();
        let (b, shape) = st.read_f32("a").unwrap();
        assert_eq!(a, b);
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(st.metadata["k"], "v");
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        std::fs::remove_dir_all(dir).ok();
    }
}
