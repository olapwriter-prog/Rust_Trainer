//! Tokenisation and Qwen ChatML formatting.
//!
//! The trainer only needs `encode`. In production the engine's own tokenizer is
//! plugged in through [`Tokenizer`] (or `agentd` exports `*_ids` directly);
//! with the `hf-tokenizers` feature a Hugging Face `tokenizer.json` can be
//! loaded. [`ByteTokenizer`] is for tests and toy runs.

use crate::data::Text;

pub trait Tokenizer: Send + Sync {
    /// Encode plain text (no special-token parsing).
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
    fn vocab_size(&self) -> usize;
    /// Id of a special token such as `<|im_start|>`.
    fn special(&self, name: &str) -> Option<u32>;
}

/// Bytes are tokens 0..256; `<|im_start|>`=256, `<|im_end|>`=257, `<|endoftext|>`=258.
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }
    fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().filter(|&&i| i < 256).map(|&i| i as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
    fn vocab_size(&self) -> usize {
        259
    }
    fn special(&self, name: &str) -> Option<u32> {
        match name {
            "<|im_start|>" => Some(256),
            "<|im_end|>" => Some(257),
            "<|endoftext|>" => Some(258),
            _ => None,
        }
    }
}

#[cfg(feature = "hf-tokenizers")]
pub struct HfTokenizer(pub tokenizers::Tokenizer);

#[cfg(feature = "hf-tokenizers")]
impl HfTokenizer {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        tokenizers::Tokenizer::from_file(path).map(HfTokenizer).map_err(|e| e.to_string())
    }
}

#[cfg(feature = "hf-tokenizers")]
impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.0.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
    }
    fn decode(&self, ids: &[u32]) -> String {
        self.0.decode(ids, false).unwrap_or_default()
    }
    fn vocab_size(&self) -> usize {
        self.0.get_vocab_size(true)
    }
    fn special(&self, name: &str) -> Option<u32> {
        self.0.token_to_id(name)
    }
}

/// A tokenised training sequence: loss is taken on `tokens[answer_start..]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Example {
    pub tokens: Vec<u32>,
    pub answer_start: usize,
}

impl Example {
    pub fn answer_len(&self) -> usize {
        self.tokens.len().saturating_sub(self.answer_start.max(1))
    }
}

/// Qwen ChatML:
/// `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n{prefix}{answer}<|im_end|>`
/// The answer and its closing `<|im_end|>` are the only trained tokens.
pub struct ChatFormat {
    pub im_start: u32,
    pub im_end: u32,
    /// Text placed after `assistant\n` that is part of the prompt, e.g.
    /// `"<think>\n\n</think>\n\n"` for Qwen3.5 in non-thinking mode.
    pub assistant_prefix: String,
    pub max_len: usize,
}

impl ChatFormat {
    pub fn qwen(tok: &dyn Tokenizer, assistant_prefix: &str, max_len: usize) -> Result<Self, String> {
        Ok(ChatFormat {
            im_start: tok.special("<|im_start|>").ok_or("tokenizer lacks <|im_start|>")?,
            im_end: tok.special("<|im_end|>").ok_or("tokenizer lacks <|im_end|>")?,
            assistant_prefix: assistant_prefix.to_string(),
            max_len,
        })
    }

    fn turn(&self, tok: &dyn Tokenizer, out: &mut Vec<u32>, role: &str, content: &str) {
        out.push(self.im_start);
        out.extend(tok.encode(&format!("{role}\n{content}")));
        out.push(self.im_end);
        out.extend(tok.encode("\n"));
    }

    /// Prompt tokens up to and including `<|im_start|>assistant\n{prefix}`.
    pub fn prompt(&self, tok: &dyn Tokenizer, system: Option<&str>, prompt: &Text) -> Vec<u32> {
        if let Text::Ids(ids) = prompt {
            return ids.clone(); // already formatted by the exporter
        }
        let mut out = Vec::new();
        if let Some(s) = system {
            self.turn(tok, &mut out, "system", s);
        }
        match prompt {
            Text::Str(s) => self.turn(tok, &mut out, "user", s),
            Text::Turns(turns) => {
                for (role, content) in turns {
                    self.turn(tok, &mut out, role, content);
                }
            }
            Text::Ids(_) => unreachable!(),
        }
        out.push(self.im_start);
        out.extend(tok.encode(&format!("assistant\n{}", self.assistant_prefix)));
        out
    }

    pub fn answer(&self, tok: &dyn Tokenizer, answer: &Text) -> Vec<u32> {
        match answer {
            Text::Ids(ids) => ids.clone(),
            Text::Str(s) => {
                let mut v = tok.encode(s);
                v.push(self.im_end);
                v
            }
            Text::Turns(_) => Vec::new(),
        }
    }

    /// Build a training example, truncating the *prompt* from the left (and, if
    /// the answer alone is too long, the answer's tail) to fit `max_len`.
    pub fn example(&self, tok: &dyn Tokenizer, system: Option<&str>, prompt: &Text, answer: &Text) -> Option<Example> {
        let mut p = self.prompt(tok, system, prompt);
        let mut a = self.answer(tok, answer);
        if a.is_empty() {
            return None;
        }
        let max = self.max_len.max(2);
        if a.len() > max - 1 {
            a.truncate(max - 1);
        }
        if p.len() + a.len() > max {
            let drop = p.len() + a.len() - max;
            p.drain(..drop);
        }
        if p.is_empty() {
            return None;
        }
        let answer_start = p.len();
        p.extend(a);
        Some(Example { tokens: p, answer_start })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chatml_layout_and_mask() {
        let tok = ByteTokenizer;
        let f = ChatFormat::qwen(&tok, "", 4096).unwrap();
        let ex = f.example(&tok, Some("sys"), &Text::Str("hi".into()), &Text::Str("yo".into())).unwrap();
        let prompt_txt = tok.decode(&ex.tokens[..ex.answer_start]);
        assert_eq!(prompt_txt, "system\nsys\nuser\nhi\nassistant\n");
        assert_eq!(&ex.tokens[ex.answer_start..], &[b'y' as u32, b'o' as u32, 257]);
        // Truncation keeps the whole answer and the end of the prompt.
        let f2 = ChatFormat::qwen(&tok, "", 8).unwrap();
        let ex2 = f2.example(&tok, None, &Text::Str("long prompt".into()), &Text::Str("ab".into())).unwrap();
        assert_eq!(ex2.tokens.len(), 8);
        assert_eq!(ex2.answer_len(), 3);
    }
}
