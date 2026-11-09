#!/usr/bin/env python3
"""Download a GGUF model from Docker Hub's open model catalog (OCI artifacts).

usage: fetch_docker_model.py ai/qwen3:0.6B-Q4_0 OUT.gguf

Docker Model Runner publishes models such as ai/qwen3, ai/qwen2.5 and
ai/smollm2 on Docker Hub. Each tag is an OCI manifest whose GGUF layer has
media type application/vnd.docker.ai.gguf.v3. This uses only the public
registry API with an anonymous pull token, so no Docker daemon is needed.
"""
import hashlib, json, sys, urllib.request

def get(url, token=None, accept=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if accept:
        req.add_header("Accept", accept)
    return urllib.request.urlopen(req, timeout=60)

def main(ref, out):
    repo, tag = ref.split(":", 1)
    token = json.load(get(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"))["token"]
    accept = ",".join([
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ])
    man = json.load(get(f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}", token, accept))
    if "manifests" in man:  # index: take the first entry
        man = json.load(get(f"https://registry-1.docker.io/v2/{repo}/manifests/{man['manifests'][0]['digest']}", token, accept))
    layers = [l for l in man["layers"] if "gguf" in l.get("mediaType", "")]
    if not layers:
        sys.exit(f"no GGUF layer in {ref}: {[l.get('mediaType') for l in man['layers']]}")
    layer = layers[0]
    print(f"{ref}: {layer['size'] / 1e6:.0f} MB {layer['digest']}", file=sys.stderr)
    h = hashlib.sha256()
    with get(f"https://registry-1.docker.io/v2/{repo}/blobs/{layer['digest']}", token) as r, open(out + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
    if "sha256:" + h.hexdigest() != layer["digest"]:
        sys.exit("digest mismatch")
    import os
    os.replace(out + ".part", out)
    print(f"wrote {out}", file=sys.stderr)

if __name__ == "__main__":
    main(*sys.argv[1:3])
