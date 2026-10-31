"""Checkpoint digest values and their provenance, without loading tensors."""

import hashlib
import json
import os
from pathlib import Path


def file_digests(ckpt):
    """Returns digest values and their sources for config and weight files.

    Cached blob names are identities, not a fresh content verification.
    Other files are hashed from their contents.
    """
    index = ckpt / "model.safetensors.index.json"
    if index.exists():
        weights = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        weights = ["model.safetensors"]
    out, sources = {}, {}
    for name in ["config.json"] + ([index.name] if index.exists() else []) + weights:
        p = ckpt / name
        if p.is_symlink():
            blob = os.path.basename(os.readlink(p))
            if len(blob) == 64 and all(c in "0123456789abcdefABCDEF" for c in blob):
                out[name] = blob.lower()
                sources[name] = "hf_cache_blob_name"
                continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        out[name] = h.hexdigest()
        sources[name] = "sha256_content"
    return out, sources


def revision(ckpt):
    if (ckpt / "REVISION").exists():
        return (ckpt / "REVISION").read_text().strip()
    if ckpt.parent.name == "snapshots":
        return ckpt.name
    return None



def completed_reference(out, checkpoint):
    """Reads a successful reference phase bound to the requested checkpoint."""
    out, checkpoint = Path(out), Path(checkpoint).resolve()
    phase = out / "reference"
    if not (phase / "PASS").is_file():
        raise ValueError("reference phase has no PASS receipt")
    try:
        code = (phase / "exit-code.txt").read_text().strip()
    except FileNotFoundError as exc:
        raise ValueError("reference phase has no exit-code receipt") from exc
    if code != "0":
        raise ValueError("reference phase did not exit successfully")
    suite = json.loads((out / "ref/suite.json").read_text())
    if not isinstance(suite.get("checkpoint"), str) or Path(suite["checkpoint"]).resolve() != checkpoint:
        raise ValueError("reference checkpoint differs from requested checkpoint")
    if not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError("reference has no cases")
    return suite
