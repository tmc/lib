"""Write a local, float32 Transformers oracle for PJRT checkpoint tests."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
versions = {name: importlib.metadata.version(name) for name in ["torch", "transformers", "tokenizers"]}
assert versions == {"torch": "2.9.0", "transformers": "4.57.1", "tokenizers": "0.22.2"}, versions
args.output.mkdir(parents=True, exist_ok=True)
if (args.output / "reference.json").exists():
    raise SystemExit("reference.json already exists")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = {p.name: sha256(p) for p in args.checkpoint.iterdir()
         if p.name in ["config.json", "tokenizer.json", "generation_config.json"] or p.suffix == ".safetensors"}
torch.set_num_threads(4)
torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    args.checkpoint, local_files_only=True, trust_remote_code=False,
    dtype=torch.float32, attn_implementation="eager").eval()
tok = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
cases = []
prompts = [("The capital of France is", 5), ("def fibonacci(n):", 5),
           ("你好，世界", 5), ("The quick brown fox jumps over the lazy dog. " * 4, 24)]
for i, (prompt, steps) in enumerate(prompts):
    ids = tok.encode(prompt).ids
    inputs = torch.tensor([ids], dtype=torch.long)
    cache = None
    rows, generated = [], []
    with torch.inference_mode():
        for step in range(steps):
            out = model(input_ids=inputs, past_key_values=cache, use_cache=True)
            row = out.logits[0, -1].float().contiguous()
            rows.append(row)
            token = int(row.argmax())
            generated.append(token)
            cache = out.past_key_values
            inputs = torch.tensor([[token]], dtype=torch.long)
    filename = f"logits-{i}.f32"
    torch.stack(rows).numpy().astype("<f4").tofile(args.output / filename)
    cases.append({"prompt": prompt, "input_ids": ids, "generated_ids": generated,
                  "decoded": tok.decode(generated), "logits": filename,
                  "logits_sha256": sha256(args.output / filename)})
    print(json.dumps(cases[-1], ensure_ascii=False), flush=True)
reference = {"versions": versions, "files": files, "capacity": 96,
             "vocab": model.config.vocab_size, "cases": cases}
(args.output / "reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")
