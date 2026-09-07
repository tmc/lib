"""Write local Transformers logits for PJRT checkpoint tests."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--capacity", type=int)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompts", type=Path, help="JSON list of {prompt, steps} objects")
    source.add_argument("--teacher", type=Path, help="directory containing a free-running F32 reference")
    args = parser.parse_args()
    teacher = None
    if args.teacher:
        teacher = json.loads((args.teacher / "reference.json").read_text())
        if teacher.get("teacher_forced", False) or teacher.get("compute_dtype", "float32") != "float32":
            parser.error("teacher must be a free-running F32 reference")
        prompts = [{"prompt": c["prompt"], "steps": len(c["generated_ids"])} for c in teacher["cases"]]
    elif args.prompts:
        prompts = json.loads(args.prompts.read_text())
    else:
        prompts = [{"prompt": p, "steps": n} for p, n in [
            ("The capital of France is", 5), ("def fibonacci(n):", 5),
            ("你好，世界", 5), ("The quick brown fox jumps over the lazy dog. " * 4, 24)]]
    capacity = args.capacity if args.capacity is not None else (teacher["capacity"] if teacher else 96)
    if type(capacity) is not int or capacity <= 0:
        parser.error("capacity must be positive")
    if not isinstance(prompts, list) or not prompts:
        parser.error("prompts must be a nonempty list")
    for case in prompts:
        if (not isinstance(case, dict) or set(case) != {"prompt", "steps"}
                or not isinstance(case["prompt"], str) or not case["prompt"]
                or type(case["steps"]) is not int or not 0 < case["steps"] <= capacity):
            parser.error("each prompt requires nonempty text and a positive step count within capacity")
    if args.output.exists():
        parser.error("output directory already exists")
    versions = {name: importlib.metadata.version(name) for name in ["torch", "transformers", "tokenizers"]}
    if versions != {"torch": "2.9.0", "transformers": "4.57.1", "tokenizers": "0.22.2"}:
        parser.error(f"unexpected dependency versions: {versions}")
    files = {p.name: sha256(p) for p in args.checkpoint.iterdir()
             if p.name in ["config.json", "tokenizer.json", "generation_config.json"] or p.suffix == ".safetensors"}
    if teacher and files != teacher["files"]:
        parser.error("teacher checkpoint hashes differ")

    import torch
    from transformers import AutoModelForCausalLM
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    inputs = [tok.encode(c["prompt"]).ids for c in prompts]
    for i, (case, ids) in enumerate(zip(prompts, inputs)):
        if not ids or len(ids) + case["steps"] - 1 > capacity:
            parser.error(f"prompt {i} exceeds capacity or has no tokens")
        if teacher:
            original = teacher["cases"][i]
            if ids != original["input_ids"]:
                parser.error(f"teacher prompt {i} token IDs differ")
            if any(type(v) is not int or not 0 <= v < teacher["vocab"] for v in original["generated_ids"]):
                parser.error(f"invalid teacher token ID in prompt {i}")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False,
        dtype=getattr(torch, args.dtype), attn_implementation="eager").eval()
    if teacher and teacher["vocab"] != model.config.vocab_size:
        parser.error("teacher vocabulary differs")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for i, (case, ids) in enumerate(zip(prompts, inputs)):
        current = torch.tensor([ids], dtype=torch.long)
        cache = None
        rows, generated = [], []
        forced = teacher["cases"][i]["generated_ids"] if teacher else None
        with torch.inference_mode():
            for step in range(case["steps"]):
                out = model(input_ids=current, past_key_values=cache, use_cache=True)
                row = out.logits[0, -1].float().contiguous()
                if not torch.isfinite(row).all():
                    raise ValueError(f"nonfinite logits in prompt {i}, step {step}")
                rows.append(row)
                token = int(row.argmax())
                generated.append(token)
                cache = out.past_key_values
                current = torch.tensor([[forced[step] if forced is not None else token]], dtype=torch.long)
        filename = f"logits-{i}.f32"
        torch.stack(rows).numpy().astype("<f4").tofile(args.output / filename)
        result = {"prompt": case["prompt"], "input_ids": ids, "generated_ids": generated,
                  "decoded": "" if forced is not None else tok.decode(generated), "logits": filename,
                  "logits_sha256": sha256(args.output / filename)}
        if forced is not None:
            result["teacher_ids"] = forced
        cases.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    reference = {"compute_dtype": args.dtype, "teacher_forced": teacher is not None,
                 "versions": versions, "files": files, "capacity": capacity,
                 "vocab": model.config.vocab_size, "cases": cases}
    if teacher:
        reference["teacher_reference_sha256"] = sha256(args.teacher / "reference.json")
    (args.output / "reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
