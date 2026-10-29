"""Generate Transformers reference logits for llama-infer's TestHFParity.

    hf_reference.py --checkpoint DIR --output DIR [--dtype float32|bfloat16]
                    [--device cuda|cpu] [--max-new-tokens 24] [--no-bf16]

For every turn of a fixed set of chat conversations it records the
chat-template token IDs, the greedy continuation (stopping at EOS), and the
full-vocabulary logits of a full causal pass at every position from the
turn's prefill start to its last generated token. The prefill start of a
follow-up turn is the length of the prefix it shares with the previous
turn's prompt and continuation, which a chat runner keeps in its KV cache.

The oracle is float32 with eager attention and TF32 disabled. Use
--dtype bfloat16 only for checkpoints whose float32 copy does not fit; the
suite records the weaker oracle. Unless --no-bf16 is given, the same
sequences are also run in bfloat16, and the bfloat16 error against the
float32 oracle is recorded: it measures what BF16 arithmetic alone costs.
"""

import argparse
import json
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_checkpoint import file_digests, revision

PASSAGE = (
    "The printing press, developed by Johannes Gutenberg around 1440, combined "
    "movable metal type, oil-based ink and a screw press adapted from wine "
    "making. Books that had taken scribes months to copy could be produced in "
    "days, and within fifty years presses operated in more than two hundred "
    "European cities. Cheaper books spread literacy, standardized spelling, "
    "and let scholars compare identical copies of the same text."
)

# The first four follow Luminal's validation suite; "passage" makes a prompt
# longer than llama-infer's 32-row narrow prefill so a whole-prompt chunk
# takes the tile gemm path.
CONVERSATIONS = [
    ("arithmetic", [
        "What is 2 + 2? Answer with the number only.",
        "Now multiply that result by 3. Answer with the number only.",
    ]),
    ("memory", [
        "My dog's name is Pixel. Remember it. Reply briefly.",
        "What is my dog's name? Reply with the name only.",
    ]),
    ("translation", [
        "Translate 'good morning' into French. Reply briefly.",
        "Now translate 'thank you' into French. Reply briefly.",
    ]),
    ("reset_replay", ["What is 2 + 2? Answer with the number only."]),
    ("passage", [
        "Summarize this passage in one sentence.\n\n" + PASSAGE,
        "Who developed it, and when?",
    ]),
]

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def load(ckpt, dtype, device):
    return AutoModelForCausalLM.from_pretrained(
        ckpt, local_files_only=True, dtype=DTYPES[dtype], device_map=device,
        attn_implementation="eager",
    ).eval()


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def full_logits(model, seq, device):
    x = torch.tensor([seq], dtype=torch.long, device=device)
    return model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0].float().cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", choices=DTYPES, default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--no-bf16", action="store_true")
    args = ap.parse_args()
    ckpt = args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    print(f"loading {ckpt} in {args.dtype} on {args.device}", flush=True)
    model = load(ckpt, args.dtype, args.device)
    stops = set()
    for v in [model.generation_config.eos_token_id, model.config.eos_token_id, tok.eos_token_id]:
        stops.update(v if isinstance(v, list) else [] if v is None else [v])
    # Llama 3 Instruct ends a turn with <|eot_id|>, which its base
    # generation config may not list.
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot != tok.unk_token_id:
        stops.add(eot)

    files, file_sources = file_digests(ckpt)
    suite = {
        "checkpoint": str(ckpt),
        "revision": revision(ckpt),
        "files": files,
        "file_digest_sources": file_sources,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
        },
        "dtype": args.dtype,
        "device": args.device,
        "attention": "eager",
        "tf32": False,
        "max_new_tokens": args.max_new_tokens,
        "stop_tokens": sorted(stops),
        "cases": [],
    }
    seqs = []
    with torch.inference_mode():
        for conv, prompts in CONVERSATIONS:
            messages, cached = [], []
            for turn, prompt in enumerate(prompts):
                messages.append({"role": "user", "content": prompt})
                ids = tok.apply_chat_template(
                    messages, tokenize=True, return_dict=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
                x = torch.tensor([ids], dtype=torch.long, device=args.device)
                out = model.generate(
                    input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                    max_new_tokens=args.max_new_tokens, eos_token_id=sorted(stops),
                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else sorted(stops)[0],
                    use_cache=True, disable_compile=True,
                )
                gen = out[0, len(ids):].tolist()
                seq = ids + gen
                start = common_prefix(cached, ids)
                logits = full_logits(model, seq, args.device)
                if not torch.isfinite(logits).all():
                    raise ValueError(f"{conv}_{turn + 1}: non-finite reference logits")
                # The cached generation must agree with an independent full
                # causal pass before the fixture is accepted.
                replay = logits[len(ids) - 1:len(seq) - 1].argmax(-1).tolist()
                replay_ok = replay == gen
                if not replay_ok and args.dtype == "float32":
                    raise AssertionError(f"{conv}_{turn + 1}: cached generation {gen} != full causal replay {replay}")
                name = f"{conv}_{turn + 1}"
                rows = logits[start:].contiguous()
                save_file({"logits": rows}, args.output / f"{name}.safetensors")
                text = tok.decode(gen, skip_special_tokens=True)
                suite["cases"].append({
                    "name": name,
                    "start": start,
                    "prompt_tokens": ids,
                    "generated_tokens": gen,
                    "text": text,
                    "replay_matches_cache": replay_ok,
                    "logits_file": f"{name}.safetensors",
                })
                seqs.append((name, seq, start, rows))
                print(f"{name}: {text!r} ({len(ids)} prompt from {start}, {len(gen)} generated)", flush=True)
                messages.append({"role": "assistant", "content": text})
                cached = seq
        if args.dtype == "float32" and not args.no_bf16:
            del model
            model = load(ckpt, "bfloat16", args.device)
            suite["hf_bf16"] = bf16_error(model, seqs, args.device)
    (args.output / "suite.json").write_text(json.dumps(suite, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"saved {len(suite['cases'])} cases to {args.output}", flush=True)


def bf16_error(model, seqs, device):
    """Measures Transformers' own bfloat16 logits against the float32 oracle."""
    cases, worst = [], 0.0
    for name, seq, start, ref in seqs:
        got = full_logits(model, seq, device)[start:]
        if not torch.isfinite(got).all():
            raise ValueError(f"{name}: non-finite bfloat16 reference logits")
        d = (got - ref).abs()
        row_max = d.max(-1).values
        top1 = (got.argmax(-1) == ref.argmax(-1))
        cases.append({
            "name": name,
            "rows": len(ref),
            "max_abs": float(row_max.max()),
            "median_row_max_abs": float(row_max.median()),
            "top1_agree": int(top1.sum()),
            "row_max_abs": [float(v) for v in row_max],
        })
        worst = max(worst, float(row_max.max()))
        print(f"bf16 {name}: max_abs={cases[-1]['max_abs']:.4f} top1={cases[-1]['top1_agree']}/{len(ref)}", flush=True)
    return {"max_abs": worst, "cases": cases}


if __name__ == "__main__":
    main()
