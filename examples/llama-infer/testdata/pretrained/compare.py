"""Compare teacher-forced output matrices against a free-running F32 reference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix(directory, case, shape):
    name = case["logits"]
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("invalid logit filename")
    path = directory / name
    if path.stat().st_size != shape[0] * shape[1] * 4:
        raise ValueError(f"{path}: logit shape differs")
    if sha256(path) != case["logits_sha256"]:
        raise ValueError(f"{path}: logit hash differs")
    values = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: nonfinite logits")
    return values


def evaluate(reference_dir, output_dir):
    reference_path = reference_dir / "reference.json"
    reference = json.loads(reference_path.read_text())
    output = json.loads((output_dir / "reference.json").read_text())
    if reference.get("teacher_forced", False) or reference.get("compute_dtype", "float32") != "float32":
        raise ValueError("reference must be free-running F32")
    if not output.get("teacher_forced", False) or output.get("teacher_reference_sha256") != sha256(reference_path):
        raise ValueError("output does not identify this teacher reference")
    dtype = output.get("compute_dtype")
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("unknown output dtype")
    if not reference["files"] or output["files"] != reference["files"]:
        raise ValueError("checkpoint hashes differ")
    if output["vocab"] != reference["vocab"] or output["capacity"] != reference["capacity"]:
        raise ValueError("vocabulary or capacity differs")
    if not reference["cases"] or len(output["cases"]) != len(reference["cases"]):
        raise ValueError("reference cases differ or are empty")
    if any(type(reference[k]) is not int or reference[k] <= 0 for k in ("vocab", "capacity")):
        raise ValueError("invalid vocabulary or capacity")
    cases = []
    count, steps, matches, abs_sum, square_sum, maximum = 0, 0, 0, 0.0, 0.0, 0.0
    for ref, got in zip(reference["cases"], output["cases"]):
        teacher = ref["generated_ids"]
        if (ref["prompt"] != got["prompt"] or ref["input_ids"] != got["input_ids"]
                or got.get("teacher_ids") != teacher):
            raise ValueError("input or teacher histories differ")
        if not teacher or not ref["input_ids"] or reference["vocab"] <= 0:
            raise ValueError("empty reference case")
        for ids in (ref["input_ids"], teacher, got["generated_ids"]):
            if any(type(v) is not int or not 0 <= v < reference["vocab"] for v in ids):
                raise ValueError("token ID outside vocabulary")
        shape = (len(teacher), reference["vocab"])
        want = matrix(reference_dir, ref, shape)
        values = matrix(output_dir, got, shape)
        if want.argmax(axis=1).tolist() != teacher:
            raise ValueError("reference choices disagree with logits")
        if values.argmax(axis=1).tolist() != got["generated_ids"]:
            raise ValueError("output choices disagree with logits")
        delta = values.astype(np.float64) - want.astype(np.float64)
        n = int(values.size)
        a, s, m = float(np.abs(delta).sum()), float(np.square(delta).sum()), float(np.abs(delta).max())
        matched = int((values.argmax(axis=1) == np.asarray(teacher)).sum())
        cases.append({"prompt": ref["prompt"], "logits": n, "steps": shape[0], "greedy_matches": matched,
                      "mae": a / n, "rmse": (s / n)**0.5, "max_abs": m})
        count += n; steps += shape[0]; matches += matched
        abs_sum += a; square_sum += s; maximum = max(maximum, m)
    return {"directory": str(output_dir), "dtype": dtype, "logits": count, "steps": steps,
            "greedy_matches": matches, "mae": abs_sum / count, "rmse": (square_sum / count)**0.5,
            "max_abs": maximum, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--require-not-worse", action="store_true",
                        help="require RMSE <= baseline and greedy matches >= baseline")
    args = parser.parse_args()
    if args.require_not_worse and args.baseline is None:
        parser.error("--require-not-worse requires --baseline")
    try:
        candidate = evaluate(args.reference, args.candidate)
        report = {"candidate": candidate}
        if args.baseline:
            baseline = evaluate(args.reference, args.baseline)
            if candidate["dtype"] != baseline["dtype"]:
                raise ValueError("candidate and baseline dtypes differ")
            report["baseline"] = baseline
        if args.require_not_worse:
            report["criteria"] = {"rmse": candidate["rmse"] <= baseline["rmse"],
                                  "greedy_matches": candidate["greedy_matches"] >= baseline["greedy_matches"]}
        print(json.dumps(report, indent=2))
        if args.require_not_worse and not all(report["criteria"].values()):
            raise SystemExit(1)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
