import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get("TMPDIR", str(Path.home() / "tmp")))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref = self.write("ref", [3, 1, 0], False)
        self.candidate = self.write("candidate", [2.5, 1, 0], True)

    def write(self, name, values, forced):
        directory = self.root / name
        directory.mkdir()
        raw = np.asarray(values, dtype="<f4").tobytes()
        (directory / "logits-0.f32").write_bytes(raw)
        case = {"prompt": "test", "input_ids": [1], "generated_ids": [int(np.argmax(values))],
                "logits": "logits-0.f32", "logits_sha256": hashlib.sha256(raw).hexdigest()}
        data = {"files": {"model.safetensors": "test-fixture"}, "vocab": 3, "capacity": 2,
                "compute_dtype": "bfloat16" if forced else "float32", "teacher_forced": forced,
                "cases": [case]}
        if forced:
            case["teacher_ids"] = [0]
            data["teacher_reference_sha256"] = compare.sha256(self.ref / "reference.json")
        (directory / "reference.json").write_text(json.dumps(data))
        return directory

    def mutate(self, change):
        path = self.candidate / "reference.json"
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))

    def test_metrics(self):
        result = compare.evaluate(self.ref, self.candidate)
        self.assertEqual((result["logits"], result["greedy_matches"]), (3, 1))
        self.assertAlmostEqual(result["rmse"], 0.5 / math.sqrt(3))
        self.assertEqual(result["max_abs"], 0.5)

    def test_context_mismatch(self):
        self.mutate(lambda d: d["cases"][0].update(teacher_ids=[1]))
        with self.assertRaisesRegex(ValueError, "histories differ"):
            compare.evaluate(self.ref, self.candidate)

    def test_nonfinite(self):
        bad = self.write("nan", [float("nan"), 1, 0], True)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            compare.evaluate(self.ref, bad)

    def test_hash_mismatch(self):
        self.mutate(lambda d: d["cases"][0].update(logits_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            compare.evaluate(self.ref, self.candidate)

    def test_choice_mismatch(self):
        self.mutate(lambda d: d["cases"][0].update(generated_ids=[2]))
        with self.assertRaisesRegex(ValueError, "choices disagree"):
            compare.evaluate(self.ref, self.candidate)

    def test_failed_gate_reports_and_exits(self):
        better = self.write("baseline", [3, 1, 0], True)
        result = subprocess.run([sys.executable, str(Path(compare.__file__)), str(self.ref),
                                 str(self.candidate), "--baseline", str(better), "--require-not-worse"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["criteria"]["rmse"])
        self.assertTrue(report["criteria"]["greedy_matches"])


if __name__ == "__main__":
    unittest.main()
