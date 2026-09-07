#!/usr/bin/env python3
"""Model-free controls for compare-prefill.py; fixtures live under ~/tmp."""

import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).with_name("compare-prefill.py")
SPEC = importlib.util.spec_from_file_location("compare_prefill", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def write_json(path, data):
    path.write_text(json.dumps(data))


class CompareTest(unittest.TestCase):
    def setUp(self):
        root = Path.home() / "tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="compare-prefill-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.before = Path(self.temp.name) / "before"
        self.after = Path(self.temp.name) / "after"
        for path, executable in ((self.before, "a"), (self.after, "b")):
            path.mkdir()
            tokens = json.dumps([1] * 2048).encode()
            (path / "tokens.json").write_bytes(tokens)
            write_json(path / "identity.json", {
                "schema": 1, "backend": "metal", "goos": "darwin", "goarch": "arm64",
                "context": 2049, "cases": [2, 64, 2048], "continuation_token": 1,
                "tokens_sha256": compare.digest(tokens), "lifecycle": compare.LIFECYCLE,
                "model_sha256": "c" * 64, "model_bytes": 86720111488,
                "executable_sha256": executable * 64,
            })
            (path / "COMPLETE").write_bytes(compare.COMPLETE)
            for n in compare.CASES:
                for stage in ("prefill", "continuation"):
                    self.payload(path, f"n{n}-{stage}", bytes(compare.VOCAB * 4))

    def payload(self, path, name, data):
        (path / (name + ".f32le")).write_bytes(data)
        write_json(path / (name + ".json"), {
            "vocab": compare.VOCAB, "bytes": len(data), "sha256": compare.digest(data),
        })

    def identity(self, path, **changes):
        data = json.loads((path / "identity.json").read_bytes())
        data.update(changes)
        write_json(path / "identity.json", data)

    def test_valid(self):
        result = compare.compare(self.before, self.after)
        self.assertEqual(result["result"], "EXACT")
        self.assertEqual(result["vectors"], 6)
        self.assertNotEqual(result["before_executable_sha256"], result["after_executable_sha256"])

    def test_changed_bits(self):
        for word in (1, 0x80000000):
            with self.subTest(word=word):
                self.payload(self.after, "n2048-continuation", struct.pack("<I", word) + bytes(compare.VOCAB * 4 - 4))
                with self.assertRaisesRegex(compare.CaptureError, "bit mismatch"):
                    compare.compare(self.before, self.after)

    def test_missing_complete(self):
        (self.after / "COMPLETE").unlink()
        with self.assertRaises(OSError):
            compare.compare(self.before, self.after)

    def test_missing_case(self):
        (self.after / "n64-prefill.f32le").unlink()
        with self.assertRaises(OSError):
            compare.compare(self.before, self.after)

    def test_corrupted_sidecar(self):
        path = self.after / "n2-prefill.json"
        original = json.loads(path.read_bytes())
        for field, bad in (("sha256", "0" * 64), ("vocab", 1), ("bytes", 4)):
            with self.subTest(field=field):
                write_json(path, dict(original, **{field: bad}))
                with self.assertRaises(compare.CaptureError):
                    compare.compare(self.before, self.after)

    def test_identity_mismatch(self):
        original = (self.after / "identity.json").read_bytes()
        changes = {
            "backend": "cuda", "goos": "linux", "goarch": "amd64",
            "model_sha256": "d" * 64, "model_bytes": 10, "continuation_token": 2,
            "context": 2050, "lifecycle": "cold", "cases": [64, 2, 2048],
        }
        for field, bad in changes.items():
            with self.subTest(field=field):
                (self.after / "identity.json").write_bytes(original)
                self.identity(self.after, **{field: bad})
                with self.assertRaises(compare.CaptureError):
                    compare.compare(self.before, self.after)

    def test_token_digest_and_identity(self):
        data = json.dumps([2] * 2048).encode()
        (self.after / "tokens.json").write_bytes(data)
        with self.assertRaisesRegex(compare.CaptureError, "tokens digest"):
            compare.compare(self.before, self.after)
        self.identity(self.after, tokens_sha256=compare.digest(data))
        with self.assertRaisesRegex(compare.CaptureError, "tokens_sha256"):
            compare.compare(self.before, self.after)

    def test_nonfinite(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                data = struct.pack("<f", value) + bytes(compare.VOCAB * 4 - 4)
                for path in (self.before, self.after):
                    self.payload(path, "n2-prefill", data)
                with self.assertRaisesRegex(compare.CaptureError, "non-finite"):
                    compare.compare(self.before, self.after)

    def test_same_directory(self):
        with self.assertRaisesRegex(compare.CaptureError, "self-comparison"):
            compare.compare(self.before, self.before)
        alias = Path(self.temp.name) / "alias"
        alias.symlink_to(self.before, target_is_directory=True)
        with self.assertRaisesRegex(compare.CaptureError, "self-comparison"):
            compare.compare(self.before, alias)

    def test_scope_and_cli(self):
        for path in (self.before, self.after):
            self.identity(path, cases=[2])
        with self.assertRaises(compare.CaptureError):
            compare.compare(self.before, self.after)
        result = compare.compare(self.before, self.after, (2,))
        self.assertIn("SCOPED", result["scope"])
        proc = subprocess.run([sys.executable, str(SCRIPT), str(self.before), str(self.after), "--cases", "2"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("SCOPED", json.loads(proc.stdout)["scope"])
        proc = subprocess.run([sys.executable, str(SCRIPT), str(self.before), str(self.after)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
