import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("validator", ROOT / "check-oracle-workload.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class TestCapture(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text("fixture prompt")
        corpus = (ROOT / "testdata" / "oracle-workload-tokens2048.json").read_bytes()
        tokens = json.loads(corpus)
        self.prefix = struct.pack("<2048I", *tokens)
        self.assertEqual(v.digest(self.prefix), v.PREFIX_SHA256)
        self.identity = {
            "backend": "cuda", "host": "fixture", "model_sha256": v.MODEL_SHA256,
            "executable_sha256": "a" * 64, "context": 65665,
            "token_ids_encoding": "little-endian uint32", "token_ids_count": 2048,
            "token_ids_sha256": v.PREFIX_SHA256, "eos": 1,
            "prompt_sha256": v.digest(self.prompt.read_bytes()), "goos": "linux", "goarch": "arm64",
            "tokens_sha256": v.digest(corpus), "tokenizer_discrepancy_sha256": v.digest(b"{}"),
        }
        for key in ("input_contract", "eos_provenance", "tokenizer_discrepancy_receipt", "lifecycle", "scope"):
            self.identity[key] = "synthetic control, not physical evidence"
        self.result = {"prefill_tokens": 2048, "generated_tokens": [2] * 128,
                       "prefill_seconds": 2, "generation_seconds": 2,
                       "first_seconds": .01, "steady_seconds": 1.27,
                       "prefill_logits_sha256": "b" * 64, "final_logits_sha256": "c" * 64}
        (self.root / "tokens.json").write_bytes(corpus)
        (self.root / "token-prefix.u32le").write_bytes(self.prefix)
        (self.root / "tokenizer-discrepancy.json").write_bytes(b"{}")
        (self.root / "COMPLETE").write_bytes(v.COMPLETE)
        self.write()

    def write(self):
        (self.root / "identity.json").write_text(json.dumps(self.identity))
        (self.root / "result.json").write_text(json.dumps(self.result))

    def check(self):
        return v.check(self.root, "cuda", "fixture", "a" * 64, self.prompt)

    def test_positive(self):
        result = self.check()
        self.assertEqual(result["result"], "VALID_ARTIFACTS")
        self.assertEqual(result["generation_tokens_per_second_including_selection"], 64)
        self.assertEqual(result["steady_tokens_per_second_excluding_selection"], 100)

    def test_missing_complete(self):
        (self.root / "COMPLETE").unlink()
        with self.assertRaises((ValueError, OSError)):
            self.check()

    def test_empty_complete(self):
        (self.root / "COMPLETE").write_bytes(b"")
        with self.assertRaises(ValueError):
            self.check()

    def test_identity_mismatches(self):
        original = copy.deepcopy(self.identity)
        for key, value in (("host", "wrong"), ("backend", "metal"), ("executable_sha256", "d"*64),
                           ("model_sha256", "d"*64), ("context", 2049), ("eos", True),
                           ("token_ids_count", 2047), ("prompt_sha256", "d"*64)):
            with self.subTest(key=key):
                self.identity = dict(original, **{key: value})
                self.write()
                with self.assertRaises(ValueError):
                    self.check()

    def test_durations(self):
        original = dict(self.result)
        for key in ("prefill_seconds", "generation_seconds", "first_seconds", "steady_seconds"):
            for value in (0, -1, True, float("nan"), float("inf"), "1"):
                with self.subTest(key=key, value=value):
                    self.result = dict(original, **{key: value})
                    self.write()
                    with self.assertRaises(ValueError):
                        self.check()

    def test_overflowing_rate(self):
        for key in ("prefill_seconds", "steady_seconds"):
            original = dict(self.result)
            self.result[key] = 1e-320
            self.write()
            with self.assertRaises(ValueError):
                self.check()
            self.result = original

    def test_metal_identity(self):
        self.identity.update(backend="metal", goos="darwin", host="metal-fixture")
        self.write()
        result = v.check(self.root, "metal", "metal-fixture", "a" * 64, self.prompt)
        self.assertEqual(result["backend"], "metal")
        self.identity["goos"] = "linux"
        self.write()
        with self.assertRaises(ValueError):
            v.check(self.root, "metal", "metal-fixture", "a" * 64, self.prompt)

    def test_short_generation_window(self):
        self.result["generation_seconds"] = 1
        self.write()
        with self.assertRaises(ValueError):
            self.check()

    def test_generated(self):
        for tokens in ([2]*127, [2]*129, [1]*128, [True]*128, [129280]*128):
            with self.subTest(tokens=tokens[:2]):
                self.result["generated_tokens"] = tokens
                self.write()
                with self.assertRaises(ValueError):
                    self.check()

    def test_duplicate_key(self):
        (self.root / "result.json").write_text('{"prefill_tokens":1,' + json.dumps(self.result)[1:])
        with self.assertRaises(ValueError):
            self.check()

    def test_prefix_changed_consistently(self):
        tokens = json.loads((self.root / "tokens.json").read_bytes())
        tokens[7] ^= 1
        corpus = json.dumps(tokens).encode()
        (self.root / "tokens.json").write_bytes(corpus)
        (self.root / "token-prefix.u32le").write_bytes(struct.pack("<2048I", *tokens))
        self.identity["tokens_sha256"] = v.digest(corpus)
        self.write()
        with self.assertRaises(ValueError):
            self.check()

    def test_binary_prefix(self):
        (self.root / "token-prefix.u32le").write_bytes(self.prefix[:-4])
        with self.assertRaises(ValueError):
            self.check()

    def test_logit_hash(self):
        self.result["final_logits_sha256"] = ""
        self.write()
        with self.assertRaises(ValueError):
            self.check()

    def test_discrepancy(self):
        (self.root / "tokenizer-discrepancy.json").write_bytes(b"")
        with self.assertRaises(ValueError):
            self.check()


if __name__ == "__main__":
    unittest.main()
