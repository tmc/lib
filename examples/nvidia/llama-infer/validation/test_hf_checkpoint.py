"""CPU-only provenance checks; does not import torch or run inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hf_checkpoint import completed_reference, file_digests, revision


class CheckpointTests(unittest.TestCase):
    def test_content_and_cache_identity_are_distinct(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            ckpt = root / "snapshots" / "revision"
            ckpt.mkdir(parents=True)
            config = b'{"hidden_size": 8}\n'
            (ckpt / "config.json").write_bytes(config)
            blob = root / ("a" * 64)
            blob.write_bytes(b"contents intentionally differ from blob name")
            (ckpt / "model.safetensors").symlink_to(blob)
            values, sources = file_digests(ckpt)
            self.assertEqual(values["config.json"], hashlib.sha256(config).hexdigest())
            self.assertEqual(sources["config.json"], "sha256_content")
            self.assertEqual(values["model.safetensors"], "a" * 64)
            self.assertEqual(sources["model.safetensors"], "hf_cache_blob_name")
            self.assertEqual(revision(ckpt), "revision")

    def test_nonhex_symlink_name_is_not_a_digest(self):
        with tempfile.TemporaryDirectory() as work:
            ckpt = Path(work)
            (ckpt / "config.json").write_bytes(b"{}")
            data = b"weights"
            blob = ckpt / ("g" * 64)
            blob.write_bytes(data)
            (ckpt / "model.safetensors").symlink_to(blob)
            values, sources = file_digests(ckpt)
            self.assertEqual(values["model.safetensors"], hashlib.sha256(data).hexdigest())
            self.assertEqual(sources["model.safetensors"], "sha256_content")

    def test_index_is_hashed_and_shared_shard_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as work:
            ckpt = Path(work)
            (ckpt / "config.json").write_bytes(b"{}")
            index = json.dumps({"weight_map": {"a": "part.safetensors", "b": "part.safetensors"}}).encode()
            (ckpt / "model.safetensors.index.json").write_bytes(index)
            (ckpt / "part.safetensors").write_bytes(b"weights")
            values, sources = file_digests(ckpt)
            self.assertEqual(len(values), 3)
            self.assertEqual(values["model.safetensors.index.json"], hashlib.sha256(index).hexdigest())
            self.assertEqual(set(sources.values()), {"sha256_content"})


class ReferencePhaseTests(unittest.TestCase):
    def fixture(self, work):
        root = Path(work)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        out = root / "out"
        (out / "ref").mkdir(parents=True)
        phase = out / "reference"
        phase.mkdir()
        suite = {"checkpoint": str(checkpoint), "cases": [{"name": "case"}]}
        (out / "ref/suite.json").write_text(json.dumps(suite))
        return checkpoint, out, phase

    def test_orphaned_suite_is_not_success(self):
        with tempfile.TemporaryDirectory() as work:
            checkpoint, out, _ = self.fixture(work)
            with self.assertRaisesRegex(ValueError, "PASS"):
                completed_reference(out, checkpoint)

    def test_failed_or_missing_exit_receipt_is_not_success(self):
        with tempfile.TemporaryDirectory() as work:
            checkpoint, out, phase = self.fixture(work)
            (phase / "PASS").write_text("stale marker")
            with self.assertRaisesRegex(ValueError, "exit-code"):
                completed_reference(out, checkpoint)
            for code in ["1", "124", "", "not a return code"]:
                (phase / "exit-code.txt").write_text(code)
                with self.assertRaisesRegex(ValueError, "successfully"):
                    completed_reference(out, checkpoint)

    def test_checkpoint_join_and_canonical_alias(self):
        with tempfile.TemporaryDirectory() as work:
            checkpoint, out, phase = self.fixture(work)
            (phase / "PASS").write_text("completed")
            (phase / "exit-code.txt").write_text("0\n")
            self.assertEqual(len(completed_reference(out, checkpoint)["cases"]), 1)
            alias = Path(work) / "alias"
            alias.symlink_to(checkpoint, target_is_directory=True)
            self.assertEqual(len(completed_reference(out, alias)["cases"]), 1)
            other = Path(work) / "other"
            other.mkdir()
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                completed_reference(out, other)


if __name__ == "__main__":
    unittest.main()
