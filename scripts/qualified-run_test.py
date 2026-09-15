"""Deterministic controls for the measurement wrapper; no model or GPU needed."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class QualifiedRunTest(unittest.TestCase):
    def setUp(self):
        root = Path.home() / "tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="qualified-run-test-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.qualifier = self.root / "qualifier"
        self.qualifier.write_text('''#!/bin/bash
n=0
[ ! -f "$COUNT" ] || read -r n < "$COUNT"
n=$((n + 1))
printf '%s\\n' "$n" > "$COUNT"
[ "$n" != "$FAIL_AT" ]
''')
        self.qualifier.chmod(0o700)
        self.runner = Path(os.environ.get("QUALIFIED_RUN", Path(__file__).with_name("qualified-run.sh"))).resolve()

    def run_case(self, fail_at="0", command=None, extra=(), want=0):
        env = dict(os.environ, COUNT=str(self.root / "count"), FAIL_AT=fail_at)
        args = ["bash", str(self.runner), "-q", str(self.qualifier), "-o",
                str(self.root / "receipt"), "-n", "1", "-w", "0", *extra,
                "--", *(command or ["true"])]
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, want, result.stderr)
        return self.root / "receipt"

    def test_success(self):
        receipt = self.run_case()
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "QUALIFIED\n")
        result = subprocess.run(["shasum", "-a", "256", "-c", str(receipt / "MANIFEST.sha256")], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_initial_timeout_keeps_manifest(self):
        receipt = self.run_case(fail_at="1", want=1)
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "SKIPPED\n")
        self.assertTrue((receipt / "MANIFEST.sha256").exists())
        self.assertFalse((receipt / "rep1/exit.txt").exists())

    def test_before_failure_does_not_launch(self):
        marker = self.root / "launched"
        receipt = self.run_case(fail_at="1", command=["touch", str(marker)], want=1)
        self.assertFalse(marker.exists())
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "SKIPPED\n")

    def test_after_failure(self):
        receipt = self.run_case(fail_at="2", want=1)
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "DISQUALIFIED\n")
        self.assertEqual((receipt / "summary.txt").read_text(),
                         "reps=1 qualified=0 disqualified=1 unattributable=0 failed=0 skipped=0\n")

    def test_disqualified_rep_fails_a_multi_rep_run(self):
        # One qualified repetition used to be enough to exit 0 while another
        # ran on a contended host: the disqualified rep was recorded in its
        # own verdict file and counted nowhere.
        env = dict(os.environ, COUNT=str(self.root / "count"), FAIL_AT="4")
        args = ["bash", str(self.runner), "-q", str(self.qualifier), "-o",
                str(self.root / "receipt"), "-n", "2", "-w", "0", "--", "true"]
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        receipt = self.root / "receipt"
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "QUALIFIED\n")
        self.assertEqual((receipt / "rep2/verdict.txt").read_text(), "DISQUALIFIED\n")
        self.assertEqual((receipt / "summary.txt").read_text(),
                         "reps=2 qualified=1 disqualified=1 unattributable=0 failed=0 skipped=0\n")

    def test_identity_change_during_the_run_is_unattributable(self):
        # An identity file was hashed once, before the run. A source or binary
        # edited while the work was running -- routine in a shared worktree --
        # left a receipt naming code that may not be what was measured.
        source = self.root / "source.go"
        source.write_text("before\n")
        receipt = self.run_case(
            command=["bash", "-c", f"printf after > {source}"],
            extra=["-i", str(source)], want=1)
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "UNATTRIBUTABLE\n")
        self.assertEqual((receipt / "summary.txt").read_text(),
                         "reps=1 qualified=0 disqualified=0 unattributable=1 failed=0 skipped=0\n")
        # The before and after hashes are both retained, so the receipt shows
        # what moved rather than only that something did.
        self.assertNotEqual((receipt / "identity.sha256").read_text(),
                            (receipt / "rep1/identity-after.sha256").read_text())

    def test_identity_removed_during_the_run_is_unattributable(self):
        source = self.root / "source.go"
        source.write_text("before\n")
        receipt = self.run_case(
            command=["rm", str(source)], extra=["-i", str(source)], want=1)
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "UNATTRIBUTABLE\n")

    def test_unchanged_identity_still_qualifies(self):
        source = self.root / "source.go"
        source.write_text("before\n")
        receipt = self.run_case(extra=["-i", str(source)])
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "QUALIFIED\n")
        self.assertEqual((receipt / "identity.sha256").read_text(),
                         (receipt / "rep1/identity-after.sha256").read_text())

    def test_relative_identity_path_qualifies(self):
        # shasum writes the path beside the hash, and the comparison is over
        # the whole line. This script hashes before and after from one cwd, so
        # a spelling mismatch is not reachable here; it was reachable in a
        # sibling wrapper on the CUDA host, which reported UNATTRIBUTABLE --
        # the strongest verdict this class of check has -- for a clean run
        # whose paths were relative before and absolute after.
        #
        # So what this pins is the weaker, real property: a relative -i path
        # qualifies, and the receipt names the file absolutely, which is what
        # keeps the spelling out of the comparison and the receipt readable
        # from somewhere other than the cwd the run used.
        source = self.root / "source.go"
        source.write_text("before\n")
        env = dict(os.environ, COUNT=str(self.root / "count"), FAIL_AT="0")
        args = ["bash", str(self.runner), "-q", str(self.qualifier), "-o",
                str(self.root / "receipt"), "-n", "1", "-w", "0",
                "-i", "source.go", "--", "true"]
        result = subprocess.run(args, env=env, cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = self.root / "receipt"
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "QUALIFIED\n")
        # The receipt names the file absolutely, so it is readable later from
        # anywhere rather than only from the cwd the run happened to use.
        self.assertIn(str(source), (receipt / "identity.sha256").read_text())

    def test_relative_identity_path_still_detects_a_change(self):
        # Resolving the path must not cost the detection it exists for.
        source = self.root / "source.go"
        source.write_text("before\n")
        env = dict(os.environ, COUNT=str(self.root / "count"), FAIL_AT="0")
        args = ["bash", str(self.runner), "-q", str(self.qualifier), "-o",
                str(self.root / "receipt"), "-n", "1", "-w", "0",
                "-i", "source.go", "--",
                "bash", "-c", f"printf after > {source}"]
        result = subprocess.run(args, env=env, cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual((self.root / "receipt/rep1/verdict.txt").read_text(),
                         "UNATTRIBUTABLE\n")

    def test_identity_change_outranks_a_command_failure(self):
        # Both are true; the receipt must say the run is unattributable,
        # because a failure whose inputs moved is not evidence about the code.
        source = self.root / "source.go"
        source.write_text("before\n")
        receipt = self.run_case(
            command=["bash", "-c", f"printf after > {source}; exit 7"],
            extra=["-i", str(source)], want=1)
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "UNATTRIBUTABLE\n")

    def test_unattributable_rep_fails_a_multi_rep_run(self):
        # The exit predicate needs its own term for this: with one rep, the
        # run already fails because nothing qualified. It takes a second rep
        # that did qualify to show the counter carrying the verdict.
        source = self.root / "source.go"
        source.write_text("before\n")
        marker = self.root / "marker"
        env = dict(os.environ, COUNT=str(self.root / "count"), FAIL_AT="0")
        args = ["bash", str(self.runner), "-q", str(self.qualifier), "-o",
                str(self.root / "receipt"), "-n", "2", "-w", "0",
                "-i", str(source), "--", "bash", "-c",
                f"if [ -f {marker} ]; then printf after > {source}; else touch {marker}; fi"]
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        receipt = self.root / "receipt"
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "QUALIFIED\n")
        self.assertEqual((receipt / "rep2/verdict.txt").read_text(), "UNATTRIBUTABLE\n")
        self.assertEqual((receipt / "summary.txt").read_text(),
                         "reps=2 qualified=1 disqualified=0 unattributable=1 failed=0 skipped=0\n")

    def test_command_failure(self):
        receipt = self.run_case(command=["bash", "-c", "exit 7"], want=1)
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "FAILED\n")
        self.assertEqual((receipt / "rep1/exit.txt").read_text(), "7\n")

    def test_missing_identity_prevents_launch(self):
        marker = self.root / "launched"
        self.run_case(command=["touch", str(marker)], extra=["-i", str(self.root / "absent")], want=1)
        self.assertFalse(marker.exists())

    def test_identity_hash_failure_prevents_launch(self):
        marker = self.root / "launched"
        self.run_case(command=["touch", str(marker)], extra=["-i", str(self.root)], want=1)
        self.assertFalse(marker.exists())

    def test_output_file_prevents_launch(self):
        (self.root / "receipt").write_text("preserve me")
        marker = self.root / "launched"
        self.run_case(command=["touch", str(marker)], want=1)
        self.assertFalse(marker.exists())
        self.assertEqual((self.root / "receipt").read_text(), "preserve me")

    def test_existing_receipt_is_preserved(self):
        receipt = self.run_case()
        manifest = (receipt / "MANIFEST.sha256").read_bytes()
        self.run_case(want=1)
        self.assertEqual((receipt / "MANIFEST.sha256").read_bytes(), manifest)

    def test_invalid_repetition_count(self):
        self.run_case(extra=["-n", "no"], want=2)
        self.assertFalse((self.root / "receipt").exists())

    def test_partial_success_is_failure(self):
        receipt = self.run_case(fail_at="3", extra=["-n", "2"], want=1)
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "QUALIFIED\n")
        self.assertEqual((receipt / "rep2/verdict.txt").read_text(), "SKIPPED\n")



class MultipleQualifiersTest(unittest.TestCase):
    """-q may be repeated; the host is qualified only when every gate passes.

    A load-average gate and a memory gate refuse for different reasons, and
    composing them through a wrapper script would put the composition outside
    the tested tooling.
    """

    def setUp(self):
        root = Path.home() / "tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="qualified-run-multi-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runner = Path(os.environ.get("QUALIFIED_RUN", Path(__file__).with_name("qualified-run.sh"))).resolve()

    def gate(self, name, exit_code):
        path = self.root / name
        path.write_text("#!/bin/bash\n"
                        "n=$(basename \"$0\")\n"
                        "echo \"$n ran\"\n"
                        "echo \"$n\" >> \"$RAN\"\n"
                        "exit " + str(exit_code) + "\n")
        path.chmod(0o700)
        return path

    def run_case(self, gates, want=0, command=None):
        env = dict(os.environ, RAN=str(self.root / "ran"))
        args = ["bash", str(self.runner)]
        for g in gates:
            args += ["-q", str(g)]
        args += ["-o", str(self.root / "receipt"), "-n", "1", "-w", "0",
                 "--", *(command or ["true"])]
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, want, result.stdout + result.stderr)
        return self.root / "receipt"

    def ran(self):
        path = self.root / "ran"
        return path.read_text().split() if path.exists() else []

    def test_all_passing_qualifies(self):
        receipt = self.run_case([self.gate("load", 0), self.gate("memory", 0)])
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "QUALIFIED\n")
        before = (receipt / "rep1/qualify-before.stdout").read_text()
        self.assertIn("load ran", before)
        self.assertIn("memory ran", before)

    def test_second_gate_refusing_blocks_the_run(self):
        marker = self.root / "launched"
        receipt = self.run_case([self.gate("load", 0), self.gate("memory", 1)],
                                want=1, command=["touch", str(marker)])
        self.assertFalse(marker.exists())
        self.assertEqual((receipt / "rep1/verdict.txt").read_text(), "SKIPPED\n")
        self.assertIn("memory", (receipt / "rep1/qualify-before.stdout").read_text())

    def test_a_refusal_short_circuits_the_rest(self):
        # The gates are tried in the order given so the cheap filter can be
        # listed first; running the expensive one after a refusal would spend
        # the sampling window this ordering exists to save.
        self.run_case([self.gate("first", 1), self.gate("second", 0)], want=1)
        self.assertNotIn("second", self.ran())

    def test_qualifier_output_does_not_accumulate_across_checks(self):
        # qualify() truncates before each attempt; appending would grow the
        # receipt without bound on a host that waits out many checks.
        receipt = self.run_case([self.gate("load", 0)])
        self.assertEqual((receipt / "rep1/qualify-before.stdout").read_text().count("load ran"), 1)


if __name__ == "__main__":
    unittest.main()
