"""Controls for the load-gate wrapper; the gate binary is injected, no host sampling."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LoadQualifyTest(unittest.TestCase):
    def setUp(self):
        root = Path.home() / "tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="load-qualify-test-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = Path(__file__).with_name("load-qualify.sh").resolve()

    def fake(self, exit_code=0, executable=True):
        # Records the argv it was called with, so the test can assert which
        # limit reached the gate rather than only whether it passed.
        path = self.root / "hostqualify"
        path.write_text('#!/bin/bash\necho "$@" > "$ARGV"\nexit %d\n' % exit_code)
        path.chmod(0o700 if executable else 0o600)
        return path

    def run_it(self, *flags, env=None, bin=None):
        e = dict(os.environ, ARGV=str(self.root / "argv"))
        e.pop("DS4_MAX_LOAD", None)
        e["HOSTQUALIFY"] = str(bin if bin is not None else self.fake())
        e.update(env or {})
        return subprocess.run(["bash", str(self.script), *flags], env=e,
                              capture_output=True, text=True)

    def argv(self):
        return (self.root / "argv").read_text().strip()

    def test_passes_the_registered_limit_through(self):
        r = self.run_it("-max-load", "8.00")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.argv(), "-max-load 8.00")
        self.assertIn("-max-load 8.00", r.stdout)

    def test_environment_supplies_the_limit(self):
        r = self.run_it(env={"DS4_MAX_LOAD": "4.00"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.argv(), "-max-load 4.00")

    def test_no_limit_refuses_and_does_not_run_the_gate(self):
        # hostqualify's own -max-load default is NumCPU/4. Omitting the flag
        # would enforce a limit nobody registered and record a pass against it,
        # so the absence of a limit is an error, not a default.
        r = self.run_it()
        self.assertEqual(r.returncode, 2)
        self.assertFalse((self.root / "argv").exists())
        self.assertIn("no default", r.stderr)

    def test_gate_refusal_is_propagated(self):
        r = self.run_it("-max-load", "8.00", bin=self.fake(exit_code=1))
        self.assertEqual(r.returncode, 1)

    def test_missing_binary_is_an_error_not_a_pass(self):
        r = self.run_it("-max-load", "8.00", bin=self.root / "absent")
        self.assertEqual(r.returncode, 2)
        self.assertIn("not executable", r.stderr)

    def test_non_executable_binary_is_an_error_not_a_pass(self):
        r = self.run_it("-max-load", "8.00", bin=self.fake(executable=False))
        self.assertEqual(r.returncode, 2)

    def test_a_non_numeric_or_zero_limit_is_rejected(self):
        self.assertEqual(self.run_it("-max-load", "high").returncode, 2)
        self.assertEqual(self.run_it("-max-load", "0").returncode, 2)
        self.assertFalse((self.root / "argv").exists())


if __name__ == "__main__":
    unittest.main()
