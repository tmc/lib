"""Controls for the memory qualifier; no model, no GPU, no real host state.

Both host probes are injected, so these tests assert what the script decides
rather than what this machine's memory happens to be doing.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

# 8 GiB of 4096-byte pages, with the free/inactive/purgeable/compressor counts
# as fields the test varies. vm_stat's real preamble line carries the page size
# in the same shape, because that is the line the script parses it out of.
VM_STAT = """Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                              {free}.
Pages active:                            100000.
Pages inactive:                          {inactive}.
Pages speculative:                       1000.
Pages throttled:                         0.
Pages wired down:                        200000.
Pages purgeable:                         {purgeable}.
"Translation faults":                    1000000.
Pages occupied by compressor:            {compressor}.
"""

MEMSIZE = 8 * 1024**3
PAGES_PER_GIB = 1024**3 // 4096


class MemoryQualifyTest(unittest.TestCase):
    def setUp(self):
        root = Path.home() / "tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="memory-qualify-test-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = Path(__file__).with_name("memory-qualify.sh").resolve()

    def qualify(self, *flags, stat=None, env=None, **fields):
        fake = self.root / "vm_stat"
        text = stat if stat is not None else VM_STAT.format(**fields)
        fake.write_text("#!/bin/bash\ncat <<'EOF'\n" + text + "EOF\n")
        fake.chmod(0o700)
        e = dict(os.environ, VM_STAT=str(fake), MEMSIZE_BYTES=str(MEMSIZE))
        e.pop("DS4_MIN_AVAILABLE_GIB", None)
        e.pop("DS4_MAX_COMPRESSOR", None)
        e.update(env or {})
        return subprocess.run(["bash", str(self.script), *flags], env=e,
                              capture_output=True, text=True)

    def roomy(self, **overrides):
        # 6 GiB available, compressor at 1/128 of the host.
        fields = dict(free=4 * PAGES_PER_GIB, inactive=PAGES_PER_GIB,
                      purgeable=PAGES_PER_GIB, compressor=MEMSIZE // 4096 // 128)
        fields.update(overrides)
        return fields

    def test_qualifies_when_both_thresholds_are_met(self):
        r = self.qualify("-min-available", "5", "-max-compressor", "0.05", **self.roomy())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("available 6.00 GiB", r.stdout)

    def test_refuses_on_available_memory(self):
        r = self.qualify("-min-available", "7", "-max-compressor", "0.05", **self.roomy())
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not have memory", r.stderr)

    def test_refuses_on_compressor_share_alone(self):
        # The failure this script exists to prevent: plenty of free memory, but
        # the host is already paging, which is what killed the run that
        # qualified on load average at 5.41 and died ten minutes later.
        crowded = self.roomy(compressor=MEMSIZE // 4096 // 2)
        r = self.qualify("-min-available", "5", "-max-compressor", "0.10", **crowded)
        self.assertEqual(r.returncode, 1)
        self.assertIn("compressor 0.5000", r.stdout)

    def test_environment_supplies_the_thresholds(self):
        # qualified-run.sh invokes a qualifier with no arguments, so the
        # thresholds have to be reachable without flags or this composes only
        # through a wrapper script.
        env = {"DS4_MIN_AVAILABLE_GIB": "7", "DS4_MAX_COMPRESSOR": "0.05"}
        self.assertEqual(self.qualify(env=env, **self.roomy()).returncode, 1)
        env["DS4_MIN_AVAILABLE_GIB"] = "5"
        self.assertEqual(self.qualify(env=env, **self.roomy()).returncode, 0)

    def test_flags_beat_the_environment(self):
        env = {"DS4_MIN_AVAILABLE_GIB": "7"}
        r = self.qualify("-min-available", "5", "-max-compressor", "0.05", env=env, **self.roomy())
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_missing_page_counts_are_an_error_not_a_pass(self):
        # An unparsed field used to read as zero, which qualifies a host with no
        # memory at all -- the exact shape of a gate that passes without
        # checking. Exit 2 is neither qualified nor "wait for a quieter host".
        stat = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 100.\n"
        r = self.qualify("-min-available", "0", "-max-compressor", "1", stat=stat)
        self.assertEqual(r.returncode, 2)
        self.assertIn("could not read page counts", r.stderr)

    def test_a_non_numeric_threshold_is_rejected(self):
        r = self.qualify("-min-available", "lots", **self.roomy())
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
