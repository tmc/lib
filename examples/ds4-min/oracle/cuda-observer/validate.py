#!/usr/bin/env python3
"""Apply the pinned oracle overlays in an isolated checkout and test on CPU."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile

PIN = "b0a147a7fba6d1a104d047d5a140e9bb4bfc13cd"
bundle = Path(__file__).resolve().parent


def run(*args, cwd=None, env=None):
    subprocess.run(args, cwd=cwd, env=env, check=True)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: validate.py /path/to/local/antirez/ds4")
    source = Path(sys.argv[1]).resolve()
    scratch = Path.home() / "tmp"
    scratch.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="ds4-observer-check-", dir=scratch))
    print(f"Validation artifacts: {output}", flush=True)
    tree = output / "oracle"
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    run("git", "clone", "--quiet", "--shared", "--no-checkout", str(source), str(tree))
    run("git", "checkout", "--quiet", "--detach", PIN, cwd=tree)
    for name in ("compatibility.patch", "scalar.patch"):
        run("git", "apply", "--check", str(bundle / name), cwd=tree)
        run("git", "apply", str(bundle / name), cwd=tree)
    shutil.copyfile(tree / "ds4_cuda.cu", output / "ds4_cuda.cu")
    shutil.copyfile(bundle / "compatibility.patch", output / "pre-existing.patch")
    run("git", "apply", "--check", str(bundle / "window.patch"), cwd=tree)
    run("git", "apply", str(bundle / "window.patch"), cwd=tree)
    expected = json.loads((bundle / "source-hashes.json").read_text())
    for name, digest in expected.items():
        actual = hashlib.sha256((tree / name).read_bytes()).hexdigest()
        if actual != digest:
            sys.exit(f"source hash mismatch: {name}: {actual}")
    run("git", "diff", "--check", cwd=tree)
    env = dict(os.environ, DS4_OBSERVER_TEST_DIR=str(output))
    for name in ("test-scalar.py", "test-window.py"):
        run(sys.executable, str(bundle / name), cwd=tree, env=env)
    print("PASS pinned overlays, source hashes, C/C++ ABI and synthetic controls; no GPU execution")


if __name__ == "__main__":
    main()
