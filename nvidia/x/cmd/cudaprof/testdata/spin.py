"""Launch the same native control from Python; ranges belong to its thread."""
import subprocess
import sys

if len(sys.argv) < 2:
    sys.exit("usage: spin.py NATIVE_SPIN ARGS...")
try:
    result = subprocess.run(sys.argv[1:], check=False)
except OSError as error:
    sys.exit(f"spin.py: {error}")
sys.exit(result.returncode if result.returncode >= 0 else 128 - result.returncode)
