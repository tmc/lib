#!/usr/bin/env python3
"""Compare the GPU encode-call sequence of two revisions of a Go source file.

A refactor that claims to move no float32 bits must leave untouched the order,
the identity and the arguments of every call that tells the GPU what to do. This
extracts that sequence -- pipeline binding, buffer and scalar binding,
threadgroup memory, dispatch, end-encoding -- from two files and diffs it, whole
file and per enclosing function. Indentation is normalized, because wrapping a
body in a closure reindents it; nothing else is.

A difference is not automatically a defect: an equivalent rewrite (binding two
buffers by a loop instead of two statements) shows up here. The tool localizes
the question to a handful of lines; the kernel source answers it.
"""

import argparse
import json
from pathlib import Path
import re
import sys

# Every apple/metal call through which a command encoder learns what to run.
# Adding a call to this set is how the tool keeps up with the binding API; a
# call that escapes it is a hole, which is why extraction must never come back
# empty (see require_nonvacuous).
ENCODE_CALLS = (
    "SetComputePipelineState",
    "SetBufferWithOffsetAtIndex",
    "SetBytes",
    "SetScalar",
    "SetThreadgroupMemoryLength",
    "DispatchThreads",
    "DispatchThreadgroups",
    "DispatchThreadsThreadsPerThreadgroup",
    "DispatchThreadgroupsThreadsPerThreadgroup",
    "DispatchThreadgroupsWithIndirectBuffer",
    "EndEncoding",
)
CALL = re.compile(r"\b(?:" + "|".join(sorted(ENCODE_CALLS, key=len, reverse=True)) + r")\s*\(")
FUNC = re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(")


class EncodeError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EncodeError(message)


def extract(source, label):
    """Return [(function, call)] in source order, joining calls split over lines."""
    calls, function, pending, pending_function, depth = [], "<file>", None, "<file>", 0
    for number, raw in enumerate(source.splitlines(), start=1):
        line = raw.strip()
        match = FUNC.match(raw)
        if match:
            function = match.group(1)
        if pending is not None:
            pending = f"{pending} {line}"
            depth += line.count("(") - line.count(")")
            if depth <= 0:
                calls.append((pending_function, re.sub(r"\s+", " ", pending)))
                pending = None
            continue
        if not CALL.search(line):
            continue
        require(not line.startswith("//"), f"{label}:{number}: encode call inside a comment; the extractor would count it")
        depth = line.count("(") - line.count(")")
        if depth > 0:
            pending, pending_function = line, function
            continue
        calls.append((function, re.sub(r"\s+", " ", line)))
    require(pending is None, f"{label}: unbalanced parentheses; a call is still open at end of file")
    return calls


def require_nonvacuous(calls, label):
    # An empty sequence compares equal to any other empty sequence. That is the
    # fail-open shape: a tool pointed at the wrong file, or at one whose calls
    # all escaped ENCODE_CALLS, would report IDENTICAL and prove nothing.
    require(calls, f"{label}: no encode calls found; the comparison would be vacuous")


def diff(before, after):
    """Per-function and whole-file differences between two extracted sequences."""
    names = []
    for function, _ in before + after:
        if function not in names:
            names.append(function)
    functions = []
    for name in names:
        old = [call for function, call in before if function == name]
        new = [call for function, call in after if function == name]
        if old != new:
            functions.append({
                "function": name,
                "before_calls": len(old),
                "after_calls": len(new),
                "before": old,
                "after": new,
            })
    return functions


def compare(before_path, before_source, after_path, after_source):
    require(before_path != after_path, f"{before_path}: self-comparison is not evidence; the two sides must name different revisions")
    old = extract(before_source, before_path)
    new = extract(after_source, after_path)
    require_nonvacuous(old, before_path)
    require_nonvacuous(new, after_path)
    functions = diff(old, new)
    return {
        "result": "CHANGED" if functions else "IDENTICAL",
        "before": before_path,
        "after": after_path,
        "before_calls": len(old),
        "after_calls": len(new),
        "changed_functions": functions,
        "calls_tracked": sorted(ENCODE_CALLS),
        "scope": "order, identity and arguments of encode calls, per enclosing function; indentation normalized",
        "limitation": "a reported difference may still be an equivalent rewrite, and an IDENTICAL sequence says nothing about kernel source, host arithmetic, or buffer contents",
    }


def read(spec):
    """Read PATH, or REV:PATH from git."""
    revision, separator, path = spec.partition(":")
    if not separator:
        file = Path(spec)
        require(file.is_file(), f"{spec}: not a regular file")
        return spec, file.read_text()
    import subprocess
    done = subprocess.run(["git", "show", f"{revision}:{path}"], capture_output=True, text=True)
    require(done.returncode == 0, f"{spec}: git show failed: {done.stderr.strip()}")
    return spec, done.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", help="path, or REV:PATH read from git (e.g. d6fd6b8:examples/ds4-min/metal_ops_darwin.go)")
    parser.add_argument("after", help="path, or REV:PATH")
    args = parser.parse_args()
    try:
        before_path, before_source = read(args.before)
        after_path, after_source = read(args.after)
        report = compare(before_path, before_source, after_path, after_source)
        print(json.dumps(report, indent=2))
        return 1 if report["result"] == "CHANGED" else 0
    except (OSError, ValueError, TypeError) as err:
        print(f"compare-encode-sequence: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
