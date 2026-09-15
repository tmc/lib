#!/usr/bin/env python3
"""Require one named Go test to run and pass in retained go test -json output."""

import argparse
import json
from pathlib import Path
import sys


def check(data, package, test, exit_code):
    if exit_code != 0:
        raise ValueError(f"command exit {exit_code}, want 0")
    state = "missing"
    package_pass = False
    stamped = False
    for number, line in enumerate(data.splitlines(), 1):
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError(f"line {number}: event is not an object")
        action = event.get("Action")
        if action == "fail":
            raise ValueError(f"line {number}: failure event")
        if "Package" in event:
            stamped = True
        if event.get("Package") != package:
            continue
        name = event.get("Test")
        if name == test:
            if package_pass:
                raise ValueError("test event after package completion")
            if action == "run":
                if state != "missing":
                    raise ValueError("test ran more than once")
                state = "running"
            elif action == "pass":
                if state != "running":
                    raise ValueError("test passed without one active run")
                state = "passed"
            elif action == "skip":
                raise ValueError("required test skipped")
        elif name is None and action == "pass":
            if package_pass or state != "passed":
                raise ValueError("package passed without one completed required test")
            package_pass = True
    if state != "passed" or not package_pass:
        if not stamped:
            # A compiled test binary's output carries no Package field, so every
            # event above was skipped and the run looks like it never happened.
            # Saying "missing" here would be indistinguishable from a test that
            # truly did not run, which is the one thing this check exists to
            # tell apart. Convert the retained stdout with
            # `go tool test2json -p <package> -t` and check that instead.
            raise ValueError("no event carries a Package field: this looks like "
                             "raw test binary output, not go test -json; "
                             "re-convert it with go tool test2json -p <package> -t")
        raise ValueError("required test or package pass is missing")
    return {"result": "PASS_REACHED", "package": package, "test": test,
            "scope": "retained JSON event reach and command exit only; source and executable identity require separate binding"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("events", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.events.read_text(), args.package, args.test, args.exit_code)
    except (OSError, ValueError) as err:
        print(f"check-go-test: {err}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
