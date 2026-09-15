#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("checker", Path(__file__).with_name("check-go-test.py"))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def events(*actions):
    return "\n".join(json.dumps({"Action": action, "Package": "p", **({"Test": "TestGate"} if target else {})}) for action, target in actions)


class ReachTest(unittest.TestCase):
    def test_positive(self):
        report = checker.check(events(("run", True), ("pass", True), ("pass", False)), "p", "TestGate", 0)
        self.assertEqual(report["result"], "PASS_REACHED")

    def test_reject(self):
        for name, data, code in [
            ("empty", "", 0),
            ("wrong filter", events(("pass", False)), 0),
            ("skip", events(("run", True), ("skip", True), ("pass", False)), 0),
            ("failure", events(("run", True), ("fail", True), ("pass", False)), 0),
            ("missing run", events(("pass", True), ("pass", False)), 0),
            ("truncated", events(("run", True), ("pass", True)), 0),
            ("repeated", events(("run", True), ("pass", True), ("run", True), ("pass", True), ("pass", False)), 0),
            ("exit failure", events(("run", True), ("pass", True), ("pass", False)), 7),
            ("malformed", "not json", 0),
        ]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                checker.check(data, "p", "TestGate", code)

    def test_unstamped_binary_output_is_named(self):
        # A bare test binary emits no Package field. Rejecting is correct, but
        # the message must not read as "the test never ran".
        data = "\n".join(json.dumps({"Action": a, **({"Test": "TestGate"} if t else {})}) for a, t in
                         [("run", True), ("pass", True), ("pass", False)])
        with self.assertRaisesRegex(ValueError, "test2json"):
            checker.check(data, "p", "TestGate", 0)

    def test_stamped_output_keeps_the_generic_message(self):
        with self.assertRaisesRegex(ValueError, "required test or package pass is missing"):
            checker.check(events(("run", True)), "p", "TestGate", 0)


if __name__ == "__main__":
    unittest.main()
