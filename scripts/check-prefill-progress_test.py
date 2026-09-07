import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("checker", Path(__file__).with_name("check-prefill-progress.py"))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def record(i):
    return dict(schema=v.SCHEMA, completed_tokens=i * 128,
                elapsed_ns=i * 1000000000, total_tokens=2048)


def encode(records):
    return b"".join(json.dumps(r).encode() + b"\n" for r in records)


class TestProgress(unittest.TestCase):
    def test_all_prefixes(self):
        for count in range(1, 18):
            with self.subTest(count=count):
                result = v.check(encode([record(i) for i in range(count)]))
                self.assertEqual(result["records"], count)
                self.assertEqual(result["completed_tokens"], (count - 1) * 128)
                self.assertEqual(result["result"], "PREFILL_REACHED" if count == 17 else "OBSERVED_PREFIX")
                self.assertEqual(len(result["intervals"]), count - 1)
                for interval in result["intervals"]:
                    self.assertEqual(interval["elapsed_ns"], 1000000000)
                self.assertNotIn("tokens_per_second", result)

    def test_empty_and_truncated(self):
        for data in (b"", b"\n", b"{", encode([record(0)])[:-1], encode([record(0)]) + b"{"):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    v.check(data)

    def test_steps(self):
        for indexes in ([1], [0, 2], [0, 1, 1], [0, 2, 1], list(range(18))):
            with self.subTest(indexes=indexes):
                with self.assertRaises(ValueError):
                    v.check(encode([record(i) for i in indexes]))

    def test_identity(self):
        for key, value in (("schema", "other"), ("total_tokens", 64), ("total_tokens", True)):
            with self.subTest(key=key, value=value):
                row = record(0)
                row[key] = value
                with self.assertRaises(ValueError):
                    v.check(encode([row]))

    def test_types(self):
        for key in ("completed_tokens", "elapsed_ns", "total_tokens"):
            for value in (True, 0.0, "0", None, [], float("nan"), float("inf")):
                with self.subTest(key=key, value=value):
                    row = record(0)
                    row[key] = value
                    with self.assertRaises(ValueError):
                        v.check(encode([row]))

    def test_elapsed(self):
        for value in (-1, 0, 2**63):
            with self.subTest(value=value):
                row = record(1)
                row["elapsed_ns"] = value
                with self.assertRaises(ValueError):
                    v.check(encode([record(0), row]))
        row = record(0)
        row["elapsed_ns"] = 1
        with self.assertRaises(ValueError):
            v.check(encode([row]))

    def test_fields(self):
        for row in ([], None, {}, dict(record(0), extra=1)):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    v.check(encode([row]))
        for key in v.FIELDS:
            row = record(0)
            del row[key]
            with self.assertRaises(ValueError):
                v.check(encode([row]))

    def test_duplicate_key(self):
        data = encode([record(0)]).replace(b'{', b'{"elapsed_ns":9,', 1)
        with self.assertRaises(ValueError):
            v.check(data)


if __name__ == "__main__":
    unittest.main()
