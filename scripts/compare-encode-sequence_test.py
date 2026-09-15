#!/usr/bin/env python3
"""Controls for the encode-sequence differ; fixtures are synthetic Go text."""

import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("compare_encode_sequence", Path(__file__).with_name("compare-encode-sequence.py"))
encode = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(encode)

BASE = """package main

func (o *ops) MatvecQ2K(x []float32) error {
\tenc.SetComputePipelineState(o.kernels.MatvecQ2K)
\tenc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
\tlibmetal.SetScalar(enc, uint32(rows), 1)
\tenc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(rows), libmetal.Threads(min(rows, 256)))
\tenc.EndEncoding()
\treturn nil
}

func (o *ops) Layer0AttentionPost() error {
\tenc.SetComputePipelineState(o.kernels.Layer0Post)
\tenc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
\tenc.EndEncoding()
\treturn nil
}
"""


def swap(source, old, new):
    assert source.count(old) == 1, f"fixture must contain {old!r} exactly once"
    return source.replace(old, new)


class ExtractTest(unittest.TestCase):
    def test_attributes_every_call_to_its_enclosing_function(self):
        calls = encode.extract(BASE, "base")
        self.assertEqual([name for name, _ in calls], ["MatvecQ2K"] * 5 + ["Layer0AttentionPost"] * 3)

    def test_joins_a_call_split_over_lines(self):
        wrapped = swap(
            BASE,
            "\tenc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))",
            "\tenc.DispatchThreadsThreadsPerThreadgroup(\n\t\tlibmetal.Threads(1),\n\t\tlibmetal.Threads(1),\n\t)",
        )
        calls = [call for name, call in encode.extract(wrapped, "wrapped") if name == "Layer0AttentionPost"]
        self.assertEqual(calls[1], "enc.DispatchThreadsThreadsPerThreadgroup( libmetal.Threads(1), libmetal.Threads(1), )")

    def test_rejects_a_call_left_open_at_end_of_file(self):
        with self.assertRaises(encode.EncodeError):
            encode.extract("func (o *ops) F() {\n\tenc.DispatchThreadsThreadsPerThreadgroup(\n", "truncated")

    def test_rejects_an_encode_call_inside_a_comment(self):
        with self.assertRaises(encode.EncodeError):
            encode.extract("func (o *ops) F() {\n\t// enc.EndEncoding() used to live here\n}\n", "commented")


class CompareTest(unittest.TestCase):
    def compare(self, before, after):
        return encode.compare("before.go", before, "after.go", after)

    def test_reindentation_alone_is_identical(self):
        # Wrapping a body in a closure reindents every line inside it. That is
        # the shape of the refactor this tool exists to clear, so it must not
        # register as a difference.
        closure = "".join(
            "\t" + line + "\n" if line.startswith("\tenc.") or line.startswith("\tlibmetal.") else line + "\n"
            for line in BASE.splitlines()
        )
        report = self.compare(BASE, closure)
        self.assertEqual(report["result"], "IDENTICAL")
        self.assertEqual(report["changed_functions"], [])

    def test_a_changed_dispatch_shape_names_its_function(self):
        mutant = swap(BASE, "libmetal.Threads(1), libmetal.Threads(1)", "libmetal.Threads(4096), libmetal.Threads(256)")
        report = self.compare(BASE, mutant)
        self.assertEqual(report["result"], "CHANGED")
        self.assertEqual([item["function"] for item in report["changed_functions"]], ["Layer0AttentionPost"])

    def test_a_changed_dispatch_api_is_a_difference(self):
        mutant = swap(
            BASE,
            "enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(rows), libmetal.Threads(min(rows, 256)))",
            "enc.DispatchThreadgroupsThreadsPerThreadgroup(apple.MTLSize{Width: uint(rows)}, apple.MTLSize{Width: 32})",
        )
        self.assertEqual([item["function"] for item in self.compare(BASE, mutant)["changed_functions"]], ["MatvecQ2K"])

    def test_a_reordered_binding_is_a_difference(self):
        mutant = swap(
            BASE,
            "\tenc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)\n\tlibmetal.SetScalar(enc, uint32(rows), 1)\n",
            "\tlibmetal.SetScalar(enc, uint32(rows), 1)\n\tenc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)\n",
        )
        self.assertEqual(self.compare(BASE, mutant)["result"], "CHANGED")

    def test_a_changed_buffer_index_is_a_difference(self):
        mutant = swap(BASE, "in.Offset(), 0)", "in.Offset(), 1)")
        self.assertEqual(self.compare(BASE, mutant)["result"], "CHANGED")

    def test_a_dropped_call_is_a_difference(self):
        mutant = swap(BASE, "\tlibmetal.SetScalar(enc, uint32(rows), 1)\n", "")
        report = self.compare(BASE, mutant)
        self.assertEqual(report["result"], "CHANGED")
        self.assertEqual(report["changed_functions"][0]["before_calls"] - report["changed_functions"][0]["after_calls"], 1)

    def test_a_call_moved_between_functions_is_a_difference(self):
        # Whole-file counts stay equal; only the per-function attribution moves.
        mutant = swap(BASE, "\tenc.EndEncoding()\n\treturn nil\n}\n\nfunc (o *ops) Layer0AttentionPost() error {\n", "\treturn nil\n}\n\nfunc (o *ops) Layer0AttentionPost() error {\n\tenc.EndEncoding()\n")
        report = self.compare(BASE, mutant)
        self.assertEqual(report["before_calls"], report["after_calls"])
        self.assertEqual(report["result"], "CHANGED")

    def test_an_empty_side_is_refused_rather_than_called_identical(self):
        # The fail-open shape: two empty sequences compare equal. A file with no
        # encode calls means the tool was pointed at the wrong file or the call
        # vocabulary has a hole, and either way the comparison proves nothing.
        with self.assertRaises(encode.EncodeError):
            self.compare(BASE, "package main\n\nfunc (o *ops) F() error { return nil }\n")
        with self.assertRaises(encode.EncodeError):
            encode.compare("a.go", "package main\n", "b.go", "package main\n")

    def test_self_comparison_is_refused(self):
        with self.assertRaises(encode.EncodeError):
            encode.compare("same.go", BASE, "same.go", BASE)


if __name__ == "__main__":
    unittest.main()
