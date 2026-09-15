package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestRun(t *testing.T) {
	input := `{"version":1,"config":{"pkg":"example.com/decode"},"samples":[{"name":"BenchmarkDecode/full-8","sample_id":"run-1","source":"synthetic-c.json","scope":"full","operations":128,"measurements":[{"unit":"gpu-ns/op","value":256000,"kind":"device-elapsed","basis":"total"}]}]}`
	var out, diagnostics bytes.Buffer
	if err := run(nil, strings.NewReader(input), &out, &diagnostics); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "BenchmarkDecode/full-8 128 2000 gpu-ns/op\n") {
		t.Fatal(out.String())
	}
}

func TestRunRejectsMalformedAndMultipleFiles(t *testing.T) {
	for _, input := range []string{`{`, `{"version":1,"samples":[]}`, `{"unknown":1}`} {
		var out bytes.Buffer
		if run(nil, strings.NewReader(input), &out, &bytes.Buffer{}) == nil {
			t.Fatal("accepted", input)
		}
		if out.Len() != 0 {
			t.Fatal("invalid input wrote output")
		}
	}
	if run([]string{"a", "b"}, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{}) == nil {
		t.Fatal("accepted multiple files")
	}
}
