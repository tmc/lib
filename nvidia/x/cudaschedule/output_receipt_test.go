package cudaschedule

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaoutputcoverage"
)

func TestOutputReceipt(t *testing.T) {
	out := &cudaoutputcoverage.RecordedOutput{PTXDigest: strings.Repeat("a", 64), Ordinary: []uint32{1, 2}}
	var b bytes.Buffer
	if err := WriteOutputReceipt(&b, out); err != nil {
		t.Fatal(err)
	}
	wire := b.String()
	clean, got, err := takeOutputReceipt("before\n"+wire+"after\n", os.Getpid(), out.PTXDigest)
	if err != nil || clean != "before\nafter\n" || got == nil || len(got.Ordinary) != 2 || got.Ordinary[1] != 2 {
		t.Fatalf("round trip: clean=%q output=%+v err=%v", clean, got, err)
	}
	for _, tt := range []struct {
		name, wire, ptx string
		pid             int
	}{
		{"wrong pid", wire, out.PTXDigest, os.Getpid() + 1},
		{"wrong PTX", wire, strings.Repeat("b", 64), os.Getpid()},
		{"duplicate", wire + wire, out.PTXDigest, os.Getpid()},
		{"torn", strings.TrimSuffix(wire, "\n"), out.PTXDigest, os.Getpid()},
		{"unknown field", strings.Replace(wire, "{", "{\"Unknown\":0,", 1), out.PTXDigest, os.Getpid()},
		{"duplicate field", strings.Replace(wire, "{", "{\"PID\":0,", 1), out.PTXDigest, os.Getpid()},
		{"too large", outputReceiptPrefix + strings.Repeat("x", outputReceiptLimit+1) + "\n", out.PTXDigest, os.Getpid()},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, _, err := takeOutputReceipt(tt.wire, tt.pid, tt.ptx); err == nil {
				t.Fatal("accepted invalid receipt")
			}
		})
	}
	for _, bad := range []*cudaoutputcoverage.RecordedOutput{nil, {}, {PTXDigest: out.PTXDigest, Ordinary: make([]uint32, 65537)}} {
		if err := WriteOutputReceipt(&bytes.Buffer{}, bad); err == nil {
			t.Fatal("accepted invalid writer evidence")
		}
	}
	for _, tt := range []struct {
		name string
		set  func(*cudaoutputcoverage.RecordedOutput, string)
	}{
		{"buffer", func(o *cudaoutputcoverage.RecordedOutput, s string) { o.TwoFill.Contract.Buffer = s }},
		{"kind", func(o *cudaoutputcoverage.RecordedOutput, s string) {
			o.TwoFill.Contract.Kind = cudaoutputcoverage.Kind(s)
		}},
		{"target", func(o *cudaoutputcoverage.RecordedOutput, s string) { o.TwoFill.Target = s }},
		{"policy", func(o *cudaoutputcoverage.RecordedOutput, s string) { o.Policy.MathMode = s }},
	} {
		t.Run("metadata/"+tt.name, func(t *testing.T) {
			copy := *out
			tt.set(&copy, strings.Repeat("x", 16<<10))
			var b bytes.Buffer
			if err := WriteOutputReceipt(&b, &copy); err == nil || b.Len() != 0 {
				t.Fatalf("oversized metadata: err=%v bytes=%d", err, b.Len())
			}
		})
	}
	copy := *out
	copy.TwoFill.Target = strings.Repeat("x", 8192)
	copy.Policy.Target = strings.Repeat("x", 8192)
	if err := WriteOutputReceipt(&bytes.Buffer{}, &copy); err == nil {
		t.Fatal("accepted aggregate oversized metadata")
	}
}

func ExampleWriteOutputReceipt() {
	var b bytes.Buffer
	err := WriteOutputReceipt(&b, &cudaoutputcoverage.RecordedOutput{PTXDigest: strings.Repeat("a", 64)})
	fmt.Println(err == nil, strings.HasPrefix(b.String(), outputReceiptPrefix))
	// Output: true true
}

func FuzzOutputReceipt(f *testing.F) {
	f.Add("ordinary output\n")
	f.Add(outputReceiptPrefix + "{}\n")
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > 2*outputReceiptLimit {
			t.Skip()
		}
		_, _, _ = takeOutputReceipt(s, os.Getpid(), strings.Repeat("a", 64))
	})
}
