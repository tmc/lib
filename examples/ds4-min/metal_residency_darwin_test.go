//go:build darwin && arm64

package main

import (
	"os"
	"testing"
	"time"

	libmetal "github.com/tmc/lib/apple/x/metal"

	"github.com/tmc/lib/gguf"
)

// The sweep spends ~40s in its first GPU dispatch, and nothing else in layer 0
// costs more than 19ms. mapMetalModel maps the whole 86GB checkpoint zero-copy
// in MaxBufferLength-sized views, so the suspicion is that the first dispatch
// makes every mapped byte resident, whether or not the kernel reads it. This
// varies only the mapped span and times that first dispatch: if the cost is
// residency it scales with the span, and if it is device warmup it does not.
func TestMetalFirstDispatchResidency(t *testing.T) {
	path := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	if path == "" {
		t.Skip("profile model path not set")
	}
	for _, span := range []uint64{1 << 30, 4 << 30, 16 << 30, 64 << 30} {
		t.Run(byteSpanName(span), func(t *testing.T) {
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			info, err := f.Stat()
			if err != nil {
				t.Fatal(err)
			}
			if span > uint64(info.Size()) {
				t.Skipf("span %d exceeds model %d", span, info.Size())
			}
			g, err := gguf.Read(f, info.Size())
			if err != nil {
				t.Fatal(err)
			}
			var maxTensor uint64
			for _, tensor := range g.Tensors {
				n, err := tensor.Bytes()
				if err != nil {
					t.Fatal(err)
				}
				maxTensor = max(maxTensor, n)
			}
			dev, err := libmetal.NewDevice()
			if err != nil {
				t.Fatal(err)
			}
			defer dev.Close()
			views, err := mapMetalModel(dev, f, span, maxTensor)
			if err != nil {
				t.Skipf("map %d bytes: %v", span, err)
			}
			defer views.Close()
			ops, err := newMetalPinnedOps(dev)
			if err != nil {
				t.Fatal(err)
			}
			defer ops.Close()
			// blk.0.ffn_gate_tid2eid.weight lives at file offset 5.3MB, so it
			// is inside every span measured here; token_embd sits at 77.9GB and
			// would force the full map.
			table, err := views.Bind(g, "blk.0.ffn_gate_tid2eid.weight")
			if err != nil {
				t.Skipf("router table not inside a %d-byte span: %v", span, err)
			}
			logits := make([]float32, 256)
			for i := range logits {
				logits[i] = float32(i%7) * 0.25
			}
			start := time.Now()
			if _, _, _, err := ops.RouterHash(logits, table, 0); err != nil {
				t.Fatal(err)
			}
			first := time.Since(start)
			start = time.Now()
			if _, _, _, err := ops.RouterHash(logits, table, 0); err != nil {
				t.Fatal(err)
			}
			t.Logf("span=%-6s views=%d first=%-10v second=%v",
				byteSpanName(span), len(views.views), first.Round(time.Millisecond), time.Since(start).Round(time.Millisecond))
		})
	}
}

func byteSpanName(n uint64) string {
	switch {
	case n >= 1<<30:
		return itoaGiB(n)
	default:
		return "small"
	}
}

func itoaGiB(n uint64) string {
	g := n >> 30
	digits := ""
	for g > 0 {
		digits = string(rune('0'+g%10)) + digits
		g /= 10
	}
	return digits + "GiB"
}
