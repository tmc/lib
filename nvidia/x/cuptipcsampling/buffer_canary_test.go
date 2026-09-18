package cuptipcsampling

import (
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// TestReasonCanaryDetectsOverrun is the pair that makes the canary a gate
// rather than an unread byte. Nothing tells CUPTI how long a per-PC reason
// array is: the only bound is the reason count given at configuration time, so
// a mismatch is a heap overrun that surfaces as a fault inside libcupti with no
// Go frame naming it. The canary has to fail on the overrun and stay silent on
// a legitimate full-capacity write, or it is either useless or a false alarm on
// every saturated PC.
func TestReasonCanaryDetectsOverrun(t *testing.T) {
	const reasonCap = 2

	// full is the legitimate extreme: CUPTI fills every configured reason for
	// a PC and writes nothing beyond them. Without this arm a canary that
	// fired one entry early would still pass the overrun arm.
	t.Run("full capacity is not an overrun", func(t *testing.T) {
		b := newCanaryBuffer(t, reasonCap)
		b.pcs[0].StallReasonCount = reasonCap
		for i := 0; i < reasonCap; i++ {
			b.reasons[0][i] = cupti.PCSamplingStallReason{PcSamplingStallReasonIndex: uint32(i + 1), Samples: 7}
		}
		if err := b.checkPC(0); err != nil {
			t.Fatalf("checkPC on a full but legal PC: %v", err)
		}
	})

	// overrun writes one entry past the configured count, which is what an
	// undersized array does under a native writer that was told a larger
	// reason count than the arrays were built for.
	t.Run("one entry past the configured count", func(t *testing.T) {
		b := newCanaryBuffer(t, reasonCap)
		b.pcs[0].StallReasonCount = reasonCap
		b.reasons[0][reasonCap] = cupti.PCSamplingStallReason{PcSamplingStallReasonIndex: 9, Samples: 1}
		err := b.checkPC(0)
		if err == nil {
			t.Fatal("checkPC accepted a PC whose reason array was written past its configured end")
		}
		if !strings.Contains(err.Error(), "overran") {
			t.Errorf("checkPC error = %q, want it to name the overrun", err)
		}
	})

	// A count past the configured cap is the other spelling of the same fault
	// and must not be reported as a canary hit: the count is CUPTI's own claim,
	// and it is checked before the array is read.
	t.Run("count past the configured cap", func(t *testing.T) {
		b := newCanaryBuffer(t, reasonCap)
		b.pcs[0].StallReasonCount = reasonCap + 1
		if err := b.checkPC(0); err == nil {
			t.Fatal("checkPC accepted a StallReasonCount past the configured cap")
		}
	})
}

func newCanaryBuffer(t *testing.T, reasonCap int) *samplingBuffer {
	t.Helper()
	header := uint64(unsafe.Sizeof(cupti.PCSamplingData{}))
	stride := uint64(unsafe.Sizeof(cupti.PCSamplingPCData{})) + uint64(reasonCap)*uint64(unsafe.Sizeof(cupti.PCSamplingStallReason{}))
	b, err := newSamplingBuffer(1, reasonCap, header+stride)
	if err != nil {
		t.Fatalf("newSamplingBuffer: %v", err)
	}
	t.Cleanup(b.release)
	if got := len(b.reasons[0]); got != reasonCap+1 {
		t.Fatalf("per-PC reason array = %d entries, want %d configured plus one canary", got, reasonCap+1)
	}
	return b
}
