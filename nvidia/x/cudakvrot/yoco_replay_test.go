package cudakvrot_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudakvrot"
)

func TestResolveSWABoundedReplay(t *testing.T) {
	cfg := cudakvrot.SWAReplayConfig{
		WindowSize:           128,
		SpeculativeOvershoot: 6,
		PageBlockSize:        64,
	}

	// Case 1: Long context (seqLenK=500, queryLen=4) -> retains [500-134, 500) = [366, 500), 134 tokens, 3 pages.
	win, err := cudakvrot.ResolveSWABoundedReplay(500, 4, cfg)
	if err != nil {
		t.Fatalf("ResolveSWABoundedReplay(500, 4) unexpected error: %v", err)
	}
	if win.StartPos != 366 || win.EndPos != 500 || win.ActiveTokens != 134 || win.RequiredPages != 3 {
		t.Fatalf("ResolveSWABoundedReplay(500, 4) = %+v, want StartPos=366 EndPos=500 ActiveTokens=134 RequiredPages=3", win)
	}
	if len(win.RingSlots) != 134 {
		t.Fatalf("len(RingSlots) = %d, want 134", len(win.RingSlots))
	}
	for i, slot := range win.RingSlots {
		wantSlot := (366 + i) % 134
		if slot != wantSlot {
			t.Fatalf("RingSlots[%d] = %d, want %d", i, slot, wantSlot)
		}
	}

	// Case 2: Short context (seqLenK=50, queryLen=10) -> retains [0, 50), 50 tokens, 1 page.
	winShort, err := cudakvrot.ResolveSWABoundedReplay(50, 10, cudakvrot.SWAReplayConfig{})
	if err != nil {
		t.Fatalf("ResolveSWABoundedReplay(50, 10) unexpected error: %v", err)
	}
	if winShort.StartPos != 0 || winShort.EndPos != 50 || winShort.ActiveTokens != 50 || winShort.RequiredPages != 1 {
		t.Fatalf("ResolveSWABoundedReplay(50, 10) = %+v, want StartPos=0 EndPos=50 ActiveTokens=50 RequiredPages=1", winShort)
	}

	// Negative controls.
	if _, err := cudakvrot.ResolveSWABoundedReplay(0, 1, cfg); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("seqLenK=0 error = %v, want ErrUnsupported", err)
	}
	if _, err := cudakvrot.ResolveSWABoundedReplay(10, 12, cfg); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("queryLen > seqLenK error = %v, want ErrUnsupported", err)
	}
	if _, err := cudakvrot.ResolveSWABoundedReplay(10, 1, cudakvrot.SWAReplayConfig{WindowSize: -1}); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("negative WindowSize error = %v, want ErrUnsupported", err)
	}
}
