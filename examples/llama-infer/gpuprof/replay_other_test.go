//go:build !darwin

package gpuprof

import (
	"context"
	"testing"
)

// TestProfileReplayFromCapture checks the mapping from a capture summary to the
// replay fields the benchmark path reads. PeakGPRs stays zero because a capture
// does not record register usage; reporting anything else would be invented.
func TestProfileReplayFromCapture(t *testing.T) {
	got, err := ProfileReplay(context.Background(), writeTestCapture(t, cudaprofProfile()))
	if err != nil {
		t.Fatalf("ProfileReplay: %v", err)
	}
	if want := int64(2272 + 4096 + 1500); got.ActiveTimeNS != want {
		t.Errorf("ActiveTimeNS = %d, want %d", got.ActiveTimeNS, want)
	}
	if got.KernelCount != 3 {
		t.Errorf("KernelCount = %d, want 3", got.KernelCount)
	}
	if got.PeakGPRs != 0 {
		t.Errorf("PeakGPRs = %d, want 0: a capture carries no register usage", got.PeakGPRs)
	}
	if got.Capture == nil {
		t.Fatal("Capture summary not retained")
	}
	if got.Capture.CopyBytes != 65536 {
		t.Errorf("Capture.CopyBytes = %d, want 65536", got.Capture.CopyBytes)
	}
}

func TestProfileReplayCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ProfileReplay(ctx, writeTestCapture(t, cudaprofProfile())); err == nil {
		t.Error("ProfileReplay with a cancelled context succeeded, want error")
	}
}
