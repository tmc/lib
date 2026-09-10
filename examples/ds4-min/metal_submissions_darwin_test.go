//go:build darwin && arm64

package main

import (
	"context"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	metalmem "github.com/tmc/lib/apple/x/metal/metalmem"
)

// TestMetalSubmissionsCountsEveryBatch pins the counter to real batches. The
// numbers it asserts are the ones a prefill claim is argued from, so a counter
// that silently stopped counting would make every such claim unfalsifiable.
func TestMetalSubmissionsCountsEveryBatch(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()
	in := metalmem.UploadIn(arena, "input", []float32{1, 2, 3, 4})
	out := metalmem.AllocIn[float32](arena, "output", in.Len())

	if got := ops.submissions; got != (metalSubmissions{}) {
		t.Fatalf("fresh ops already counted %+v, want zero", got)
	}
	start := ops.submissions

	const batches = 3
	for range batches {
		b, err := newMetalBatch(ops)
		if err != nil {
			t.Fatal(err)
		}
		if err := b.EncodeClamp10(in, out); err != nil {
			t.Fatal(err)
		}
		if err := b.Wait(context.Background()); err != nil {
			t.Fatal(err)
		}
	}

	got := ops.submissions.sub(start)
	if got.Buffers != batches {
		t.Errorf("Buffers = %d, want %d", got.Buffers, batches)
	}
	if got.Commits != batches {
		t.Errorf("Commits = %d, want %d", got.Commits, batches)
	}
	if got.Waits != batches {
		t.Errorf("Waits = %d, want %d", got.Waits, batches)
	}
	if got.Encoded < batches {
		t.Errorf("Encoded = %d, want at least one dispatch per batch (%d)", got.Encoded, batches)
	}
}

// TestMetalSubmissionsCountsAnUncommittedBuffer is the control for the claim
// the counter exists to make. A batch that is created and never committed still
// consumed a command buffer, and a submission-structure argument that counted
// only commits would miss it.
func TestMetalSubmissionsCountsAnUncommittedBuffer(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()

	start := ops.submissions
	if _, err := newMetalBatch(ops); err != nil {
		t.Fatal(err)
	}
	got := ops.submissions.sub(start)
	if got.Buffers != 1 {
		t.Fatalf("Buffers = %d, want 1", got.Buffers)
	}
	if got.Commits != 0 || got.Waits != 0 {
		t.Fatalf("uncommitted batch counted Commits=%d Waits=%d, want 0 and 0", got.Commits, got.Waits)
	}
}

func TestMetalSubmissionsPerToken(t *testing.T) {
	s := metalSubmissions{Buffers: 176128, Commits: 176128, Waits: 176128, Encoded: 1000000}
	buffers, commits, waits, _ := s.perToken(2048)
	// 43 layers x 2 command buffers is the shape the sidecar measured.
	if buffers != 86 || commits != 86 || waits != 86 {
		t.Fatalf("perToken = %v %v %v, want 86 86 86", buffers, commits, waits)
	}
	if b, c, w, e := s.perToken(0); b != 0 || c != 0 || w != 0 || e != 0 {
		t.Fatalf("perToken(0) = %v %v %v %v, want zeroes rather than a division", b, c, w, e)
	}
}

func TestMetalSubmissionsSub(t *testing.T) {
	end := metalSubmissions{Buffers: 10, Commits: 9, Waits: 8, Encoded: 700}
	start := metalSubmissions{Buffers: 4, Commits: 4, Waits: 4, Encoded: 200}
	if got, want := end.sub(start), (metalSubmissions{Buffers: 6, Commits: 5, Waits: 4, Encoded: 500}); got != want {
		t.Fatalf("sub = %+v, want %+v", got, want)
	}
}
