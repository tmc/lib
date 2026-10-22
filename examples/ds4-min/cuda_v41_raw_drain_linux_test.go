//go:build linux

package main

import (
	"errors"
	"os"
	"strconv"
	"testing"
)

func TestCUDAV41RawProductionDrainCustody(t *testing.T) {
	if os.Getenv("DS4_CUDA_V41_OPS_GATE") != "1" {
		t.Skip("opt-in V4.1 CUDA gate")
	}
	if pid, err := strconv.Atoi(os.Getenv("GPU_LOCK_HELD")); err != nil || pid <= 0 {
		t.Fatal("gpu-lock required")
	}
	r, resources := newV41RawTestResources(t)
	op, err := newCUDAV41RawAttention(r, 2)
	if op != nil {
		resources.ownerClose = op.Close
	}
	if err != nil {
		resources.retainError(err)
		t.Fatal(err)
	}
	q, kv, sinks := make([]float32, 64*512), make([]float32, 512), make([]float32, 64)
	for i := range kv {
		kv[i] = 0.5
	}
	for i := range sinks {
		sinks[i] = -1000
	}
	dq, dk, ds := v41RawUpload(t, r, resources, q), v41RawUpload(t, r, resources, kv), v41RawUpload(t, r, resources, sinks)
	out := v41RawUpload(t, r, resources, q)
	planted := errors.New("planted production drain uncertainty")
	drains := 0
	op.drainProbe = func() error {
		drains++
		// Actual submitted device work reaches terminal first; only its host
		// receipt is replaced. This is not a physically failed GPU simulation.
		if err := r.device.WithContext(func() error { return syncStream(r.stream()) }); err != nil {
			return err
		}
		return planted
	}
	if err := op.Append(out, dq, dk, ds, 0); !errors.Is(err, planted) {
		t.Fatalf("missing planted append uncertainty: %v", err)
	}
	if drains != 1 || op.valid || op.position != 0 || len(op.pending) != 4 {
		t.Fatal("production failed drain lost pending custody")
	}
	got := make([]float32, out.Len())
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if got[0] != 0.5 {
		t.Fatal("production drain probe did not follow actual device work")
	}
	if err := op.Append(out, dq, dk, ds, 0); err == nil || drains != 1 {
		t.Fatal("uncertain request admitted retry")
	}
	if err := op.Close(); !errors.Is(err, planted) {
		t.Fatalf("missing planted close uncertainty: %v", err)
	}
	if drains != 2 || op.module == 0 || op.ops == nil || op.q.Ptr() == 0 || len(op.pending) != 4 {
		t.Fatal("production close drain lost dependency custody")
	}
	op.drainProbe = nil
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if op.module != 0 || op.ops != nil || op.q.Ptr() != 0 || len(op.pending) != 0 {
		t.Fatal("production terminal retry failed to release")
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
}
