//go:build darwin && arm64

package dsv4expertcache

import (
	"context"
	"crypto/sha256"
	"os"
	"testing"

	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// Physical execution is a separately authorized block, not a package default.
func TestMetalSSDDevice18Bytes(t *testing.T) {
	if os.Getenv("DS4_METAL_SSD_DEVICE_TEST") != "1" {
		t.Skip("requires separately reviewed physical block")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("hardware lock required")
	}
	d, err := libmetal.NewDevice()
	o := new(metalDeviceGateOwner)
	// Transfer device ownership immediately, including constructor failures.
	if d != nil {
		o.cleanup = append(o.cleanup, d.Close)
	}
	defer func() {
		if err := o.close(); err != nil {
			retainedMetalDeviceGates = append(retainedMetalDeviceGates, o)
			t.Errorf("retained physical gate owner: %v", err)
		}
	}()
	if err != nil {
		t.Fatal(err)
	}
	u, err := NewMetalUpload(d, 3, [3]int{2, 2, 2}, uint64(3*os.Getpagesize()))
	if u != nil {
		o.cleanup = append(o.cleanup, u.Close)
	}
	if err != nil {
		t.Fatal(err)
	}
	src, resident := source(t), cache(t)
	s, err := u.Select(src, resident, 0, []int{1, 0, 1}, false)
	released := false
	o.cleanup = append(o.cleanup, func() error {
		if released || s == nil {
			return nil
		}
		if !s.Ready() {
			return s.Discard()
		}
		return s.Release(noMetalRead{})
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := u.Close(); err == nil {
		t.Fatal("active consumer storage freed")
	}
	// Adjacent, unselected slot must not affect any selected operand.
	b := u.backend.(*metalUploadDevice)
	unused, err := b.slot(2)
	if err != nil {
		t.Fatal(err)
	}
	unused.HostSlice()[0] = 255
	out, err := metalmem.Alloc[byte](d, 18)
	if err != nil {
		t.Fatal(err)
	}
	o.cleanup = append(o.cleanup, func() error { out.Free(); return nil })
	lib, err := d.CompileLibrary(`#include <metal_stdlib>
using namespace metal;
kernel void copy_component(device const uchar *in [[buffer(0)]], device uchar *out [[buffer(1)]], constant uint &off [[buffer(2)]], uint i [[thread_position_in_grid]]) { if(i<2)out[off+i]=in[i]; }`)
	if err != nil {
		t.Fatal(err)
	}
	o.cleanup = append(o.cleanup, func() error { objectivec.ObjectFromID(lib.GetID()).Release(); return nil })
	pipeline, err := d.NewComputePipeline(lib, "copy_component")
	if err != nil {
		t.Fatal(err)
	}
	o.cleanup = append(o.cleanup, func() error { objectivec.ObjectFromID(pipeline.GetID()).Release(); return nil })
	cmd := d.Queue().CommandBuffer()
	objectivec.ObjectFromID(cmd.GetID()).Retain()
	o.cleanup = append(o.cleanup, func() error { objectivec.ObjectFromID(cmd.GetID()).Release(); return nil })
	enc := cmd.ComputeCommandEncoder()
	objectivec.ObjectFromID(enc.GetID()).Retain()
	ended := false
	o.cleanup = append(o.cleanup, func() error {
		if !ended {
			enc.EndEncoding()
			ended = true
		}
		objectivec.ObjectFromID(enc.GetID()).Release()
		return nil
	})
	enc.SetComputePipelineState(pipeline)
	for route := range 3 {
		views, err := s.Views(route)
		if err != nil {
			t.Fatal(err)
		}
		for component, view := range views {
			enc.SetBufferWithOffsetAtIndex(view.Buffer(), view.Offset(), 0)
			enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 1)
			libmetal.SetScalar(enc, uint32(route*6+component*2), 2)
			enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(2), libmetal.Threads(2))
		}
	}
	enc.EndEncoding()
	ended = true
	cmd.Commit()
	o.submitted = true
	o.wait = func() error { return libmetal.WaitCommandBuffer(context.Background(), cmd) }
	o.terminal = func() bool { return cmd.Status() == 4 || cmd.Status() == 5 }
	if err := s.Release(MetalCommandCompletion{Context: context.Background(), Buffer: cmd}); err != nil {
		t.Fatal(err)
	}
	released = true
	got := string(out.HostSlice())
	if got != "cdghklabefijcdghkl" {
		t.Fatalf("device bytes %q", got)
	}
	if _, err := s.Views(0); err == nil {
		t.Fatal("released view accepted")
	}
	// Released selections must not prevent exact-slot reuse. No command reads
	// this second selection; closing it therefore uses a no-consumer fence.
	s2, err := u.Select(src, resident, 0, []int{0}, false)
	released2 := false
	o.cleanup = append(o.cleanup, func() error {
		if released2 || s2 == nil {
			return nil
		}
		if !s2.Ready() {
			return s2.Discard()
		}
		return s2.Release(noMetalRead{})
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := s2.Release(noMetalRead{}); err != nil {
		t.Fatal(err)
	}
	released2 = true
	t.Logf("device=%s output_sha256=%x dispatches=9 bytes=18", d.Name(), sha256.Sum256(out.HostSlice()))
}

type noMetalRead struct{}

func (noMetalRead) Wait() error { return nil }
