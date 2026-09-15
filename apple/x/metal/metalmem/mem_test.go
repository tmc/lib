package metalmem_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

func TestAllocMMapZeroCopyPrivateMapping(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	t.Logf("device=%s max_buffer=%d page=%d", dev.Name(), dev.Raw().MaxBufferLength(), os.Getpagesize())
	path := filepath.Join(t.TempDir(), "mapping")
	contents := make([]byte, os.Getpagesize()-7)
	contents[17] = 0x29
	if err := os.WriteFile(path, contents, 0o400); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s, err := metalmem.AllocMMapZeroCopy[byte](dev, f, 0, len(contents))
	if err != nil {
		t.Fatal(err)
	}
	if got := s.HostSlice()[17]; got != 0x29 {
		t.Fatalf("mapped byte = %#x, want %#x", got, byte(0x29))
	}
	compiler, err := metalx.NewCompiler(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer compiler.Close()
	pipeline, err := compiler.Compile(`
#include <metal_stdlib>
using namespace metal;
kernel void write_byte(device uchar *p [[buffer(0)]]) { p[17] = 0x5a; }
`, "write_byte")
	if err != nil {
		t.Fatal(err)
	}
	cb := dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(pipeline)
	enc.SetBufferWithOffsetAtIndex(s.Buffer(), s.Offset(), 0)
	enc.DispatchThreadsThreadsPerThreadgroup(metal.Threads(1), metal.Threads(1))
	enc.EndEncoding()
	cb.Commit()
	if err := metal.WaitCommandBuffer(context.Background(), cb); err != nil {
		t.Fatal(err)
	}
	if got := s.HostSlice()[17]; got != 0x5a {
		t.Fatalf("GPU-written byte = %#x, want %#x", got, byte(0x5a))
	}
	b := []byte{0}
	if _, err := f.ReadAt(b, 17); err != nil {
		t.Fatal(err)
	}
	if b[0] != 0x29 {
		t.Fatalf("file byte changed to %#x", b[0])
	}
	s.Free()
}

func TestAllocMMapZeroCopyNonPageOffset(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	path := filepath.Join(t.TempDir(), "mapping")
	contents := make([]byte, os.Getpagesize())
	contents[37] = 0x6b
	if err := os.WriteFile(path, contents, 0o400); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s, err := metalmem.AllocMMapZeroCopy[byte](dev, f, 37, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Free()
	if s.Offset() != 37 || s.Len() != 1 || s.Cap() != 1 || s.HostSlice()[0] != 0x6b {
		t.Fatalf("offset=%d len=%d cap=%d data=%#x", s.Offset(), s.Len(), s.Cap(), s.HostSlice()[0])
	}
	if _, err := metalmem.AllocMMapZeroCopy[uint32](dev, f, 1, 1); err == nil || !strings.Contains(err.Error(), "not aligned for element") {
		t.Fatalf("error = %v, want element alignment error", err)
	}
}

func TestAllocMMapZeroCopyRejectsOversizedBuffer(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	path := filepath.Join(t.TempDir(), "mapping")
	if err := os.WriteFile(path, []byte{0}, 0o400); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	length := int(dev.Raw().MaxBufferLength()) + 1
	if _, err := metalmem.AllocMMapZeroCopy[byte](dev, f, 0, length); err == nil || !strings.Contains(err.Error(), "exceeds Metal maximum") {
		t.Fatalf("error = %v, want Metal maximum error", err)
	}
}

func TestAllocationInputsFailClosed(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	tooMany := int(^uint(0)>>1)/8 + 1
	tests := []struct {
		name string
		fn   func() error
		want string
	}{
		{
			name: "alloc byte count overflows int",
			fn: func() error {
				_, err := metalmem.Alloc[uint64](dev, tooMany)
				return err
			},
			want: "allocation size overflows",
		},
		{
			name: "mmap byte count overflows int",
			fn: func() error {
				_, err := metalmem.AllocMMapZeroCopy[uint64](dev, nil, 0, tooMany)
				return err
			},
			want: "allocation size overflows",
		},
		{
			name: "nil mmap file",
			fn: func() error {
				_, err := metalmem.AllocMMapZeroCopy[byte](dev, nil, 0, 1)
				return err
			},
			want: "nil mmap file",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.fn()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want text %q", err, test.want)
			}
		})
	}
}

func TestAllocAndCopies(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}

	slice, err := metalmem.Alloc[float32](dev, 64)
	if err != nil {
		t.Fatalf("Alloc failed: %v", err)
	}

	if slice.Len() != 64 || slice.Cap() != 64 {
		t.Errorf("unexpected len/cap: %d/%d", slice.Len(), slice.Cap())
	}

	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i) * 1.5
	}

	if err := slice.CopyFromHost(src); err != nil {
		t.Fatalf("CopyFromHost failed: %v", err)
	}

	dst := make([]float32, 64)
	if err := slice.CopyToHost(dst); err != nil {
		t.Fatalf("CopyToHost failed: %v", err)
	}

	for i := range src {
		if src[i] != dst[i] {
			t.Fatalf("mismatch at %d: want %f, got %f", i, src[i], dst[i])
		}
	}
}

func TestReleasedSliceFailsClosed(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	s, err := metalmem.Alloc[uint32](dev, 4)
	if err != nil {
		t.Fatal(err)
	}
	sub, err := s.Slice(1, 3)
	if err != nil {
		t.Fatal(err)
	}
	s.Free()
	sub.Free()
	if s.Buffer() != nil || s.HostSlice() != nil || sub.HostSlice() != nil {
		t.Fatal("released allocation remains accessible")
	}
	if _, err := sub.Slice(0, 1); err == nil {
		t.Fatal("slice of released allocation succeeded")
	}
	if err := sub.CopyFromHost([]uint32{1}); err == nil {
		t.Fatal("copy to released allocation succeeded")
	}
}

func TestCopyToHostChecksEmptyDestination(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	s, err := metalmem.Alloc[uint32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Free()
	if err := s.CopyToHost(nil); err == nil {
		t.Fatal("non-empty source copied to empty destination")
	}
}
