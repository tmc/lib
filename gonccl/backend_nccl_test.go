//go:build gonccl_hardware

package gonccl

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"unsafe"
)

func TestGeneratedNCCLDataTypeMapping(t *testing.T) {
	for _, dtype := range []DType{Int8, Int32, Int64, Uint8, Uint32, Uint64, Float32, Float64} {
		if _, err := ncclDataType(dtype); err != nil {
			t.Fatalf("ncclDataType(%v): %v", dtype, err)
		}
	}
	if _, err := ncclDataType(DTypeInvalid); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("ncclDataType(DTypeInvalid) error = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneratedNCCLReduceOpMapping(t *testing.T) {
	for _, op := range []ReduceOp{Sum, Max, Min, Prod} {
		if _, err := ncclReduceOp(op); err != nil {
			t.Fatalf("ncclReduceOp(%v): %v", op, err)
		}
	}
	if _, err := ncclReduceOp(ReduceOp(-1)); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("ncclReduceOp(invalid) error = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneratedNCCLBufferPtrValidation(t *testing.T) {
	var x int
	valid := Buffer{Ptr: unsafe.Pointer(&x), Count: 1, Device: Device{Kind: DeviceCUDA}}
	ptr, err := ncclBufferPtr(valid)
	if err != nil {
		t.Fatalf("ncclBufferPtr(valid): %v", err)
	}
	if ptr == 0 {
		t.Fatal("valid pointer mapped to zero")
	}

	wrongDevice := Buffer{Ptr: unsafe.Pointer(&x), Count: 1, Device: Device{Kind: DeviceUnknown}}
	if _, err := ncclBufferPtr(wrongDevice); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("ncclBufferPtr(wrongDevice) error = %v, want ErrInvalidArgument", err)
	}

	nilPointer := Buffer{Count: 1, Device: Device{Kind: DeviceCUDA}}
	if _, err := ncclBufferPtr(nilPointer); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("ncclBufferPtr(nilPointer) error = %v, want ErrInvalidArgument", err)
	}

	ptr, err = ncclBufferPtr(Buffer{})
	if err != nil || ptr != 0 {
		t.Fatalf("ncclBufferPtr(zero count) = %d, %v, want 0, nil", ptr, err)
	}
}

func TestGeneratedNCCLValidateConfig(t *testing.T) {
	if err := ncclValidateConfig(Config{Rank: 0, Size: 1}); err != nil {
		t.Fatalf("ncclValidateConfig(valid): %v", err)
	}
	for _, cfg := range []Config{
		{Rank: 0, Size: 0},
		{Rank: -1, Size: 1},
		{Rank: 1, Size: 1},
		{Rank: 0, Size: 2},
	} {
		if err := ncclValidateConfig(cfg); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("ncclValidateConfig(%+v) error = %v, want ErrInvalidArgument", cfg, err)
		}
	}
}

func TestGeneratedNCCLUniqueIDFromStore(t *testing.T) {
	ctx := context.Background()
	store := newFakeStore()
	want := make([]byte, 128)
	for i := range want {
		want[i] = byte(i)
	}
	if err := store.Set(ctx, "test/nccl/unique-id", want); err != nil {
		t.Fatal(err)
	}
	id, err := ncclUniqueID(ctx, Config{Rank: 1, Size: 2, Rendezvous: "test", Store: store})
	if err != nil {
		t.Fatalf("ncclUniqueID: %v", err)
	}
	if got := id.Internal[:]; !bytes.Equal(got, want) {
		t.Fatalf("unique id = %x, want %x", got, want)
	}
	if err := store.Set(ctx, "bad/nccl/unique-id", []byte("short")); err != nil {
		t.Fatal(err)
	}
	if _, err := ncclUniqueID(ctx, Config{Rank: 1, Size: 2, Rendezvous: "bad", Store: store}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("ncclUniqueID(short) error = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneratedNCCLStreamSelection(t *testing.T) {
	dst := Buffer{Stream: 17}
	src := Buffer{Stream: 11}
	if got := uintptr(ncclStream(dst, src)); got != 17 {
		t.Fatalf("ncclStream(dst, src) = %d, want 17", got)
	}
	dst.Stream = 0
	if got := uintptr(ncclStream(dst, src)); got != 11 {
		t.Fatalf("ncclStream(zero dst, src) = %d, want 11", got)
	}
	if got := uintptr(ncclSingleStream(src)); got != 11 {
		t.Fatalf("ncclSingleStream(src) = %d, want 11", got)
	}
}
