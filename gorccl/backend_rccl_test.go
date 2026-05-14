//go:build gorccl_hardware

package gorccl

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"unsafe"
)

func TestGeneratedRCCLDataTypeMapping(t *testing.T) {
	for _, dtype := range []DType{Int8, Int32, Int64, Uint8, Uint32, Uint64, Float32, Float64} {
		if _, err := rcclDataType(dtype); err != nil {
			t.Fatalf("rcclDataType(%v): %v", dtype, err)
		}
	}
	if _, err := rcclDataType(DTypeInvalid); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("rcclDataType(DTypeInvalid) error = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneratedRCCLReduceOpMapping(t *testing.T) {
	for _, op := range []ReduceOp{Sum, Max, Min, Prod} {
		if _, err := rcclReduceOp(op); err != nil {
			t.Fatalf("rcclReduceOp(%v): %v", op, err)
		}
	}
	if _, err := rcclReduceOp(ReduceOp(-1)); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("rcclReduceOp(invalid) error = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneratedRCCLBufferPtrValidation(t *testing.T) {
	var x int
	valid := Buffer{Ptr: unsafe.Pointer(&x), Count: 1, Device: Device{Kind: DeviceROCM}}
	ptr, err := rcclBufferPtr(valid)
	if err != nil {
		t.Fatalf("rcclBufferPtr(valid): %v", err)
	}
	if ptr == 0 {
		t.Fatal("valid pointer mapped to zero")
	}

	wrongDevice := Buffer{Ptr: unsafe.Pointer(&x), Count: 1, Device: Device{Kind: DeviceUnknown}}
	if _, err := rcclBufferPtr(wrongDevice); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("rcclBufferPtr(wrongDevice) error = %v, want ErrInvalidArgument", err)
	}

	nilPointer := Buffer{Count: 1, Device: Device{Kind: DeviceROCM}}
	if _, err := rcclBufferPtr(nilPointer); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("rcclBufferPtr(nilPointer) error = %v, want ErrInvalidArgument", err)
	}

	ptr, err = rcclBufferPtr(Buffer{})
	if err != nil || ptr != 0 {
		t.Fatalf("rcclBufferPtr(zero count) = %d, %v, want 0, nil", ptr, err)
	}
}

func TestGeneratedRCCLValidateConfig(t *testing.T) {
	if err := rcclValidateConfig(Config{Rank: 0, Size: 1}); err != nil {
		t.Fatalf("rcclValidateConfig(valid): %v", err)
	}
	for _, cfg := range []Config{
		{Rank: 0, Size: 0},
		{Rank: -1, Size: 1},
		{Rank: 1, Size: 1},
		{Rank: 0, Size: 2},
	} {
		if err := rcclValidateConfig(cfg); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("rcclValidateConfig(%+v) error = %v, want ErrInvalidArgument", cfg, err)
		}
	}
}

func TestGeneratedRCCLUniqueIDFromStore(t *testing.T) {
	ctx := context.Background()
	store := newFakeStore()
	want := make([]byte, 128)
	for i := range want {
		want[i] = byte(i)
	}
	if err := store.Set(ctx, "test/rccl/unique-id", want); err != nil {
		t.Fatal(err)
	}
	id, err := rcclUniqueID(ctx, Config{Rank: 1, Size: 2, Rendezvous: "test", Store: store})
	if err != nil {
		t.Fatalf("rcclUniqueID: %v", err)
	}
	if got := id.Internal[:]; !bytes.Equal(got, want) {
		t.Fatalf("unique id = %x, want %x", got, want)
	}
	if err := store.Set(ctx, "bad/rccl/unique-id", []byte("short")); err != nil {
		t.Fatal(err)
	}
	if _, err := rcclUniqueID(ctx, Config{Rank: 1, Size: 2, Rendezvous: "bad", Store: store}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("rcclUniqueID(short) error = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneratedRCCLStreamSelection(t *testing.T) {
	dst := Buffer{Stream: 17}
	src := Buffer{Stream: 11}
	if got := uintptr(rcclStream(dst, src)); got != 17 {
		t.Fatalf("rcclStream(dst, src) = %d, want 17", got)
	}
	dst.Stream = 0
	if got := uintptr(rcclStream(dst, src)); got != 11 {
		t.Fatalf("rcclStream(zero dst, src) = %d, want 11", got)
	}
	if got := uintptr(rcclSingleStream(src)); got != 11 {
		t.Fatalf("rcclSingleStream(src) = %d, want 11", got)
	}
}
