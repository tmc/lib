package main

import (
	"bytes"
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasafetensors"
)

// stubSource serves fixed tensors, so the retention tests do not need a
// checkpoint on disk.
type stubSource struct {
	raw  map[string][]byte
	info map[string]cudasafetensors.TensorInfo
	err  error
}

func (s stubSource) TensorBytes(name string) ([]byte, cudasafetensors.TensorInfo, error) {
	if s.err != nil {
		return nil, cudasafetensors.TensorInfo{}, s.err
	}
	raw, ok := s.raw[name]
	if !ok {
		return nil, cudasafetensors.TensorInfo{}, errors.New("not found")
	}
	return raw, s.info[name], nil
}

// TestRetainedTensorPreservesSourceBytes checks byte preservation directly.
//
// The comparison is against the source bytes, not against a float32 round
// trip: widening to float32 and narrowing back is lossy for payloads that
// float32 cannot distinguish, so a round trip would pass on values this test
// exists to catch.
func TestRetainedTensorPreservesSourceBytes(t *testing.T) {
	tests := []struct {
		name  string
		dtype string
		raw   []byte
		shape []int
	}{
		// Signed zero: 0x8000 is -0.0 and 0x0000 is +0.0. They compare equal
		// as floats, so only the bytes distinguish them.
		{"f16 signed zeros", "F16", []byte{0x00, 0x00, 0x00, 0x80}, []int{2}},
		{"bf16 signed zeros", "BF16", []byte{0x00, 0x00, 0x00, 0x80}, []int{2}},

		// NaN payloads: every one of these is a NaN. Widening to float32 and
		// narrowing back does not have to preserve which NaN it was.
		{"f16 nan payloads", "F16", []byte{0x01, 0x7C, 0xFF, 0x7F, 0x01, 0xFC}, []int{3}},
		{"bf16 nan payloads", "BF16", []byte{0x01, 0x7F, 0xFF, 0x7F, 0x01, 0xFF}, []int{3}},

		// Subnormals, infinities and ordinary values.
		{"f16 subnormal", "F16", []byte{0x01, 0x00, 0xFF, 0x03}, []int{2}},
		{"f16 infinities", "F16", []byte{0x00, 0x7C, 0x00, 0xFC}, []int{2}},
		{"f32 passthrough", "F32", []byte{0xDE, 0xAD, 0xBE, 0xEF}, []int{1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := append([]byte(nil), tt.raw...)
			rt, err := NewRetainedTensor(tt.raw, tt.dtype, tt.shape)
			if err != nil {
				t.Fatalf("NewRetainedTensor: %v", err)
			}
			if !bytes.Equal(rt.Raw, src) {
				t.Errorf("Raw = % x, want % x", rt.Raw, src)
			}
			if rt.DType != tt.dtype {
				t.Errorf("DType = %q, want %q", rt.DType, tt.dtype)
			}
		})
	}
}

// TestRetainedTensorKeepsF16AndBF16Distinct checks that the tag travels with
// the bytes.
//
// F16 and BF16 are both two bytes wide, so identical bytes are two different
// values depending on the tag. Losing the tag would silently reinterpret one
// as the other.
func TestRetainedTensorKeepsF16AndBF16Distinct(t *testing.T) {
	// 0x3C00 is 1.0 in F16. The same bytes in BF16 are about 0.0078125.
	raw := []byte{0x00, 0x3C}

	half, err := NewRetainedTensor(append([]byte(nil), raw...), "F16", []int{1})
	if err != nil {
		t.Fatalf("NewRetainedTensor F16: %v", err)
	}
	brain, err := NewRetainedTensor(append([]byte(nil), raw...), "BF16", []int{1})
	if err != nil {
		t.Fatalf("NewRetainedTensor BF16: %v", err)
	}
	if !bytes.Equal(half.Raw, brain.Raw) {
		t.Fatal("test setup: the two tensors should hold identical bytes")
	}
	if half.DType == brain.DType {
		t.Fatal("F16 and BF16 tensors carry the same tag")
	}

	gotHalf, gotBrain := half.DecodeFloat32(), brain.DecodeFloat32()
	if gotHalf[0] == gotBrain[0] {
		t.Fatalf("identical bytes decoded identically as F16 and BF16: %v", gotHalf[0])
	}
	if want := float32(1.0); gotHalf[0] != want {
		t.Errorf("F16 decode = %v, want %v", gotHalf[0], want)
	}
	if want := float32(0.0078125); gotBrain[0] != want {
		t.Errorf("BF16 decode = %v, want %v", gotBrain[0], want)
	}
}

// TestRetainedTensorRejectsBadMetadata checks that metadata disagreeing with
// the payload is refused rather than interpreted.
func TestRetainedTensorRejectsBadMetadata(t *testing.T) {
	tests := []struct {
		name  string
		raw   []byte
		dtype string
		shape []int
	}{
		{"unsupported dtype", make([]byte, 4), "I64", []int{1}},
		{"unknown dtype", make([]byte, 4), "", []int{1}},
		{"f8 not supported", make([]byte, 1), "F8_E4M3", []int{1}},
		{"payload too short", make([]byte, 6), "F16", []int{4}},
		{"payload too long", make([]byte, 10), "F16", []int{4}},
		{"f32 payload tagged f16", make([]byte, 16), "F16", []int{4}},
		{"negative dimension", make([]byte, 4), "F16", []int{-2}},
		{"empty payload nonzero shape", nil, "F16", []int{1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewRetainedTensor(tt.raw, tt.dtype, tt.shape); err == nil {
				t.Fatal("accepted invalid tensor")
			}
		})
	}
}

// TestRetainedTensorRetainsTwoBytesPerHalfElement checks the representation
// actually stores two bytes per 16-bit weight.
//
// This is the accounting claim: a retained tensor that secretly kept a float32
// copy would still pass the byte-equality tests above.
func TestRetainedTensorRetainsTwoBytesPerHalfElement(t *testing.T) {
	const elems = 512
	for _, dtype := range []string{"F16", "BF16"} {
		t.Run(dtype, func(t *testing.T) {
			rt, err := NewRetainedTensor(make([]byte, elems*2), dtype, []int{elems})
			if err != nil {
				t.Fatalf("NewRetainedTensor: %v", err)
			}
			if got := rt.Elems(); got != elems {
				t.Errorf("Elems() = %d, want %d", got, elems)
			}
			if got, want := rt.Bytes(), elems*2; got != want {
				t.Errorf("Bytes() = %d, want %d", got, want)
			}
			if !rt.IsHalf() {
				t.Error("IsHalf() = false")
			}
			if got, want := float64(rt.Bytes())/float64(rt.Elems()), 2.0; got != want {
				t.Errorf("bytes per element = %v, want %v", got, want)
			}
		})
	}
}

// TestLoadRetainedTensorsMixedDTypes checks a checkpoint whose tensors are not
// all the same dtype.
func TestLoadRetainedTensorsMixedDTypes(t *testing.T) {
	src := stubSource{
		raw: map[string][]byte{
			"a.f32":  make([]byte, 4*8),
			"b.f16":  make([]byte, 2*8),
			"c.bf16": make([]byte, 2*8),
		},
		info: map[string]cudasafetensors.TensorInfo{
			"a.f32":  {DType: "F32", Shape: []int{8}},
			"b.f16":  {DType: "F16", Shape: []int{8}},
			"c.bf16": {DType: "BF16", Shape: []int{8}},
		},
	}
	c, err := loadRetainedTensors(src, []string{"a.f32", "b.f16", "c.bf16"})
	if err != nil {
		t.Fatalf("loadRetainedTensors: %v", err)
	}
	if got, want := len(c.Tensors), 3; got != want {
		t.Fatalf("tensor count = %d, want %d", got, want)
	}
	for name, want := range map[string]string{"a.f32": "F32", "b.f16": "F16", "c.bf16": "BF16"} {
		if got := c.Tensors[name].DType; got != want {
			t.Errorf("%s dtype = %q, want %q", name, got, want)
		}
	}

	// 8 F32 elements at 4 bytes plus 16 half elements at 2 bytes.
	if got, want := c.PayloadBytes(), 8*4+16*2; got != want {
		t.Errorf("PayloadBytes() = %d, want %d", got, want)
	}
	// The same 24 elements widened to float32.
	if got, want := c.PayloadBytesAsFloat32(), 24*4; got != want {
		t.Errorf("PayloadBytesAsFloat32() = %d, want %d", got, want)
	}
}

// TestLoadRetainedTensorsPreservesBytesEndToEnd checks that the bytes a
// tensor arrives with are the bytes the loader hands back.
//
// The constructor test above covers NewRetainedTensor in isolation. This one
// covers the path callers actually use, with payloads that a float32 round
// trip would not preserve.
func TestLoadRetainedTensorsPreservesBytesEndToEnd(t *testing.T) {
	want := map[string][]byte{
		// Signed zeros and NaN payloads, which survive only as bytes.
		"half":  {0x00, 0x80, 0x01, 0x7C, 0xFF, 0xFF},
		"brain": {0x00, 0x80, 0x01, 0x7F, 0xFF, 0xFF},
	}
	src := stubSource{
		raw: map[string][]byte{
			"half":  append([]byte(nil), want["half"]...),
			"brain": append([]byte(nil), want["brain"]...),
		},
		info: map[string]cudasafetensors.TensorInfo{
			"half":  {DType: "F16", Shape: []int{3}},
			"brain": {DType: "BF16", Shape: []int{3}},
		},
	}
	c, err := loadRetainedTensors(src, []string{"half", "brain"})
	if err != nil {
		t.Fatalf("loadRetainedTensors: %v", err)
	}
	for name, w := range want {
		if got := c.Tensors[name].Raw; !bytes.Equal(got, w) {
			t.Errorf("%s = % x, want % x", name, got, w)
		}
	}
	if got, want := c.PayloadBytes(), 12; got != want {
		t.Errorf("PayloadBytes() = %d, want %d", got, want)
	}
}

// TestLoadRetainedTensorsRejectsBadTensor checks that one unusable tensor
// fails the load rather than being skipped.
func TestLoadRetainedTensorsRejectsBadTensor(t *testing.T) {
	src := stubSource{
		raw:  map[string][]byte{"bad": make([]byte, 7)},
		info: map[string]cudasafetensors.TensorInfo{"bad": {DType: "F16", Shape: []int{4}}},
	}
	if _, err := loadRetainedTensors(src, []string{"bad"}); err == nil {
		t.Fatal("accepted a tensor whose payload does not match its metadata")
	}
	if _, err := loadRetainedTensors(stubSource{err: errors.New("io failure")}, []string{"x"}); err == nil {
		t.Fatal("accepted a failing source")
	}
}

// TestRetainedDecodeMatchesReferencePath checks that decoding a retained
// tensor produces exactly what the existing float32 loader produced.
//
// This is the separate numerical claim: retention must not change the values
// the float32 consumers already see.
func TestRetainedDecodeMatchesReferencePath(t *testing.T) {
	for _, dtype := range []string{"F16", "BF16", "F32"} {
		t.Run(dtype, func(t *testing.T) {
			raw := make([]byte, 64)
			for i := range raw {
				raw[i] = byte(i*7 + 1)
			}
			size, err := dtypeElemSize(dtype)
			if err != nil {
				t.Fatalf("dtypeElemSize: %v", err)
			}
			rt, err := NewRetainedTensor(append([]byte(nil), raw...), dtype, []int{len(raw) / size})
			if err != nil {
				t.Fatalf("NewRetainedTensor: %v", err)
			}
			want := decodeTensorToFloat32(raw, dtype)
			got := rt.DecodeFloat32()
			if len(got) != len(want) {
				t.Fatalf("decoded %d elements, want %d", len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] && !(math.IsNaN(float64(got[i])) && math.IsNaN(float64(want[i]))) {
					t.Fatalf("element %d = %v, want %v", i, got[i], want[i])
				}
			}
		})
	}
}
