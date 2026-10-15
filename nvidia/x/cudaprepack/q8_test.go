package cudaprepack

import (
	"bytes"
	"fmt"
	"math"
	"testing"
)

func TestQ8RejectedExtents(t *testing.T) {
	for _, tt := range []struct {
		name       string
		rows, cols int
	}{
		{"negative rows", -1, 32}, {"negative cols", 1, -32},
		{"empty rows", 0, 32}, {"empty cols", 1, 0},
		{"incomplete block", 1, 33},
		{"block overflow", math.MaxInt, 64},
		{"raw byte overflow", math.MaxInt / 32, 32},
	} {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("invalid extent panicked instead of returning an error: %v", p)
				}
			}()
			if p, err := RepackQ8_0(nil, tt.rows, tt.cols); err == nil || p != nil {
				t.Fatalf("invalid extent returned %v, %v", p, err)
			}
		})
	}
}

func TestQ8StorageAndRoundTrip(t *testing.T) {
	raw := make([]byte, 6*34)
	for i := range raw {
		raw[i] = byte(i*37 + 19)
	}
	p, err := RepackQ8_0(raw, 2, 96)
	if err != nil {
		t.Fatal(err)
	}
	s, err := DescribeQ8(p)
	if err != nil || s != (Q8Storage{2, 96, 96, 6, 204, 192, 12, 204}) {
		t.Fatalf("storage = %+v, %v", s, err)
	}
	// Check payload and scale separation against the block bytes, not a
	// mutually consistent pack/unpack implementation.
	for b := 0; b < 6; b++ {
		if !bytes.Equal(p.Scales[b*2:b*2+2], raw[b*34:b*34+2]) || !bytes.Equal(p.Payload[b*32:b*32+32], raw[b*34+2:b*34+34]) {
			t.Fatalf("block %d changed scale or payload bytes", b)
		}
	}
	// An imported host tensor may have row padding. Poison bytes must not
	// become output quants or scale bytes.
	padded := bytes.Repeat([]byte{0xee}, 2*112)
	for r := 0; r < 2; r++ {
		copy(padded[r*112:], p.Payload[r*96:r*96+96])
	}
	p.Payload = padded
	p.BoxDesc.GlobalStrides[0] = 112
	s, err = DescribeQ8(p)
	if err != nil || s.PayloadBytes != 224 || s.WorkspaceBytes != 236 {
		t.Fatal(s, err)
	}
	out, err := UnpackQ8_0(p)
	if err != nil || !bytes.Equal(out, raw) {
		t.Fatal("row padding or scale bits changed", err)
	}
}

func TestQ8MalformedStorage(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*PrepackedTensor)
	}{
		{"payload", func(p *PrepackedTensor) { p.Payload = p.Payload[:31] }},
		{"scales", func(p *PrepackedTensor) { p.Scales = p.Scales[:1] }},
		{"negative stride", func(p *PrepackedTensor) { p.BoxDesc.GlobalStrides[0] = -32 }},
		{"short stride", func(p *PrepackedTensor) { p.BoxDesc.GlobalStrides[0] = 16 }},
		{"unaligned stride", func(p *PrepackedTensor) { p.BoxDesc.GlobalStrides[0] = 33 }},
		{"wide extent", func(p *PrepackedTensor) { p.BoxDesc.GlobalDims[0] = math.MaxUint64 }},
		{"extent overflow", func(p *PrepackedTensor) { p.BoxDesc.GlobalDims[1] = uint64(math.MaxInt) }},
		{"rank", func(p *PrepackedTensor) { p.BoxDesc.Rank = 3 }},
		{"type", func(p *PrepackedTensor) { p.BoxDesc.DataType = TypeMXFP4 }},
		{"swizzle", func(p *PrepackedTensor) { p.BoxDesc.Swizzle = Swizzle32B }},
		{"interleave", func(p *PrepackedTensor) { p.BoxDesc.Interleave = Interleave16B }},
		{"element stride", func(p *PrepackedTensor) { p.BoxDesc.ElementStrides[1] = 2 }},
		{"sparsity", func(p *PrepackedTensor) { p.Metadata = []byte{1} }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p, err := RepackQ8_0(make([]byte, 34), 1, 32)
			if err != nil {
				t.Fatal(err)
			}
			tt.change(p)
			if out, err := UnpackQ8_0(p); err == nil || out != nil {
				t.Fatal("malformed storage accepted", err)
			}
		})
	}
	if _, err := UnpackQ8_0(nil); err == nil {
		t.Fatal("nil tensor accepted")
	}
}

func ExampleDescribeQ8() {
	p, err := RepackQ8_0(make([]byte, 34), 1, 32)
	if err != nil {
		panic(err)
	}
	s, err := DescribeQ8(p)
	if err != nil {
		panic(err)
	}
	fmt.Println(s.RawBytes, s.PayloadBytes, s.ScaleBytes, s.WorkspaceBytes)
	// Output: 34 32 2 34
}

func FuzzQ8Extents(f *testing.F) {
	f.Add(1, 32)
	f.Add(-1, 32)
	f.Add(0, 0)
	f.Add(math.MaxInt, 64)
	f.Fuzz(func(t *testing.T, rows, cols int) {
		// Bound the provided input; rejected arithmetic must not reach an
		// allocation based on untrusted rows or cols.
		raw := make([]byte, 34)
		p, err := RepackQ8_0(raw, rows, cols)
		if err == nil {
			if rows != 1 || cols != 32 {
				t.Fatal("extent larger than supplied block accepted")
			}
			out, err := UnpackQ8_0(p)
			if err != nil || !bytes.Equal(out, raw) {
				t.Fatal("one-block round trip failed", err)
			}
		}
	})
}
