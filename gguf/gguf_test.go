package gguf

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"testing"
)

func TestRead(t *testing.T) {
	b := fixture()
	g, err := Read(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	if g.DataOffset != 192 {
		t.Fatalf("data offset = %d, want 192", g.DataOffset)
	}
	if got := g.Metadata["general.architecture"].Value; got != "deepseek4" {
		t.Fatalf("architecture = %q", got)
	}
	if len(g.Tensors) != 1 || g.Tensors[0].Name != "token_embd.weight" {
		t.Fatalf("tensors = %#v", g.Tensors)
	}
}

func TestReadTruncated(t *testing.T) {
	b := fixture()
	for n := range len(b) {
		if _, err := Read(bytes.NewReader(b[:n]), int64(n)); err == nil {
			t.Fatalf("length %d succeeded", n)
		}
	}
}

func TestReadMalformed(t *testing.T) {
	tests := []struct {
		name string
		edit func([]byte)
	}{
		{"magic", func(b []byte) { b[0] = 'X' }},
		{"version", func(b []byte) { put32(b[4:], 2) }},
		{"tensor count", func(b []byte) { put64(b[8:], maxTensors+1) }},
		{"metadata count", func(b []byte) { put64(b[16:], maxMetadata+1) }},
		{"zero alignment", func(b []byte) {
			i := bytes.Index(b, []byte("general.alignment")) + len("general.alignment") + 4
			put32(b[i:], 0)
		}},
		{"bad dimensions", func(b []byte) {
			i := bytes.Index(b, []byte("token_embd.weight")) + len("token_embd.weight")
			put32(b[i:], 5)
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := fixture()
			tt.edit(b)
			if _, err := Read(bytes.NewReader(b), int64(len(b))); err == nil {
				t.Fatal("read succeeded")
			}
		})
	}
}

func TestReadStringBound(t *testing.T) {
	b := fixture()
	put64(b[24:], maxString+1)
	_, err := Read(bytes.NewReader(b), int64(len(b)))
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("error = %v, want invalid GGUF", err)
	}
}

func TestReadArrayAllocationBound(t *testing.T) {
	var b bytes.Buffer
	b.WriteString("GGUF")
	write32(&b, 3)
	write64(&b, 0)
	write64(&b, 1)
	writeString(&b, "test.values")
	write32(&b, TypeArray)
	write32(&b, TypeUint8)
	write64(&b, maxArray)
	_, err := Read(bytes.NewReader(b.Bytes()), int64(b.Len()))
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("error = %v, want invalid GGUF", err)
	}
}

func TestReadRejectsInvalidUTF8(t *testing.T) {
	b := fixture()
	i := bytes.Index(b, []byte("general.architecture"))
	b[i] = 0xff
	_, err := Read(bytes.NewReader(b), int64(len(b)))
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("error = %v, want invalid GGUF", err)
	}
}

func TestCheckSHA256RejectsPayloadCorruption(t *testing.T) {
	b := fixture()
	sum := sha256.Sum256(b)
	want := hex.EncodeToString(sum[:])
	if err := CheckSHA256(bytes.NewReader(b), int64(len(b)), want); err != nil {
		t.Fatal(err)
	}
	b[len(b)-1] ^= 1
	if err := CheckSHA256(bytes.NewReader(b), int64(len(b)), want); err == nil {
		t.Fatal("corrupt payload passed validation")
	}
}

func TestAlign(t *testing.T) {
	if got, err := align(97, 32); err != nil || got != 128 {
		t.Fatalf("align = %d, %v", got, err)
	}
	if _, err := align(^uint64(0)-1, 8); err == nil {
		t.Fatal("overflow succeeded")
	}
}

func TestTensorBytes(t *testing.T) {
	tests := []struct {
		name       string
		tensor     Tensor
		want       uint64
		wantErr    error
		wantErrMsg bool
	}{
		{
			name:    "row is not block aligned",
			tensor:  Tensor{Name: "x", TypeID: 16, Dimensions: []uint64{128, 2}},
			wantErr: ErrInvalid,
		},
		{
			name:   "block aligned",
			tensor: Tensor{Name: "x", TypeID: 16, Dimensions: []uint64{256, 2}},
			want:   132,
		},
		{
			name:    "unknown type",
			tensor:  Tensor{Name: "x", TypeID: 4242, Dimensions: []uint64{32}},
			wantErr: ErrUnsupported,
		},
		{
			name:    "byte count overflow",
			tensor:  Tensor{Name: "x", TypeID: 0, Dimensions: []uint64{1 << 40, 1 << 40}},
			wantErr: ErrInvalid,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n, err := tt.tensor.Bytes()
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("error = %v, want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil || n != tt.want {
				t.Fatalf("bytes = %d, %v, want %d", n, err, tt.want)
			}
		})
	}
}

func TestValueUint(t *testing.T) {
	tests := []struct {
		name  string
		value Value
		want  uint64
		ok    bool
	}{
		{"uint8", Value{TypeUint8, uint8(7)}, 7, true},
		{"uint16", Value{TypeUint16, uint16(7)}, 7, true},
		{"uint32", Value{TypeUint32, uint32(7)}, 7, true},
		{"uint64", Value{TypeUint64, uint64(7)}, 7, true},
		{"int32", Value{TypeInt32, int32(7)}, 0, false},
		{"string", Value{TypeString, "7"}, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := tt.value.Uint()
			if got != tt.want || ok != tt.ok {
				t.Fatalf("Uint = %d, %v, want %d, %v", got, ok, tt.want, tt.ok)
			}
		})
	}
}

func TestReadNestedArray(t *testing.T) {
	var b bytes.Buffer
	b.WriteString("GGUF")
	write32(&b, 3)
	write64(&b, 0)
	write64(&b, 1)
	writeString(&b, "test.values")
	write32(&b, TypeArray)
	write32(&b, TypeArray)
	write64(&b, 1)
	write32(&b, TypeUint32)
	write64(&b, 2)
	write32(&b, 7)
	write32(&b, 11)
	if _, err := Read(bytes.NewReader(b.Bytes()), int64(b.Len())); err != nil {
		t.Fatal(err)
	}
}

func TestReadUnalignedTensor(t *testing.T) {
	b := fixture()
	i := bytes.Index(b, []byte("token_embd.weight")) + len("token_embd.weight") + 4 + 16 + 4
	put64(b[i:], 1)
	_, err := Read(bytes.NewReader(b), int64(len(b)))
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("error = %v, want invalid GGUF", err)
	}
}

func TestReadAlignmentMultipleOfEight(t *testing.T) {
	b := fixture()
	i := bytes.Index(b, []byte("general.alignment")) + len("general.alignment") + 4
	put32(b[i:], 24)
	if _, err := Read(bytes.NewReader(b), int64(len(b))); err != nil {
		t.Fatal(err)
	}
}

func FuzzRead(f *testing.F) {
	f.Add(fixture())
	f.Add([]byte("GGUF"))
	f.Fuzz(func(t *testing.T, b []byte) {
		_, _ = Read(bytes.NewReader(b), int64(len(b)))
	})
}

// fixture builds a minimal well-formed GGUF file: two metadata keys and one
// F32 tensor of 4x8 elements.
func fixture() []byte {
	var b bytes.Buffer
	b.WriteString("GGUF")
	write32(&b, 3)
	write64(&b, 1)
	write64(&b, 2)
	writeString(&b, "general.architecture")
	write32(&b, TypeString)
	writeString(&b, "deepseek4")
	writeString(&b, "general.alignment")
	write32(&b, TypeUint32)
	write32(&b, 32)
	writeString(&b, "token_embd.weight")
	write32(&b, 2)
	write64(&b, 4)
	write64(&b, 8)
	write32(&b, 0)
	write64(&b, 0)
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(make([]byte, 128))
	return b.Bytes()
}

func writeString(b *bytes.Buffer, s string) { write64(b, uint64(len(s))); b.WriteString(s) }
func write32(b *bytes.Buffer, n uint32)     { var p [4]byte; put32(p[:], n); b.Write(p[:]) }
func write64(b *bytes.Buffer, n uint64)     { var p [8]byte; put64(p[:], n); b.Write(p[:]) }
func put32(b []byte, n uint32)              { binary.LittleEndian.PutUint32(b, n) }
func put64(b []byte, n uint64)              { binary.LittleEndian.PutUint64(b, n) }
