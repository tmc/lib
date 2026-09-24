package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

var allDtypes = []Dtype{BOOL, U8, I8, F8E4M3, F8E5M2, I16, U16, F16, BF16, I32, U32, F32, I64, U64, F64}

func randomTensor(r *rand.Rand, name string, dtype Dtype, shape []int) Tensor {
	n := dtype.Size()
	for _, d := range shape {
		n *= d
	}
	data := make([]byte, n)
	for i := range data {
		data[i] = byte(r.Uint32())
	}
	if dtype == BOOL {
		for i := range data {
			data[i] &= 1
		}
	}
	return Tensor{Name: name, Dtype: dtype, Shape: shape, Data: data}
}

func readAll(t *testing.T, f *File) []Tensor {
	t.Helper()
	var out []Tensor
	for _, info := range f.Tensors {
		x, err := f.Tensor(info.Name)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, x)
	}
	return out
}

func TestRoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	shapes := [][]int{{}, {1}, {3}, {2, 3}, {3, 1, 5}, {0, 4}}
	var tensors []Tensor
	for _, dtype := range allDtypes {
		for i, shape := range shapes {
			tensors = append(tensors, randomTensor(r, string(dtype)+"."+string(rune('a'+i)), dtype, shape))
		}
	}
	tests := []struct {
		name     string
		tensors  []Tensor
		metadata map[string]string
	}{
		{"empty", nil, nil},
		{"metadata only", nil, map[string]string{"format": "pt"}},
		{"all dtypes", tensors, map[string]string{"format": "pt", "step": "100", "quote": `"x"\n`}},
		{"reversed", reverse(tensors), nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := Write(&buf, tt.tensors, tt.metadata); err != nil {
				t.Fatal(err)
			}
			hlen := binary.LittleEndian.Uint64(buf.Bytes())
			if (8+hlen)%8 != 0 {
				t.Errorf("data section starts at %d, not a multiple of 8", 8+hlen)
			}
			f, err := NewFile(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
			if err != nil {
				t.Fatal(err)
			}
			got := readAll(t, f)
			if len(tt.tensors) == 0 {
				if len(got) != 0 {
					t.Fatalf("read %d tensors, want 0", len(got))
				}
			} else if !reflect.DeepEqual(got, tt.tensors) {
				t.Errorf("round trip changed tensors")
			}
			if len(tt.metadata) == 0 && f.Metadata != nil || len(tt.metadata) > 0 && !reflect.DeepEqual(f.Metadata, tt.metadata) {
				t.Errorf("metadata = %v, want %v", f.Metadata, tt.metadata)
			}
		})
	}
}

func reverse(ts []Tensor) []Tensor {
	out := make([]Tensor, len(ts))
	for i, t := range ts {
		out[len(ts)-1-i] = t
	}
	return out
}

func TestWriteRejects(t *testing.T) {
	ok := Tensor{Name: "a", Dtype: F32, Shape: []int{2}, Data: make([]byte, 8)}
	tests := []struct {
		name    string
		tensors []Tensor
		want    string
	}{
		{"duplicate", []Tensor{ok, ok}, "duplicate tensor name"},
		{"empty name", []Tensor{{Dtype: F32, Shape: []int{}, Data: make([]byte, 4)}}, "empty name"},
		{"reserved", []Tensor{{Name: "__metadata__", Dtype: F32, Shape: []int{}, Data: make([]byte, 4)}}, "reserved"},
		{"dtype", []Tensor{{Name: "a", Dtype: "F12", Shape: []int{1}, Data: make([]byte, 4)}}, "unknown dtype"},
		{"short", []Tensor{{Name: "a", Dtype: F32, Shape: []int{3}, Data: make([]byte, 8)}}, "data bytes"},
		{"negative", []Tensor{{Name: "a", Dtype: F32, Shape: []int{-1}}}, "negative"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := Write(&buf, tt.tensors, nil)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Write error = %v, want %q", err, tt.want)
			}
			if buf.Len() != 0 {
				t.Errorf("Write wrote %d bytes before failing", buf.Len())
			}
		})
	}
}

// file assembles a safetensors file from a raw header and data section.
func file(header string, data int) []byte {
	b := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	b = append(b, header...)
	return append(b, make([]byte, data)...)
}

func TestNewFileRejects(t *testing.T) {
	tests := []struct {
		name string
		file []byte
		want string
	}{
		{"duplicate name", file(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`, 8), "duplicate"},
		{"duplicate metadata", file(`{"__metadata__":{},"__metadata__":{}}`, 0), "duplicate"},
		{"gap", file(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}`, 12), "not contiguous"},
		{"overlap", file(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`, 8), "not contiguous"},
		{"trailing data", file(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 8), "cover 4 of 8"},
		{"size mismatch", file(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`, 4), "do not hold"},
		{"unknown dtype", file(`{"a":{"dtype":"Q4","shape":[1],"data_offsets":[0,1]}}`, 1), "unknown dtype"},
		{"missing offsets", file(`{"a":{"dtype":"F32","shape":[1]}}`, 4), "missing"},
		{"not an object", file(`[1]`, 0), "not a JSON object"},
		{"trailing garbage", file(`{} x`, 0), "trailing"},
		{"bad metadata", file(`{"__metadata__":{"a":1}}`, 0), "metadata"},
		{"header too long", binary.LittleEndian.AppendUint64(nil, 100), "header length"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewFile(bytes.NewReader(tt.file), int64(len(tt.file)))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("NewFile error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestWriteFile(t *testing.T) {
	dir := t.TempDir()
	name := filepath.Join(dir, "m.safetensors")
	a := []Tensor{{Name: "x", Dtype: F16, Shape: []int{3}, Data: []byte{1, 2, 3, 4, 5, 6}}}
	if err := WriteFile(name, a, map[string]string{"v": "1"}); err != nil {
		t.Fatal(err)
	}
	// A failing write must leave the previous file intact and no temporary behind.
	bad := []Tensor{{Name: "x", Dtype: F16, Shape: []int{3}, Data: []byte{1}}}
	if err := WriteFile(name, bad, nil); err == nil {
		t.Fatal("WriteFile accepted a short tensor")
	}
	f, err := Open(name)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if got := readAll(t, f); !reflect.DeepEqual(got, a) || f.Metadata["v"] != "1" {
		t.Errorf("file holds %v %v, want %v", got, f.Metadata, a)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("directory holds %d entries, want only the output file", len(entries))
	}
}

// TestFrozenCopy copies a tensor from one file to another through Tensor and
// Write, as a trainer does with frozen weights, and checks the bytes survive.
func TestFrozenCopy(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	src := randomTensor(r, "embed", F16, []int{7, 5})
	var buf bytes.Buffer
	if err := Write(&buf, []Tensor{src}, nil); err != nil {
		t.Fatal(err)
	}
	in, err := NewFile(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	frozen, err := in.Tensor("embed")
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	head := Tensor{Name: "head", Dtype: F32, Shape: []int{1}, Data: make([]byte, 4)}
	if err := Write(&out, []Tensor{head, frozen}, nil); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out.Bytes(), src.Data) {
		t.Error("frozen tensor bytes changed")
	}
}

// torchFixture is a file written by safetensors.torch.save_file, stored as
// hex. It was produced with
//
//	import torch
//	from safetensors.torch import save_file
//	save_file({"w": torch.tensor([[1.5, -2.0], [0.25, 3.0]], dtype=torch.float16),
//	           "b": torch.tensor([1.0, -1.0, 65504.0], dtype=torch.bfloat16),
//	           "n": torch.tensor([7, -8], dtype=torch.int64)},
//	          "f.safetensors", metadata={"format": "pt"})
//
// and hex-encoded with xxd -p.
const torchFixture = "testdata/torch_save_file.hex"

func TestReadTorchFile(t *testing.T) {
	text, err := os.ReadFile(torchFixture)
	if err != nil {
		t.Fatal(err)
	}
	data, err := hex.DecodeString(strings.Join(strings.Fields(string(text)), ""))
	if err != nil {
		t.Fatal(err)
	}
	f, err := NewFile(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if f.Metadata["format"] != "pt" {
		t.Errorf("metadata = %v", f.Metadata)
	}
	check := func(name string, dtype Dtype, shape []int, want []float32) {
		x, err := f.Tensor(name)
		if err != nil {
			t.Fatal(err)
		}
		if x.Dtype != dtype || !reflect.DeepEqual(x.Shape, shape) {
			t.Errorf("%s: %s%v, want %s%v", name, x.Dtype, x.Shape, dtype, shape)
		}
		if want == nil {
			return
		}
		got, err := DecodeFloat32(x.Dtype, x.Data)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
	check("w", F16, []int{2, 2}, []float32{1.5, -2, 0.25, 3})
	check("b", BF16, []int{3}, []float32{1, -1, 65536})
	check("n", I64, []int{2}, nil)
}

// TestPythonLoad writes a file and loads it with safetensors.torch.load_file.
// Importing torch can take minutes, so the test runs only when
// SAFETENSORS_TORCH_TEST=1, and it fails if python3 lacks torch or
// safetensors.
func TestPythonLoad(t *testing.T) {
	if os.Getenv("SAFETENSORS_TORCH_TEST") != "1" {
		t.Skip("set SAFETENSORS_TORCH_TEST=1 to load the output with safetensors.torch")
	}
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal(err)
	}
	f16, _ := EncodeFloat32(F16, []float32{1.5, -2, 0.25, 3, 65504, -0.0})
	bf16, _ := EncodeFloat32(BF16, []float32{1, -1, 3.140625})
	f32, _ := EncodeFloat32(F32, []float32{0.1, 1e-30, -7})
	tensors := []Tensor{
		{Name: "odd.f16", Dtype: F16, Shape: []int{2, 3}, Data: f16},
		{Name: "bf16", Dtype: BF16, Shape: []int{3}, Data: bf16},
		{Name: "f32", Dtype: F32, Shape: []int{3, 1}, Data: f32},
		{Name: "i8", Dtype: I8, Shape: []int{3}, Data: []byte{1, 0xff, 0x80}},
		{Name: "scalar", Dtype: I64, Shape: []int{}, Data: binary.LittleEndian.AppendUint64(nil, 42)},
		{Name: "flag", Dtype: BOOL, Shape: []int{2}, Data: []byte{1, 0}},
	}
	name := filepath.Join(t.TempDir(), "go.safetensors")
	if err := WriteFile(name, tensors, map[string]string{"format": "pt", "writer": "go"}); err != nil {
		t.Fatal(err)
	}
	script := `
import sys, torch
from safetensors import safe_open
from safetensors.torch import load_file
d = load_file(sys.argv[1])
want = {
    "odd.f16": torch.tensor([[1.5, -2.0, 0.25], [3.0, 65504.0, -0.0]], dtype=torch.float16),
    "bf16": torch.tensor([1.0, -1.0, 3.140625], dtype=torch.bfloat16),
    "f32": torch.tensor([[0.1], [1e-30], [-7.0]], dtype=torch.float32),
    "i8": torch.tensor([1, -1, -128], dtype=torch.int8),
    "scalar": torch.tensor(42, dtype=torch.int64),
    "flag": torch.tensor([True, False]),
}
assert list(d) == list(want) or set(d) == set(want), list(d)
for k, v in want.items():
    assert d[k].dtype == v.dtype and d[k].shape == v.shape and torch.equal(d[k], v), (k, d[k], v)
with safe_open(sys.argv[1], framework="pt") as f:
    assert f.metadata() == {"format": "pt", "writer": "go"}, f.metadata()
print("ok")
`
	out, err := exec.Command(python, "-c", script, name).CombinedOutput()
	if err != nil || strings.TrimSpace(string(out)) != "ok" {
		t.Fatalf("python load_file: %v\n%s", err, out)
	}
}
