package cudasafetensors

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/x/safetensors"
)

// hostAllocation is a device allocation in host memory: it keeps the bytes
// copied into it, so a test can compare them with the tensor's.
type hostAllocation[T any] struct {
	length int
	got    *[]byte
}

func (a hostAllocation[T]) CopyFromHost(src []T) error {
	if len(src) != a.length {
		return fmt.Errorf("copy %d elements into a %d-element allocation", len(src), a.length)
	}
	var elem T
	*a.got = bytes.Clone(unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(src))), len(src)*int(unsafe.Sizeof(elem))))
	return nil
}

func (a hostAllocation[T]) Free() error { return nil }

// loadHost loads the named tensor through loadSlice into host memory and
// returns the bytes the copy delivered.
func loadHost[T any](r TensorReader, name string) ([]byte, error) {
	var got []byte
	_, err := loadSlice(r, name, func(n int) (hostAllocation[T], error) {
		return hostAllocation[T]{length: n, got: &got}, nil
	})
	return got, err
}

// loadAs loads the named tensor with the storage type its dtype calls for.
func loadAs(r TensorReader, name string, dtype safetensors.Dtype) ([]byte, error) {
	switch dtype {
	case safetensors.BOOL:
		return loadHost[bool](r, name)
	case safetensors.I8:
		return loadHost[int8](r, name)
	case safetensors.I16:
		return loadHost[int16](r, name)
	case safetensors.I32:
		return loadHost[int32](r, name)
	case safetensors.I64:
		return loadHost[int64](r, name)
	case safetensors.U8, safetensors.F8E4M3, safetensors.F8E5M2:
		return loadHost[uint8](r, name)
	case safetensors.U16, safetensors.F16, safetensors.BF16:
		return loadHost[uint16](r, name)
	case safetensors.U32:
		return loadHost[uint32](r, name)
	case safetensors.U64:
		return loadHost[uint64](r, name)
	case safetensors.F32:
		return loadHost[float32](r, name)
	case safetensors.F64:
		return loadHost[float64](r, name)
	}
	return nil, fmt.Errorf("no storage type for dtype %s", dtype)
}

// FuzzLoadSliceCopiesWholeTensor loads every tensor of an arbitrary
// safetensors file into host memory through loadSlice, the half of LoadSlice
// that runs without a device. A nonempty tensor must load, and the bytes
// copied must be the tensor's bytes, all of them. Loading as float32 must
// succeed exactly for a nonempty F32 tensor.
func FuzzLoadSliceCopiesWholeTensor(f *testing.F) {
	text, err := os.ReadFile("../safetensors/testdata/torch_save_file.hex")
	if err != nil {
		f.Fatal(err)
	}
	torch, err := hex.DecodeString(strings.Join(strings.Fields(string(text)), ""))
	if err != nil {
		f.Fatal(err)
	}
	f.Add(torch)
	var buf bytes.Buffer
	err = safetensors.Write(&buf, []safetensors.Tensor{
		{Name: "f", Dtype: safetensors.F32, Shape: []int{3}, Data: make([]byte, 12)},
		{Name: "e", Dtype: safetensors.F32, Shape: []int{0}},
		{Name: "b", Dtype: safetensors.BOOL, Shape: []int{1, 2}, Data: []byte{1, 0}},
		{Name: "d", Dtype: safetensors.F64, Shape: []int{1}, Data: make([]byte, 8)},
	}, nil)
	if err != nil {
		f.Fatal(err)
	}
	f.Add(buf.Bytes())
	f.Fuzz(func(t *testing.T, data []byte) {
		file, err := safetensors.NewFile(bytes.NewReader(data), int64(len(data)))
		if err != nil {
			return
		}
		for _, info := range file.Tensors {
			want, err := file.Tensor(info.Name)
			if err != nil {
				t.Fatalf("tensor %q: %v", info.Name, err)
			}
			got, err := loadAs(file, info.Name, info.Dtype)
			switch {
			case len(want.Data) == 0:
				if err == nil {
					t.Errorf("tensor %q has no elements but loaded", info.Name)
				}
			case err != nil:
				t.Errorf("tensor %q (%s%v): %v", info.Name, info.Dtype, info.Shape, err)
			case !bytes.Equal(got, want.Data):
				t.Errorf("tensor %q (%s%v): copied %d bytes, want the tensor's %d", info.Name, info.Dtype, info.Shape, len(got), len(want.Data))
			}
			_, err = loadHost[float32](file, info.Name)
			if ok := info.Dtype == safetensors.F32 && len(want.Data) > 0; (err == nil) != ok {
				t.Errorf("tensor %q (%s%v) as float32: error %v, want success %v", info.Name, info.Dtype, info.Shape, err, ok)
			}
		}
	})
}
