package cudasafetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"unsafe"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// TensorHeader describes a single tensor entry in the safetensors JSON header.
type TensorHeader struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

// TensorInfo contains parsed metadata and byte offsets for a tensor.
type TensorInfo struct {
	Name        string
	DType       string
	Shape       []int
	OffsetStart uint64
	OffsetEnd   uint64
}

// Weights represents an opened .safetensors model file.
type Weights struct {
	file         *os.File
	data         []byte
	dataBase     int64
	Tensors      map[string]TensorInfo
	MetadataJSON map[string]string
}

// Open reads the header of a .safetensors file and parses tensor metadata.
func Open(path string) (*Weights, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cudasafetensors: read file: %w", err)
	}

	if len(data) < 8 {
		return nil, fmt.Errorf("cudasafetensors: file too small (%d bytes)", len(data))
	}

	headerLen := binary.LittleEndian.Uint64(data[:8])
	if uint64(len(data)) < 8+headerLen {
		return nil, fmt.Errorf("cudasafetensors: truncated header (expected %d bytes, got %d)", 8+headerLen, len(data))
	}

	headerJSON := data[8 : 8+headerLen]
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &rawHeader); err != nil {
		return nil, fmt.Errorf("cudasafetensors: unmarshal header: %w", err)
	}

	tensors := make(map[string]TensorInfo)
	metadata := make(map[string]string)

	for k, v := range rawHeader {
		if k == "__metadata__" {
			_ = json.Unmarshal(v, &metadata)
			continue
		}

		var th TensorHeader
		if err := json.Unmarshal(v, &th); err != nil {
			continue
		}

		if len(th.DataOffsets) == 2 {
			tensors[k] = TensorInfo{
				Name:        k,
				DType:       th.DType,
				Shape:       th.Shape,
				OffsetStart: th.DataOffsets[0],
				OffsetEnd:   th.DataOffsets[1],
			}
		}
	}

	return &Weights{
		data:         data,
		dataBase:     int64(8 + headerLen),
		Tensors:      tensors,
		MetadataJSON: metadata,
	}, nil
}

// TensorBytes returns the raw bytes slice corresponding to tensor name.
func (w *Weights) TensorBytes(name string) ([]byte, TensorInfo, error) {
	info, exists := w.Tensors[name]
	if !exists {
		return nil, TensorInfo{}, fmt.Errorf("cudasafetensors: tensor %q not found", name)
	}

	start := uint64(w.dataBase) + info.OffsetStart
	end := uint64(w.dataBase) + info.OffsetEnd

	if end > uint64(len(w.data)) || start > end {
		return nil, TensorInfo{}, fmt.Errorf("cudasafetensors: tensor %q offset out of bounds", name)
	}

	return w.data[start:end], info, nil
}

// LoadSlice allocates device memory on dev and copies tensor data to GPU.
func LoadSlice[T any](w *Weights, name string, dev *cuda.Device) (cudamem.Slice[T], error) {
	rawBytes, info, err := w.TensorBytes(name)
	if err != nil {
		return cudamem.Slice[T]{}, err
	}

	var t T
	elemSize := int(unsafe.Sizeof(t))
	if elemSize == 0 {
		elemSize = 1
	}

	numElements := len(rawBytes) / elemSize
	if numElements == 0 {
		return cudamem.Slice[T]{}, fmt.Errorf("cudasafetensors: zero elements for tensor %s", info.Name)
	}

	deviceSlice, err := cudamem.Alloc[T](dev, numElements)
	if err != nil {
		return cudamem.Slice[T]{}, fmt.Errorf("cudasafetensors: alloc device memory: %w", err)
	}

	// Cast rawBytes to []T
	hostSlice := unsafe.Slice((*T)(unsafe.Pointer(&rawBytes[0])), numElements)

	if err := deviceSlice.CopyFromHost(hostSlice); err != nil {
		_ = deviceSlice.Free()
		return cudamem.Slice[T]{}, fmt.Errorf("cudasafetensors: copy to device: %w", err)
	}


	return deviceSlice, nil
}

// Close releases resources.
func (w *Weights) Close() error {
	w.data = nil
	return nil
}
