//go:build darwin

package metalsafetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"syscall"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
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

// Weights represents an opened .safetensors file with memory-mapped tensors.
type Weights struct {
	f        *os.File
	dataBase int64
	dataMMap []byte
	Tensors  map[string]TensorInfo
}

// Open parses the JSON header of a .safetensors file and mmaps the tensor data.
func Open(path string) (*Weights, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("metalsafetensors: open file: %w", err)
	}

	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("metalsafetensors: stat file: %w", err)
	}

	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("metalsafetensors: read header len: %w", err)
	}

	headerBytes := make([]byte, headerLen)
	if _, err := f.ReadAt(headerBytes, 8); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("metalsafetensors: read header bytes: %w", err)
	}

	var rawMap map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawMap); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("metalsafetensors: unmarshal header: %w", err)
	}

	tensors := make(map[string]TensorInfo, len(rawMap))
	for k, v := range rawMap {
		if k == "__metadata__" {
			continue
		}
		var th TensorHeader
		if err := json.Unmarshal(v, &th); err != nil {
			continue
		}
		tensors[k] = TensorInfo{
			Name:        k,
			DType:       th.DType,
			Shape:       th.Shape,
			OffsetStart: th.DataOffsets[0],
			OffsetEnd:   th.DataOffsets[1],
		}
	}

	dataBase := int64(8 + headerLen)
	totalSize := int(st.Size())

	dataMMap, err := syscall.Mmap(int(f.Fd()), 0, totalSize, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("metalsafetensors: mmap file: %w", err)
	}

	return &Weights{
		f:        f,
		dataBase: dataBase,
		dataMMap: dataMMap,
		Tensors:  tensors,
	}, nil
}

// Close unmaps memory and closes the underlying file handle.
func (w *Weights) Close() error {
	if w.dataMMap != nil {
		_ = syscall.Munmap(w.dataMMap)
		w.dataMMap = nil
	}
	if w.f != nil {
		err := w.f.Close()
		w.f = nil
		return err
	}
	return nil
}

// TensorFloat32 reads or decodes a named tensor into float32.
func (w *Weights) TensorFloat32(name string) ([]float32, error) {
	info, ok := w.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("metalsafetensors: tensor %q not found", name)
	}

	start := int(w.dataBase + int64(info.OffsetStart))
	end := int(w.dataBase + int64(info.OffsetEnd))

	if start < 0 || end > len(w.dataMMap) || start > end {
		return nil, fmt.Errorf("metalsafetensors: invalid offset range [%d, %d]", start, end)
	}

	raw := w.dataMMap[start:end]
	return decodeToFloat32(raw, info.DType), nil
}

// TensorMetalSlice allocates a shared Metal unified memory slice and loads the tensor data.
func (w *Weights) TensorMetalSlice(dev *metal.Device, name string) (mempool.Slice[float32], error) {
	floats, err := w.TensorFloat32(name)
	if err != nil {
		return mempool.Slice[float32]{}, err
	}
	return mempool.AllocFromSlice(dev, floats)
}

func decodeToFloat32(raw []byte, dtype string) []float32 {
	switch dtype {
	case "F32":
		count := len(raw) / 4
		out := make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4])
			out[i] = math.Float32frombits(bits)
		}
		return out
	case "BF16":
		count := len(raw) / 2
		out := make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint16(raw[i*2 : (i+1)*2])
			f32Bits := uint32(bits) << 16
			out[i] = math.Float32frombits(f32Bits)
		}
		return out
	case "F16":
		count := len(raw) / 2
		out := make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint16(raw[i*2 : (i+1)*2])
			out[i] = fp16ToFloat32(bits)
		}
		return out
	default:
		return nil
	}
}

func fp16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h&0x7C00) >> 10
	mant := uint32(h & 0x03FF)

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for (mant & 0x0400) == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x03FF
		exp = exp + (127 - 15)
		return math.Float32frombits(sign | (exp << 23) | (mant << 13))
	} else if exp == 0x1F {
		if mant == 0 {
			return math.Float32frombits(sign | 0x7F800000)
		}
		return math.Float32frombits(sign | 0x7FC00000 | (mant << 13))
	}

	exp = exp + (127 - 15)
	return math.Float32frombits(sign | (exp << 23) | (mant << 13))
}
