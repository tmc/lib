package cudasafetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"unsafe"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

const maxHeaderSize = 100 << 20

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
	ShardFile   string
}

// Weights represents an opened single-file .safetensors model.
type Weights struct {
	f            *os.File
	dataBase     int64
	fileSize     int64
	Tensors      map[string]TensorInfo
	MetadataJSON map[string]string
}

// ShardedWeights represents a collection of sharded .safetensors files mapped via an index.
type ShardedWeights struct {
	dir       string
	shards    map[string]*Weights
	tensorMap map[string]string
	Tensors   map[string]TensorInfo
}

// Open reads the header of a .safetensors file and parses tensor metadata without reading entire file into memory.
func Open(path string) (*Weights, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cudasafetensors: open file: %w", err)
	}

	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("cudasafetensors: stat file: %w", err)
	}
	fileSize := st.Size()
	if fileSize < 8 {
		_ = f.Close()
		return nil, fmt.Errorf("cudasafetensors: file too small (%d bytes)", fileSize)
	}

	var headerLenBuf [8]byte
	if _, err := f.ReadAt(headerLenBuf[:], 0); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("cudasafetensors: read header length: %w", err)
	}
	headerLen := binary.LittleEndian.Uint64(headerLenBuf[:])
	if headerLen > maxHeaderSize {
		_ = f.Close()
		return nil, fmt.Errorf("cudasafetensors: header too large (%d bytes)", headerLen)
	}
	if headerLen > uint64(fileSize-8) {
		_ = f.Close()
		return nil, fmt.Errorf("cudasafetensors: truncated header (expected %d bytes, got %d)", 8+headerLen, fileSize)
	}

	headerJSON := make([]byte, headerLen)
	if _, err := f.ReadAt(headerJSON, 8); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("cudasafetensors: read header JSON: %w", err)
	}

	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &rawHeader); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("cudasafetensors: unmarshal header: %w", err)
	}

	tensors := make(map[string]TensorInfo)
	metadata := make(map[string]string)

	for k, v := range rawHeader {
		if k == "__metadata__" {
			if err := json.Unmarshal(v, &metadata); err != nil {
				_ = f.Close()
				return nil, fmt.Errorf("cudasafetensors: invalid metadata: %w", err)
			}
			continue
		}
		if k == "" {
			_ = f.Close()
			return nil, fmt.Errorf("cudasafetensors: empty tensor name")
		}

		var th TensorHeader
		if err := json.Unmarshal(v, &th); err != nil {
			_ = f.Close()
			return nil, fmt.Errorf("cudasafetensors: tensor %q header: %w", k, err)
		}
		if err := validateTensorHeader(k, th, uint64(fileSize)-8-headerLen); err != nil {
			_ = f.Close()
			return nil, err
		}
		tensors[k] = TensorInfo{Name: k, DType: th.DType, Shape: append([]int(nil), th.Shape...), OffsetStart: th.DataOffsets[0], OffsetEnd: th.DataOffsets[1]}
	}
	if err := validateCoverage(tensors, uint64(fileSize)-8-headerLen); err != nil {
		_ = f.Close()
		return nil, err
	}

	return &Weights{
		f:            f,
		dataBase:     int64(8 + headerLen),
		fileSize:     fileSize,
		Tensors:      tensors,
		MetadataJSON: metadata,
	}, nil
}

func validateTensorHeader(name string, h TensorHeader, dataSize uint64) error {
	if len(h.DataOffsets) != 2 {
		return fmt.Errorf("cudasafetensors: tensor %q has %d data offsets, want 2", name, len(h.DataOffsets))
	}
	start, end := h.DataOffsets[0], h.DataOffsets[1]
	if start > end || end > dataSize {
		return fmt.Errorf("cudasafetensors: tensor %q offsets [%d,%d] exceed data size %d", name, start, end, dataSize)
	}
	size, ok := dtypeSize(h.DType)
	if !ok {
		return fmt.Errorf("cudasafetensors: tensor %q has unsupported dtype %q", name, h.DType)
	}
	elements := uint64(1)
	for i, dim := range h.Shape {
		if dim < 0 {
			return fmt.Errorf("cudasafetensors: tensor %q has negative dimension %d", name, i)
		}
		if dim != 0 && elements > math.MaxUint64/uint64(dim) {
			return fmt.Errorf("cudasafetensors: tensor %q shape overflows", name)
		}
		elements *= uint64(dim)
	}
	if elements > math.MaxUint64/size || end-start != elements*size {
		return fmt.Errorf("cudasafetensors: tensor %q byte length %d does not match dtype %s shape %v", name, end-start, h.DType, h.Shape)
	}
	return nil
}

func dtypeSize(dtype string) (uint64, bool) {
	switch dtype {
	case "BOOL", "U8", "I8", "F8_E4M3", "F8_E5M2":
		return 1, true
	case "U16", "I16", "F16", "BF16":
		return 2, true
	case "U32", "I32", "F32":
		return 4, true
	case "U64", "I64", "F64":
		return 8, true
	}
	return 0, false
}

func validateCoverage(tensors map[string]TensorInfo, dataSize uint64) error {
	infos := make([]TensorInfo, 0, len(tensors))
	for _, info := range tensors {
		infos = append(infos, info)
	}
	sort.Slice(infos, func(i, j int) bool { return infos[i].OffsetStart < infos[j].OffsetStart })
	next := uint64(0)
	for _, info := range infos {
		if info.OffsetStart != next {
			return fmt.Errorf("cudasafetensors: tensor %q starts at %d, want %d", info.Name, info.OffsetStart, next)
		}
		next = info.OffsetEnd
	}
	if next != dataSize {
		return fmt.Errorf("cudasafetensors: tensor data covers %d bytes, file contains %d", next, dataSize)
	}
	return nil
}

// OpenSharded loads a multi-file sharded model directory containing model.safetensors.index.json.
func OpenSharded(dir string) (*ShardedWeights, error) {
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	indexBytes, err := os.ReadFile(indexPath)
	if err != nil {
		if !os.IsNotExist(err) {
			return nil, fmt.Errorf("cudasafetensors: read index: %w", err)
		}
		// Fallback: single file model.safetensors
		singlePath := filepath.Join(dir, "model.safetensors")
		w, singleErr := Open(singlePath)
		if singleErr != nil {
			return nil, fmt.Errorf("cudasafetensors: no index found and single file failed: %w", singleErr)
		}
		sw := &ShardedWeights{
			dir:       dir,
			shards:    map[string]*Weights{"model.safetensors": w},
			tensorMap: make(map[string]string),
			Tensors:   w.Tensors,
		}
		for tName := range w.Tensors {
			sw.tensorMap[tName] = "model.safetensors"
		}
		return sw, nil
	}

	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(indexBytes, &idx); err != nil {
		return nil, fmt.Errorf("cudasafetensors: parse index json: %w", err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("cudasafetensors: index has empty weight map")
	}

	shards := make(map[string]*Weights)
	allTensors := make(map[string]TensorInfo)

	for tName, shardFilename := range idx.WeightMap {
		if tName == "" || shardFilename == "" || filepath.Base(shardFilename) != shardFilename {
			closeShards(shards)
			return nil, fmt.Errorf("cudasafetensors: invalid shard mapping %q to %q", tName, shardFilename)
		}
		if _, loaded := shards[shardFilename]; !loaded {
			shardPath := filepath.Join(dir, shardFilename)
			w, err := Open(shardPath)
			if err != nil {
				closeShards(shards)
				return nil, fmt.Errorf("cudasafetensors: open shard %s: %w", shardFilename, err)
			}
			shards[shardFilename] = w
		}

		info, ok := shards[shardFilename].Tensors[tName]
		if !ok {
			closeShards(shards)
			return nil, fmt.Errorf("cudasafetensors: tensor %q missing from shard %q", tName, shardFilename)
		}
		info.ShardFile = shardFilename
		allTensors[tName] = info
	}

	return &ShardedWeights{
		dir:       dir,
		shards:    shards,
		tensorMap: idx.WeightMap,
		Tensors:   allTensors,
	}, nil
}

func closeShards(shards map[string]*Weights) {
	for _, shard := range shards {
		_ = shard.Close()
	}
}

// TensorBytes returns the raw bytes slice corresponding to tensor name.
func (w *Weights) TensorBytes(name string) ([]byte, TensorInfo, error) {
	info, exists := w.Tensors[name]
	if !exists {
		return nil, TensorInfo{}, fmt.Errorf("cudasafetensors: tensor %q not found", name)
	}

	start := uint64(w.dataBase) + info.OffsetStart
	end := uint64(w.dataBase) + info.OffsetEnd

	if end > uint64(w.fileSize) || start > end {
		return nil, TensorInfo{}, fmt.Errorf("cudasafetensors: tensor %q offset out of bounds", name)
	}

	buf := make([]byte, end-start)
	if _, err := w.f.ReadAt(buf, int64(start)); err != nil {
		return nil, TensorInfo{}, fmt.Errorf("cudasafetensors: read tensor %q: %w", name, err)
	}

	return buf, info, nil
}

// TensorBytes finds and returns raw bytes from the appropriate shard in a sharded model.
func (sw *ShardedWeights) TensorBytes(name string) ([]byte, TensorInfo, error) {
	shardFile, exists := sw.tensorMap[name]
	if !exists {
		return nil, TensorInfo{}, fmt.Errorf("cudasafetensors: tensor %q not in weight map", name)
	}
	shard, exists := sw.shards[shardFile]
	if !exists {
		return nil, TensorInfo{}, fmt.Errorf("cudasafetensors: shard %q not loaded", shardFile)
	}
	return shard.TensorBytes(name)
}

// TensorProvider provides tensor metadata and raw bytes for model parameters.
type TensorProvider interface {
	TensorBytes(name string) ([]byte, TensorInfo, error)
}

// LoadSlice allocates device memory on dev and copies tensor data to GPU from any TensorProvider.
func LoadSlice[T any](provider TensorProvider, name string, dev *cuda.Device) (cudamem.Slice[T], error) {
	return loadSlice(provider, name, func(length int) (cudamem.Slice[T], error) {
		return cudamem.Alloc[T](dev, length)
	})
}

type deviceAllocation[T any] interface {
	CopyFromHost([]T) error
	Free() error
}

func loadSlice[T any, A deviceAllocation[T]](provider TensorProvider, name string, alloc func(int) (A, error)) (A, error) {
	var zero A
	rawBytes, info, err := provider.TensorBytes(name)
	if err != nil {
		return zero, err
	}

	var t T
	elemSize := int(unsafe.Sizeof(t))
	if elemSize == 0 {
		elemSize = 1
	}
	if !compatibleStorage(reflect.TypeOf(t), info.DType) {
		return zero, fmt.Errorf("cudasafetensors: tensor %s dtype %s is incompatible with %T storage", info.Name, info.DType, t)
	}

	numElements := len(rawBytes) / elemSize
	if numElements == 0 {
		return zero, fmt.Errorf("cudasafetensors: zero elements for tensor %s", info.Name)
	}

	deviceSlice, err := alloc(numElements)
	if err != nil {
		return zero, fmt.Errorf("cudasafetensors: alloc device memory: %w", err)
	}

	// Cast rawBytes to []T
	hostSlice := unsafe.Slice((*T)(unsafe.Pointer(&rawBytes[0])), numElements)

	if err := deviceSlice.CopyFromHost(hostSlice); err != nil {
		if freeErr := deviceSlice.Free(); freeErr != nil {
			err = errors.Join(err, fmt.Errorf("free allocation: %w", freeErr))
		}
		return zero, fmt.Errorf("cudasafetensors: copy to device: %w", err)
	}

	return deviceSlice, nil
}

func compatibleStorage(t reflect.Type, dtype string) bool {
	if t == nil {
		return false
	}
	switch t.Kind() {
	case reflect.Bool:
		return dtype == "BOOL"
	case reflect.Int8:
		return dtype == "I8"
	case reflect.Int16:
		return dtype == "I16"
	case reflect.Int32:
		return dtype == "I32"
	case reflect.Int64:
		return dtype == "I64"
	case reflect.Uint8:
		return dtype == "U8" || dtype == "F8_E4M3" || dtype == "F8_E5M2"
	case reflect.Uint16:
		return dtype == "U16" || dtype == "F16" || dtype == "BF16"
	case reflect.Uint32:
		return dtype == "U32"
	case reflect.Uint64:
		return dtype == "U64"
	case reflect.Float32:
		return dtype == "F32"
	case reflect.Float64:
		return dtype == "F64"
	}
	return false
}

// LoadShardedSlice loads a tensor directly from a sharded checkpoint onto dev.
func LoadShardedSlice[T any](sw *ShardedWeights, name string, dev *cuda.Device) (cudamem.Slice[T], error) {
	return LoadSlice[T](sw, name, dev)
}

// Close releases resources.
func (w *Weights) Close() error {
	if w.f != nil {
		err := w.f.Close()
		w.f = nil
		return err
	}
	return nil
}

// Close releases all loaded shards.
func (sw *ShardedWeights) Close() error {
	for _, shard := range sw.shards {
		_ = shard.Close()
	}
	sw.shards = nil
	return nil
}
