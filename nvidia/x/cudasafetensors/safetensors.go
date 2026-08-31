package cudasafetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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
	ShardFile   string
}

// Weights represents an opened single-file .safetensors model.
type Weights struct {
	data         []byte
	dataBase     int64
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

// OpenSharded loads a multi-file sharded model directory containing model.safetensors.index.json.
func OpenSharded(dir string) (*ShardedWeights, error) {
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	indexBytes, err := os.ReadFile(indexPath)
	if err != nil {
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

	shards := make(map[string]*Weights)
	allTensors := make(map[string]TensorInfo)

	for tName, shardFilename := range idx.WeightMap {
		if _, loaded := shards[shardFilename]; !loaded {
			shardPath := filepath.Join(dir, shardFilename)
			w, err := Open(shardPath)
			if err != nil {
				return nil, fmt.Errorf("cudasafetensors: open shard %s: %w", shardFilename, err)
			}
			shards[shardFilename] = w
		}

		info := shards[shardFilename].Tensors[tName]
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

// LoadShardedSlice loads a tensor directly from a sharded checkpoint onto dev.
func LoadShardedSlice[T any](sw *ShardedWeights, name string, dev *cuda.Device) (cudamem.Slice[T], error) {
	shardFile, exists := sw.tensorMap[name]
	if !exists {
		return cudamem.Slice[T]{}, fmt.Errorf("cudasafetensors: tensor %q not in weight map", name)
	}
	shard := sw.shards[shardFile]
	return LoadSlice[T](shard, name, dev)
}

// Close releases resources.
func (w *Weights) Close() error {
	w.data = nil
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
