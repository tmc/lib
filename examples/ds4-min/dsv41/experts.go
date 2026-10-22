package dsv41

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
	"github.com/tmc/lib/gguf"
)

// ExpertTriplets describes the serialized routed experts in a verified model.
// The returned map is detached. Identity binds cache keys to the complete file,
// exact GGUF encodings and serialized extents. Unverified Bind models are refused.
//
// This admits file reads only, not a native quantization consumer or forward
// execution. The owner must separately qualify consumers for each tensor encoding
// and keep the same bound file reader alive and immutable when using NewSource.
func (m *Model) ExpertTriplets() (map[int]dsv4expertcache.Triplet, dsv4expertcache.Identity, error) {
	var identity dsv4expertcache.Identity
	if m == nil || m.catalog == nil || m.assetSHA == ([32]byte{}) || m.fileSize > math.MaxInt64 {
		return nil, identity, fmt.Errorf("unverified V4.1 expert source")
	}
	quant, layout := sha256.New(), sha256.New()
	quant.Write([]byte("dsv41/routed-gguf-encodings/v1\x00"))
	layout.Write([]byte("dsv41/routed-serialized-layout/v1\x00"))
	layers := make(map[int]dsv4expertcache.Triplet, 40)
	requirements := make(map[string]requirement)
	for _, need := range tensorRequirements() {
		requirements[need.name] = need
	}
	for layer := 0; layer < 40; layer++ {
		var parts [3]dsv4expertcache.Tensor
		var types [3]uint32
		for component, suffix := range []string{"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"} {
			name := fmt.Sprintf("blk.%d.%s", layer, suffix)
			t, ok := m.Tensor(name)
			if !ok {
				return nil, identity, fmt.Errorf("missing V4.1 expert tensor %q", name)
			}
			part, err := expertExtent(t, requirements[name], m.fileSize)
			if err != nil {
				return nil, identity, err
			}
			parts[component], types[component] = part, t.Encoding
			// Fixed name length makes the identity independent of concatenation ambiguity.
			writeExpertWord(quant, uint64(len(name)))
			quant.Write([]byte(name))
			writeExpertWord(quant, uint64(t.Encoding))
			writeExpertWord(layout, uint64(layer))
			writeExpertWord(layout, uint64(component))
			for _, dimension := range t.Dimensions {
				writeExpertWord(layout, dimension)
			}
			for _, value := range []uint64{part.Offset, part.Bytes, part.Stride, uint64(part.Experts)} {
				writeExpertWord(layout, value)
			}
		}
		if types[0] != types[1] {
			return nil, identity, fmt.Errorf("layer %d expert gate/up encoding mismatch", layer)
		}
		layers[layer] = dsv4expertcache.Triplet{Gate: parts[0], Up: parts[1], Down: parts[2]}
	}
	identity.Model = m.assetSHA
	copy(identity.Quant[:], quant.Sum(nil))
	copy(identity.Layout[:], layout.Sum(nil))
	return layers, identity, nil
}

func writeExpertWord(h hash.Hash, value uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], value)
	h.Write(b[:])
}

func expertExtent(t TensorDescriptor, need requirement, size uint64) (dsv4expertcache.Tensor, error) {
	var zero dsv4expertcache.Tensor
	// Geometry was admitted by Bind. Checking the descriptor here also prevents
	// accidental publication of a different family or a partially built Model.
	if need.name != t.Name || len(need.dims) != 3 || len(t.Dimensions) != 3 {
		return zero, fmt.Errorf("tensor %q: invalid V4.1 expert layout", t.Name)
	}
	for i, dimension := range need.dims {
		if t.Dimensions[i] != uint64(dimension) {
			return zero, fmt.Errorf("tensor %q: invalid V4.1 expert layout", t.Name)
		}
	}
	if t.Encoding >= 64 || need.types&(uint64(1)<<t.Encoding) == 0 {
		return zero, fmt.Errorf("tensor %q: invalid V4.1 expert encoding %d", t.Name, t.Encoding)
	}
	serialized := gguf.Tensor{TypeID: t.Encoding, Dimensions: t.Dimensions}
	n, err := serialized.Bytes()
	if err != nil {
		return zero, fmt.Errorf("tensor %q: %w", t.Name, err)
	}
	experts := uint64(need.dims[2])
	if t.Offset < 0 || t.Bytes <= 0 || n != uint64(t.Bytes) || n%experts != 0 ||
		uint64(t.Offset) > size || n > size-uint64(t.Offset) {
		return zero, fmt.Errorf("tensor %q: invalid V4.1 expert extent", t.Name)
	}
	return dsv4expertcache.Tensor{Offset: uint64(t.Offset), Bytes: n, Stride: n / experts, Experts: int(experts)}, nil
}
