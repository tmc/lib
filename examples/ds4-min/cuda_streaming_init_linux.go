//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"math"
	"strings"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
)

// loadStreamingExperts admits uniform IQ2_XXS/Q2_K routed weights before any
// model warming. The resident loader replaces those raw tensors with zero
// addresses and loads only the non-routed spans. The source file stays owned
// by the foundation until the selection owner has closed.
func (r *cudaFoundation) loadStreamingExperts(o streamingOptions, size uint64) error {
	triplets, err := streamingTriplets(r.gguf, cudaLayers)
	if err != nil {
		return err
	}
	base := triplets[0]
	for layer := range cudaLayers {
		tr := triplets[layer]
		if tr.Gate.Stride != base.Gate.Stride || tr.Up.Stride != base.Up.Stride || tr.Down.Stride != base.Down.Stride {
			return errors.New("CUDA streaming requires uniform expert strides")
		}
	}
	for _, t := range r.gguf.Tensors {
		if !cudaExpertReplaced(t.Name) {
			continue
		}
		want, width, rows := uint32(typeIQ2XXS), uint64(cudaWidth), uint64(2048)
		if strings.HasSuffix(t.Name, "ffn_down_exps.weight") {
			want, width, rows = typeQ2K, 2048, cudaWidth
		}
		if t.TypeID != want || len(t.Dimensions) != 3 || t.Dimensions[0] != width || t.Dimensions[1] != rows || t.Dimensions[2] != 256 {
			return errors.New("CUDA streaming expert format unsupported")
		}
	}
	strides := [3]int{}
	var bytes uint64
	for i, n := range []uint64{base.Gate.Stride, base.Up.Stride, base.Down.Stride} {
		if n > math.MaxInt || n > math.MaxUint64-bytes {
			return errors.New("CUDA streaming stride overflow")
		}
		strides[i], bytes = int(n), bytes+n
	}
	budget, err := dsv4expertcache.ParseBudget(o.budget)
	if err != nil {
		return err
	}
	slots := uint64(budget.Slots)
	if budget.Bytes != 0 {
		slots = budget.Bytes / bytes
	}
	slots = min(slots, uint64(cudaLayers*base.Gate.Experts))
	if slots < 6 || slots > math.MaxInt || slots > math.MaxUint64/bytes {
		return errors.New("CUDA streaming cache requires six bounded slots")
	}
	model, quant, err := r.stateIdentity.get(r.file, int64(size), r.gguf)
	if err != nil {
		return err
	}
	h := sha256.New()
	var word [8]byte
	for layer := range cudaLayers {
		tr := triplets[layer]
		for _, t := range []dsv4expertcache.Tensor{tr.Gate, tr.Up, tr.Down} {
			for _, n := range []uint64{t.Offset, t.Bytes, t.Stride, uint64(t.Experts)} {
				binary.LittleEndian.PutUint64(word[:], n)
				h.Write(word[:])
			}
		}
	}
	id := dsv4expertcache.Identity{Model: model, Quant: quant}
	copy(id.Layout[:], h.Sum(nil))
	source, err := dsv4expertcache.NewSource(r.file, size, id, triplets)
	if err != nil {
		return err
	}
	if err := r.loadResidentSet(); err != nil {
		return err
	}
	// Keep partial uploader ownership even before a selections owner exists.
	upload, err := dsv4expertcache.NewCUDAUpload(r.device, int(slots), strides, slots*bytes)
	r.streamingExperts = &cudaStreamingExperts{}
	r.streamingExperts.coordinator.closeCache = func() error {
		if upload != nil {
			return upload.Close()
		}
		return nil
	}
	if err != nil {
		return err
	}
	selections, err := dsv4expertcache.NewCUDASelections(source, upload)
	if err != nil {
		return err
	}
	r.streamingExperts, err = newCUDAStreamingExperts(r.device, selections)
	return err
}
