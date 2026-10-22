//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func (w *cudaDSparkWeights) encodeNoncausal(r *cudaRuntime, stage int, queries, raw, output cudamem.Slice[float32], capacity, start, count, rows int) error {
	if err := dsparkRequestOpen(r); err != nil {
		return err
	}
	if w == nil || r == nil || r.cudaFoundation == nil || r.device == nil {
		return fmt.Errorf("cuda dspark attention: missing owner or request")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed || w.model == nil || w.device != r.device || stage < 0 || stage >= w.model.stages || capacity < 1 || start < 0 || start >= capacity || count < 1 || count > capacity || rows < 1 || rows > w.model.block || raw.Len() != capacity*Flash.HeadDim || queries.Len() != rows*Flash.Heads*Flash.HeadDim || output.Len() != queries.Len() || raw.Ptr() == 0 || queries.Ptr() == 0 || output.Ptr() == 0 {
		return fmt.Errorf("cuda dspark attention: invalid owner or ring/query shape")
	}
	sink, ok := w.tensors[fmt.Sprintf("mtp.%d.attn_sinks.weight", stage)]
	if !ok || sink.typeID != 0 || len(sink.dimensions) != 1 || sink.dimensions[0] != uint64(Flash.Heads) {
		return fmt.Errorf("cuda dspark attention: invalid sink")
	}
	if err := w.loadMathLocked(); err != nil {
		return err
	}
	return r.withContext(func() error {
		fn, err := w.module.Function("ds4_dspark_attention_noncausal")
		if err != nil {
			return err
		}
		return cudalaunch.Kernel(fn).Grid(uint32(Flash.Heads), uint32(rows), 1).Block(256, 1, 1).On(r.stream()).Launch(queries, raw, sink.device, output, uint32(capacity), uint32(start), uint32(count))
	})
}

// encodeRope rotates consecutive draft rows, or a single feature row. The
// caller must dispatch feature KV separately: its position equals draft0,
// not draft0-1. Dense base10000/no YaRN is fixed in the kernel.
func (w *cudaDSparkWeights) encodeRope(r *cudaRuntime, values cudamem.Slice[float32], heads, rows, position int, inverse bool) error {
	if err := dsparkRequestOpen(r); err != nil {
		return err
	}
	if w == nil || r == nil || r.cudaFoundation == nil || r.device == nil {
		return fmt.Errorf("cuda dspark RoPE: missing owner or request")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed || w.model == nil || w.device != r.device || (heads != 1 && heads != Flash.Heads) || rows < 1 || rows > w.model.block || position < 0 || position > 2147483647-rows || values.Len() != rows*heads*Flash.HeadDim || values.Ptr() == 0 {
		return fmt.Errorf("cuda dspark RoPE: invalid owner or shape")
	}
	if err := w.loadMathLocked(); err != nil {
		return err
	}
	arg := uint32(0)
	if inverse {
		arg = 1
	}
	return r.withContext(func() error {
		fn, err := w.module.Function("ds4_dspark_rope_tail")
		if err != nil {
			return err
		}
		return cudalaunch.Kernel(fn).Grid(uint32(rows*heads), 1, 1).Block(32, 1, 1).On(r.stream()).Launch(values, uint32(heads), uint32(position), arg)
	})
}

// encodeStoreRaw writes the request-private support ring. kv includes the
// feature row and all draft rows, after their separately positioned RoPE.
// It mutates kv with the FP8 round trip before storing half-rounded values.
func (w *cudaDSparkWeights) encodeStoreRaw(r *cudaRuntime, kv, raw cudamem.Slice[float32], capacity, start, rows int) error {
	if err := dsparkRequestOpen(r); err != nil {
		return err
	}
	if w == nil || r == nil || r.cudaFoundation == nil || r.device == nil {
		return fmt.Errorf("cuda dspark store: missing owner or request")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed || w.model == nil || w.device != r.device || capacity < 1 || start < 0 || start >= capacity || rows < 1 || rows > w.model.block+1 || rows > capacity || kv.Len() != rows*Flash.HeadDim || raw.Len() != capacity*Flash.HeadDim || kv.Ptr() == 0 || raw.Ptr() == 0 {
		return fmt.Errorf("cuda dspark store: invalid owner or ring shape")
	}
	if err := w.loadMathLocked(); err != nil {
		return err
	}
	return r.withContext(func() error {
		fn, err := w.module.Function("ds4_dspark_store_raw")
		if err != nil {
			return err
		}
		return cudalaunch.Kernel(fn).Grid(uint32(rows), 1, 1).Block(64, 1, 1).On(r.stream()).Launch(kv, raw, uint32(capacity), uint32(start))
	})
}
