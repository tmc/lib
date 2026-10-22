//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// encodeInput binds published selected-layer means to the actual support
// projection, seed embedding, and noise embeddings. It does not reread host
// model bytes or substitute terminal HC for selected features.
func (w *cudaDSparkWeights) encodeInput(f *cudaDSparkFeatures, seed int, output cudamem.Slice[float32]) error {
	if w == nil || w.closed || w.model == nil || !f.bound() || !f.published || f.modelSHA != w.model.sha256 || f.r.device != w.device || f.request.generation != f.generation+1 || f.r.pos != f.position+1 || seed != f.token || seed < 0 || seed >= Flash.Vocab || output.Len() != (w.model.block+1)*Flash.HC*Flash.Embed {
		return fmt.Errorf("cuda dspark input: unpublished or foreign feature frontier")
	}
	r := f.r
	if err := dsparkRequestOpen(r); err != nil {
		return err
	}
	main := output.Slice(0, Flash.Embed)
	if err := w.encodeDense(r, "mtp.0.main_proj.weight", f.features, main); err != nil {
		return err
	}
	norm, err := w.tensor("mtp.0.main_norm.weight")
	if err != nil {
		return err
	}
	if err := r.cudaFoundation.encodeRMS(main, norm, main, 1, true); err != nil {
		return err
	}
	for row := 1; row <= w.model.block; row++ {
		token := w.model.noise
		if row == 1 {
			token = seed
		}
		start := row * Flash.HC * Flash.Embed
		if err := r.cudaFoundation.encodeEmbeddingRow(r.model.output.embedding, uint32(token), output.Slice(start, start+Flash.Embed)); err != nil {
			return err
		}
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return fmt.Errorf("cuda dspark input: support closed")
	}
	if err := w.loadMathLocked(); err != nil {
		return err
	}
	return r.withContext(func() error {
		fn, err := w.module.Function("ds4_dspark_repeat_hc")
		if err != nil {
			return err
		}
		return cudalaunch.Kernel(fn).Grid(16, uint32(w.model.block+1), 1).Block(256, 1, 1).On(r.stream()).Launch(output)
	})
}
