//go:build linux

package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type cudaDSparkHeadWork struct {
	mixed, hidden, logits, markov, bias, features, confidence cudamem.Slice[float32]
	proposal                                                  cudamem.Slice[int32]
	length, status                                            cudamem.Slice[uint32]
}

// encodeHead uses support HC-head weights with the real base output tensor,
// then sequential previous-token Markov bias and confidence. Host code only
// schedules native kernels; no support/target CPU forward manufactures drafts.
// Caller owns every workspace and checks status after request-stream sync.
func (w *cudaDSparkWeights) encodeHead(r *cudaRuntime, hc cudamem.Slice[float32], previous, count int, threshold float32, a *cudaDSparkHeadWork) error {
	if err := dsparkRequestOpen(r); err != nil {
		return err
	}
	if w == nil || w.closed || w.model == nil || r == nil || r.cudaFoundation == nil || r.model == nil || w.device != r.device || a == nil || previous < 0 || previous >= Flash.Vocab || count < 1 || count > w.model.block || threshold < 0 || threshold > 1 || math.IsNaN(float64(threshold)) {
		return fmt.Errorf("cuda dspark head: invalid owner or policy")
	}
	if hc.Len() != w.model.block*Flash.HC*Flash.Embed || a.mixed.Len() != 2*Flash.Embed || a.hidden.Len() != count*Flash.Embed || a.logits.Len() != count*Flash.Vocab || a.markov.Len() != w.model.rank || a.bias.Len() != Flash.Vocab || a.features.Len() != Flash.Embed+w.model.rank || a.confidence.Len() != count || a.proposal.Len() != count || a.length.Len() != 1 || a.status.Len() != 1 {
		return fmt.Errorf("cuda dspark head: invalid workspace")
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", w.model.stages-1, suffix) }
	var values [6]cudaModelTensor
	for i, suffix := range []string{"hc_head_fn.weight", "hc_head_scale.weight", "hc_head_base.weight", "norm.weight", "markov_head.markov_w1.weight", "confidence_head.proj.weight"} {
		var err error
		values[i], err = w.tensor(name(suffix))
		if err != nil {
			return err
		}
	}
	w1 := values[4]
	if len(w1.dimensions) != 2 || w1.dimensions[0] != uint64(w.model.rank) || w1.dimensions[1] != uint64(Flash.Vocab) || (w1.typeID != 0 && w1.typeID != 1 && w1.typeID != 8) || w1.typeID == 8 && w.model.rank%32 != 0 {
		return fmt.Errorf("cuda dspark head: invalid Markov row layout")
	}
	if err := a.status.CopyFromHost([]uint32{0}); err != nil {
		return err
	}
	out := cudaOutput{hcFn: values[0], hcScale: values[1], hcBase: values[2], norm: values[3], weight: r.model.output.weight}
	for row := 0; row < count; row++ {
		hidden := a.hidden.Slice(row*Flash.Embed, (row+1)*Flash.Embed)
		logits := a.logits.Slice(row*Flash.Vocab, (row+1)*Flash.Vocab)
		if err := r.cudaFoundation.encodeOutputTransform(hc.Slice(row*Flash.HC*Flash.Embed, (row+1)*Flash.HC*Flash.Embed), out, a.mixed, logits, false); err != nil {
			return err
		}
		if err := hidden.CopyDeviceToDeviceAsync(a.mixed.Slice(Flash.Embed, 2*Flash.Embed), r.stream()); err != nil {
			return err
		}
		if err := r.cudaFoundation.encodeQ80Matvec(out.weight, hidden, logits); err != nil {
			return err
		}
		if err := w.launchHead(r, "ds4_dspark_markov_row", uint32((w.model.rank+255)/256), w1.device, a.markov, a.proposal, uint32(previous), uint32(row), uint32(w.model.rank), w1.typeID); err != nil {
			return err
		}
		if err := w.encodeDense(r, name("markov_head.markov_w2.weight"), a.markov, a.bias); err != nil {
			return err
		}
		if err := w.launchHead(r, "ds4_dspark_markov_argmax", 1, logits, a.bias, a.proposal, a.status, uint32(row), uint32(Flash.Vocab)); err != nil {
			return err
		}
		if err := w.launchHead(r, "ds4_dspark_confidence_input", uint32((Flash.Embed+w.model.rank+255)/256), hidden, a.markov, a.features, uint32(w.model.rank)); err != nil {
			return err
		}
		if err := w.encodeDense(r, name("confidence_head.proj.weight"), a.features, a.confidence.Slice(row, row+1)); err != nil {
			return err
		}
	}
	return w.launchHead(r, "ds4_dspark_confidence_prefix", 1, a.confidence, a.length, a.status, uint32(count), threshold)
}

func (w *cudaDSparkWeights) launchHead(r *cudaRuntime, name string, grid uint32, args ...any) error {
	if err := dsparkRequestOpen(r); err != nil {
		return err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return fmt.Errorf("cuda dspark head: support closed")
	}
	if err := w.loadMathLocked(); err != nil {
		return err
	}
	return r.withContext(func() error {
		fn, err := w.module.Function(name)
		if err != nil {
			return err
		}
		return cudalaunch.Kernel(fn).Grid(grid, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(args...)
	})
}
