//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"fmt"
	"math"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// EncodeProposalHead binds actual target output weights to native support
// hidden states, then applies admitted support Markov/confidence tensors.
// It does not run the target transformer and never supplies its next token
// as a support proposal. Result publication requires terminal completion
// and the separate status/count validation below.
func (d *metalDSparkDense) EncodeProposalHead(b *metalBatch, a *metalDSparkAttention, outputWeight boundMetalTensor, hidden, logits metalmem.Slice[float32], ids metalmem.Slice[int32], probabilities, confidenceLogits metalmem.Slice[float32], result metalmem.Slice[uint32], count, previous int, threshold float32) error {
	if d == nil || d.closed || d.weights == nil || d.weights.model == nil || a == nil || a.closed || a.weights != d.weights {
		return fmt.Errorf("metal dspark head: invalid owner")
	}
	if err := a.ready(b); err != nil {
		return err
	}
	if count < 1 || count > d.weights.model.block || previous < 0 || previous >= Flash.Vocab || threshold < 0 || threshold > 1 || math.IsNaN(float64(threshold)) || hidden.Len() != count*4096 || logits.Len() != count*Flash.Vocab || ids.Len() != count || probabilities.Len() != count || confidenceLogits.Len() != count || result.Len() != 2 || len(outputWeight.dimensions) != 2 || outputWeight.dimensions[0] != 4096 || outputWeight.dimensions[1] != uint64(Flash.Vocab) {
		return fmt.Errorf("metal dspark head: invalid output or proposal extent")
	}
	stage := d.weights.model.stages - 1
	w1, err := d.weights.tensor(fmt.Sprintf("mtp.%d.markov_head.markov_w1.weight", stage))
	if err != nil {
		return err
	}
	w2, err := d.weights.tensor(fmt.Sprintf("mtp.%d.markov_head.markov_w2.weight", stage))
	if err != nil {
		return err
	}
	confidence, err := d.weights.tensor(fmt.Sprintf("mtp.%d.confidence_head.proj.weight", stage))
	if err != nil {
		return err
	}
	rank := d.weights.model.rank
	for _, t := range []boundMetalTensor{w1, w2, confidence} {
		if t.typeID != 0 && t.typeID != 1 && t.typeID != 8 {
			return fmt.Errorf("metal dspark head: native matrix type unavailable")
		}
	}
	if rank < 1 || rank > 4096 || len(w1.dimensions) != 2 || len(w2.dimensions) != 2 || w1.dimensions[0] != uint64(rank) || w2.dimensions[0] != uint64(rank) || w1.dimensions[1] != uint64(Flash.Vocab) || w2.dimensions[1] != uint64(Flash.Vocab) || len(confidence.dimensions) != 2 || confidence.dimensions[0] != uint64(4096+rank) || confidence.dimensions[1] != 1 || ((w1.typeID == 8 || w2.typeID == 8) && rank%32 != 0) || (confidence.typeID == 8 && (4096+rank)%32 != 0) {
		return fmt.Errorf("metal dspark head: invalid Markov or confidence shape")
	}
	if err := d.encodeTensor(b, outputWeight, hidden, logits, count); err != nil {
		return err
	}
	enc := b.encoder()
	enc.SetComputePipelineState(a.head)
	enc.SetBufferWithOffsetAtIndex(logits.Buffer(), logits.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(hidden.Buffer(), hidden.Offset(), 1)
	for i, t := range []boundMetalTensor{w1, w2, confidence} {
		enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), uint(i+2))
	}
	var args [28]byte
	for i, v := range []uint32{uint32(count), uint32(Flash.Vocab), uint32(rank), uint32(previous), w1.typeID, w2.typeID, confidence.typeID} {
		binary.LittleEndian.PutUint32(args[i*4:], v)
	}
	enc.SetBytesLengthAtIndex(args[:], 5)
	libmetal.SetScalar(enc, threshold, 6)
	enc.SetBufferWithOffsetAtIndex(ids.Buffer(), ids.Offset(), 7)
	enc.SetBufferWithOffsetAtIndex(probabilities.Buffer(), probabilities.Offset(), 8)
	enc.SetBufferWithOffsetAtIndex(result.Buffer(), result.Offset(), 9)
	enc.SetBufferWithOffsetAtIndex(confidenceLogits.Buffer(), confidenceLogits.Offset(), 10)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	b.encoded++
	a.batches = retainMetalDSparkBatch(a.batches, b)
	return nil
}

func metalDSparkProposalResult(ids []int32, probabilities []float32, result []uint32, count int, threshold float32) ([]int, error) {
	if count < 1 || count > 16 || len(ids) != count || len(probabilities) != count || len(result) != 2 || result[1] != 0 || result[0] > uint32(count) || threshold < 0 || threshold > 1 || math.IsNaN(float64(threshold)) {
		return nil, fmt.Errorf("metal dspark head: invalid terminal result")
	}
	n := int(result[0])
	proposal := make([]int, n)
	for i := 0; i < n; i++ {
		if ids[i] < 0 || int(ids[i]) >= Flash.Vocab || probabilities[i] < threshold || probabilities[i] > 1 || math.IsNaN(float64(probabilities[i])) || math.IsInf(float64(probabilities[i]), 0) {
			return nil, fmt.Errorf("metal dspark head: invalid accepted row %d", i)
		}
		proposal[i] = int(ids[i])
	}
	if n < count && (probabilities[n] < 0 || probabilities[n] >= threshold || math.IsNaN(float64(probabilities[n])) || math.IsInf(float64(probabilities[n]), 0)) {
		return nil, fmt.Errorf("metal dspark head: invalid declined confidence row %d", n)
	}
	return proposal, nil
}
