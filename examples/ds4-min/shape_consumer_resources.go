package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
)

type shapeConsumerBuffer struct {
	name  string
	bytes int
}

// The HC mix owner holds two 24-element float buffers.
func shapeStateReserve(reserve, consumers uint64) (uint64, error) {
	const mix = 2 * 24 * 4
	if consumers > ^uint64(0)-mix || reserve > ^uint64(0)-(consumers+mix) {
		return 0, errors.New("cuda shape state: reserve overflow")
	}
	return reserve + consumers + mix, nil
}

// These are the byte-exact scratch layouts consumed by the implemented decode
// projections/router/head. Cache/HC state is owned separately. In particular,
// this plan is not a replacement for compressor/indexer temporary storage.
func planShapeConsumerBuffers(c dsv4shape.Config, context uint64) ([]shapeConsumerBuffer, uint64, error) {
	if err := c.Validate(); err != nil {
		return nil, 0, err
	}
	_, scores, err := shapeAttentionScoreLayout(c, context)
	if err != nil {
		return nil, 0, err
	}
	var buffers []shapeConsumerBuffer
	var total uint64
	add := func(name string, n uint64) error {
		if n == 0 || n > uint64(int(^uint(0)>>1)) || total > ^uint64(0)-n {
			return errors.New("cuda shape consumers: scratch size overflow")
		}
		buffers = append(buffers, shapeConsumerBuffer{name, int(n)})
		total += n
		return nil
	}
	for _, v := range []struct {
		name string
		n    uint64
	}{
		{"attention-input", uint64(c.Embed) / 32 * 36}, {"attention-qr", uint64(c.QLoRA) / 32 * 36}, {"attention-heads", uint64(c.Heads*c.HeadDim) / 32 * 36}, {"attention-low", uint64(c.OutGroups*c.OutLoRA) / 32 * 36},
		{"router-logits", uint64(c.Experts) * 4}, {"router-probs", uint64(c.Experts) * 4}, {"router-weights", uint64(c.ExpertsUsed) * 4}, {"router-ids", uint64(c.ExpertsUsed) * 4},
		{"expert-gate", uint64(c.ExpertsUsed*c.ExpertFF) * 4}, {"expert-up", uint64(c.ExpertsUsed*c.ExpertFF) * 4}, {"expert-down", uint64(c.ExpertsUsed*c.Embed) * 4},
		{"ffn-input", uint64(c.Embed) / 32 * 36}, {"ffn-experts", uint64(c.ExpertsUsed*c.ExpertFF) / 32 * 36}, {"shared-mid", uint64(c.ExpertFF) / 32 * 36}, {"router-invalid", 4},
		{"output-input", uint64(c.Embed) / 32 * 36}, {"logits", uint64(c.Vocab) * 4},
		{"indexer-query", uint64(c.IndexerHeads*c.IndexerHeadDim) * 4}, {"indexer-invalid", 4},
		{"attention-scores", scores * 4},
	} {
		if err := add(v.name, v.n); err != nil {
			return nil, 0, fmt.Errorf("allocate %s: %w", v.name, err)
		}
	}
	return buffers, total, nil
}
