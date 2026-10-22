package main

import (
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
)

type shapeGroupingPlan struct {
	experts, pairs  int
	offsets, active int
}

func planShapeGrouping(c dsv4shape.Config, tokens int) (shapeGroupingPlan, error) {
	if err := c.Validate(); err != nil {
		return shapeGroupingPlan{}, err
	}
	// Bound the correctness-first serial kernel. Larger work requires the
	// parallel grouping implementation and a distinct device admission gate.
	if tokens <= 0 || tokens > 16384/c.ExpertsUsed {
		return shapeGroupingPlan{}, fmt.Errorf("cuda shape grouping: token count %d outside bounded kernel", tokens)
	}
	return shapeGroupingPlan{experts: c.Experts, pairs: tokens * c.ExpertsUsed, offsets: c.Experts + 1, active: c.Experts}, nil
}
