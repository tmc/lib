package main

import (
	"errors"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
)

// Each head owns a disjoint score row. Ratio four is the shortest compressor
// period in either family; visible rows never exceed ceil(context/4).
func shapeAttentionScoreLayout(c dsv4shape.Config, context uint64) (stride uint32, elements uint64, err error) {
	if err := c.Validate(); err != nil {
		return 0, 0, err
	}
	if context == 0 || context > math.MaxUint32 {
		return 0, 0, errors.New("cuda shape attention: invalid score context")
	}
	rows := min(context, uint64(c.SlidingWindow)) + (context+3)/4
	if rows > math.MaxUint32 || rows > math.MaxUint64/uint64(c.Heads) {
		return 0, 0, errors.New("cuda shape attention: score count overflow")
	}
	n := rows * uint64(c.Heads)
	if n > uint64(int(^uint(0)>>1))/4 {
		return 0, 0, errors.New("cuda shape attention: score bytes overflow")
	}
	return uint32(rows), n, nil
}
