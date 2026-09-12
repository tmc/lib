package main

import (
	"fmt"
	"strconv"

	"github.com/tmc/lib/nvidia/x/nccltest"
)

func referenceSweepArgs(sizes []int) ([]string, error) {
	if len(sizes) == 0 || sizes[0] <= 0 {
		return nil, fmt.Errorf("empty or invalid reference sweep")
	}
	args := []string{"-b", strconv.Itoa(sizes[0]), "-e", strconv.Itoa(sizes[len(sizes)-1])}
	if len(sizes) == 1 {
		return args, nil
	}
	step := sizes[1] - sizes[0]
	if step <= 0 {
		return nil, fmt.Errorf("reference sweep must increase")
	}
	linear := true
	for i := 1; i < len(sizes); i++ {
		if sizes[i] <= sizes[i-1] {
			return nil, fmt.Errorf("reference sweep must increase")
		}
		if sizes[i]-sizes[i-1] != step {
			linear = false
		}
	}
	if linear {
		return append(args, "-i", strconv.Itoa(step), "-f", "1"), nil
	}
	factor := sizes[1] / sizes[0]
	if factor <= 1 {
		return nil, fmt.Errorf("reference sweep must be linear or geometric")
	}
	for i := 1; i < len(sizes); i++ {
		if sizes[i]%sizes[i-1] != 0 || sizes[i]/sizes[i-1] != factor {
			return nil, fmt.Errorf("reference sweep must be linear or geometric")
		}
	}
	if uint(sizes[len(sizes)-1]) > ^uint(0)/uint(factor) {
		return nil, fmt.Errorf("reference sweep increment overflows")
	}
	return append(args, "-f", strconv.Itoa(factor)), nil
}

func verifyReferenceRow(row nccltest.ReferenceRow, c nccltest.Case, ranks int, config nccltest.ReferenceConfig) error {
	if row.Config != config {
		return fmt.Errorf("native controls mismatch")
	}
	root := -1
	switch c.Operation {
	case nccltest.Reduce, nccltest.Broadcast, nccltest.Scatter, nccltest.Gather:
		root = c.Root
	}
	if row.Root != root {
		return fmt.Errorf("root mismatch")
	}
	switch c.Operation {
	case nccltest.AllReduce, nccltest.Reduce, nccltest.ReduceScatter:
		if row.Reduction != string(c.Reduction) {
			return fmt.Errorf("reduction mismatch")
		}
	}
	l, err := c.Layout(ranks)
	if err != nil {
		return err
	}
	if row.Count != l.ParamCount || row.Bytes != max(l.SendCount, l.RecvCount)*c.DataType.Size() || row.DataType != string(c.DataType) {
		return fmt.Errorf("effective reference layout mismatch")
	}
	return nil
}
