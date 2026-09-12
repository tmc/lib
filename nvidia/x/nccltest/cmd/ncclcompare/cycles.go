package main

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/x/nccltest"
)

// readCycleResults requires the terminal summary as well as every result.
// The caller must separately check the child exit status and effective cases.
func readCycleResults(input io.Reader, count int) ([]nccltest.Result, error) {
	if count <= 0 {
		return nil, fmt.Errorf("invalid cycle count")
	}
	decoder := json.NewDecoder(input)
	decoder.DisallowUnknownFields()
	var results []nccltest.Result
	for i := 0; i < count; i++ {
		var row struct {
			Kind   string
			Cycle  int
			Result *nccltest.Result
		}
		if err := decoder.Decode(&row); err != nil {
			return nil, fmt.Errorf("cycle %d: %w", i+1, err)
		}
		if row.Kind != "cycle" || row.Cycle != i+1 || row.Result == nil {
			return nil, fmt.Errorf("missing or out-of-order cycle %d", i+1)
		}
		results = append(results, *row.Result)
	}
	var summary struct {
		Kind                 string
		Requested, Completed int
		Complete             bool
		Error                string
	}
	if err := decoder.Decode(&summary); err != nil {
		return nil, fmt.Errorf("cycle summary: %w", err)
	}
	if summary.Kind != "summary" || summary.Requested != count || summary.Completed != count || !summary.Complete || summary.Error != "" {
		return nil, fmt.Errorf("incomplete cycle summary: %s", summary.Error)
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("unexpected data after cycle summary")
	}
	return results, nil
}

func readSweepResults(input io.Reader, sizes []int, cycles int) ([]nccltest.Result, error) {
	if len(sizes) == 0 || cycles <= 0 || cycles > int(^uint(0)>>1)/len(sizes) {
		return nil, fmt.Errorf("invalid sweep dimensions")
	}
	decoder := json.NewDecoder(input)
	decoder.DisallowUnknownFields()
	total := cycles * len(sizes)
	var results []nccltest.Result
	for i := 0; i < total; i++ {
		var row struct {
			Kind             string
			Cycle, SizeIndex int
			Result           *nccltest.Result
		}
		if err := decoder.Decode(&row); err != nil {
			return nil, fmt.Errorf("sample %d: %w", i+1, err)
		}
		if row.Kind != "sample" || row.Cycle != i/len(sizes)+1 || row.SizeIndex != i%len(sizes)+1 || row.Result == nil || row.Result.Case.Bytes != sizes[i%len(sizes)] {
			return nil, fmt.Errorf("missing or out-of-order sample %d", i+1)
		}
		results = append(results, *row.Result)
	}
	var summary struct {
		Kind                                                string
		RequestedCycles, RequestedSamples, CompletedSamples int
		Complete                                            bool
		Error                                               string
	}
	if err := decoder.Decode(&summary); err != nil {
		return nil, fmt.Errorf("sweep summary: %w", err)
	}
	if summary.Kind != "summary" || summary.RequestedCycles != cycles || summary.RequestedSamples != total || summary.CompletedSamples != total || !summary.Complete || summary.Error != "" {
		return nil, fmt.Errorf("incomplete sweep summary: %s", summary.Error)
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("unexpected data after sweep summary")
	}
	return results, nil
}
