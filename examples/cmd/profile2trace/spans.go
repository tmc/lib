package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
)

func readHost(r io.Reader) ([]slice, error) {
	dec := json.NewDecoder(r)
	var rows []slice
	var runID string
	for {
		var v struct {
			Schema   string `json:"schema"`
			Kind     string `json:"kind"`
			Run      string `json:"run_id"`
			ID       string `json:"id"`
			Name     string `json:"name"`
			Clock    string `json:"clock_domain"`
			Start    *int64 `json:"timestamp_ns"`
			Duration *int64 `json:"duration_ns"`
		}
		err := dec.Decode(&v)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("host event %d: %w", len(rows), err)
		}
		if v.Schema != "gputrace.host-event/v1" || v.Kind != "interval" || v.Clock != "cpu_uptime_ns" || v.Run == "" || v.Name == "" || v.Start == nil || v.Duration == nil || *v.Start < 0 || *v.Duration < 0 {
			return nil, fmt.Errorf("host event %d: invalid schema, clock, identity, or interval", len(rows))
		}
		if runID != "" && runID != v.Run {
			return nil, errors.New("host sidecar contains multiple run IDs")
		}
		runID = v.Run
		rows = append(rows, slice{v.Name, "host", "host " + v.Run, "host spans (thread not recorded)", *v.Start, *v.Duration, map[string]any{"id": v.ID, "clock_domain": v.Clock}})
	}
	if len(rows) == 0 {
		return nil, errors.New("host sidecar contains no intervals")
	}
	return rows, nil
}

func readMetal(r io.Reader) ([]slice, error) {
	var receipt struct {
		Regions []struct {
			Name     string   `json:"region"`
			Layer    int      `json:"layer"`
			Position int      `json:"position"`
			Start    *float64 `json:"start_s"`
			End      *float64 `json:"end_s"`
		} `json:"regions"`
	}
	dec := json.NewDecoder(r)
	if err := dec.Decode(&receipt); err != nil {
		return nil, fmt.Errorf("Metal receipt: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		return nil, errors.New("Metal receipt: trailing data")
	}
	if len(receipt.Regions) == 0 {
		return nil, errors.New("Metal receipt contains no regions")
	}
	var rows []slice
	for i, v := range receipt.Regions {
		if v.Name == "" || v.Start == nil || v.End == nil || *v.Start <= 0 || *v.End < *v.Start || *v.End >= float64(math.MaxInt64)/1e9 {
			return nil, fmt.Errorf("Metal region %d: invalid name or GPU interval", i)
		}
		rows = append(rows, slice{v.Name, "metal", "Metal device", "command buffers", int64(math.Round(*v.Start * 1e9)), int64(math.Round((*v.End - *v.Start) * 1e9)), map[string]any{"layer": v.Layer, "position": v.Position, "clock_domain": "metal_gpu_seconds", "granularity": "command buffer"}})
	}
	return rows, nil
}
