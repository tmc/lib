package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

//go:embed cudaprofstacks.py
var pythonStackAdapter []byte

func prepareStackTransport(dir string) error {
	if err := os.Mkdir(filepath.Join(dir, "stacks"), 0700); err != nil {
		return fmt.Errorf("create stack transport: %w", err)
	}
	python := filepath.Join(dir, "python")
	if err := os.Mkdir(python, 0700); err != nil {
		return fmt.Errorf("create Python adapter directory: %w", err)
	}
	if err := os.WriteFile(filepath.Join(python, "cudaprofstacks.py"), pythonStackAdapter, 0600); err != nil {
		return fmt.Errorf("write Python stack adapter: %w", err)
	}
	return nil
}

func attachExternalStackSidecar(c *cudacapture.Capture, path string) ([]byte, error) {
	if c == nil || path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read external stack sidecar: %w", err)
	}
	tracker := cudacapture.NewCorrelationTracker()
	if err := tracker.ReadExternalStackTable(bytes.NewReader(data)); err != nil {
		return nil, err
	}
	if c.Files == nil {
		c.Files = make(map[string][]byte)
	}
	if _, exists := c.Files[cudacapture.ExternalStacksName]; exists {
		return nil, fmt.Errorf("capture already has %q", cudacapture.ExternalStacksName)
	}
	c.Files[cudacapture.ExternalStacksName] = data
	return data, nil
}

func labelExternalStacks(p *profile.Profile, activities []cudacapture.ActivityRecord, sampleRecords []int, data []byte) (*profile.Profile, error) {
	if len(data) == 0 {
		return p, nil
	}
	tracker := cudacapture.NewCorrelationTracker()
	if err := tracker.ReadExternalStackTable(bytes.NewReader(data)); err != nil {
		return nil, err
	}
	for _, activity := range activities {
		if activity.Subtype == "external_correlation" {
			if err := tracker.RecordActivity(activity); err != nil {
				return nil, err
			}
		}
	}
	for _, index := range sampleRecords {
		if index < 0 || index >= len(activities) {
			return nil, fmt.Errorf("sample record index %d is out of range", index)
		}
		a := activities[index]
		if a.Subtype != "kernel" && a.Subtype != "memcpy" && a.Subtype != "memset" {
			continue
		}
		corr := uint64(a.Correlation)
		if corr == 0 {
			corr = uint64(a.RuntimeCorrelation)
		}
		if corr == 0 {
			continue
		}
		if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: corr, KernelName: a.Name}); err != nil {
			return nil, err
		}
		if err := tracker.RecordActivity(a); err != nil {
			return nil, err
		}
	}
	matched, ambiguous, _, _ := tracker.Join()
	byCorrelation := make(map[uint64]cudacapture.CorrelationJoin, len(matched))
	for _, join := range matched {
		byCorrelation[join.CorrelationID] = join
	}
	ambiguousIDs := make(map[uint64]bool, len(ambiguous))
	for _, join := range ambiguous {
		ambiguousIDs[join.CorrelationID] = true
	}
	out := p.Copy()
	for i, index := range sampleRecords {
		a := activities[index]
		corr := uint64(a.Correlation)
		if corr == 0 {
			corr = uint64(a.RuntimeCorrelation)
		}
		if ambiguousIDs[corr] {
			setStackLabel(out.Sample[i], "cupti.external.stack_status", "ambiguous")
			continue
		}
		join, ok := byCorrelation[corr]
		if !ok || !join.HasExternalID {
			continue
		}
		if join.CallstackMissing || len(join.Callstack) == 0 {
			setStackLabel(out.Sample[i], "cupti.external.stack_status", "missing")
			continue
		}
		setStackLabel(out.Sample[i], "cupti.external.stack", strings.Join(join.Callstack, ";"))
		setStackLabel(out.Sample[i], "cupti.external.stack_status", "matched")
	}
	return out, nil
}

func setStackLabel(sample *profile.Sample, key, value string) {
	if sample.Label == nil {
		sample.Label = make(map[string][]string)
	}
	sample.Label[key] = []string{value}
}

func externalStackKinds(data []byte) map[uint32]bool {
	kinds := make(map[uint32]bool)
	var table cudaprof.ExternalStackTable
	if json.Unmarshal(data, &table) == nil {
		for _, record := range table.Stacks {
			kinds[record.ExternalKind] = true
		}
		return kinds
	}
	for _, line := range bytes.Split(data, []byte{'\n'}) {
		var record struct {
			Kind         string `json:"kind"`
			ExternalKind uint32 `json:"external_kind"`
		}
		if json.Unmarshal(line, &record) == nil && record.Kind == "stack" {
			kinds[record.ExternalKind] = true
		}
	}
	return kinds
}
