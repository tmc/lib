package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
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

type stackSidecarLine struct {
	Version      int      `json:"version"`
	Kind         string   `json:"kind"`
	Losses       string   `json:"losses,omitempty"`
	LowerBound   bool     `json:"losses_lower_bound"`
	Process      uint64   `json:"process,omitempty"`
	ExternalKind uint32   `json:"external_kind,omitempty"`
	ExternalID   uint64   `json:"external_id,omitempty"`
	Frames       []string `json:"frames,omitempty"`
}

type stackSidecarRead struct {
	data       []byte
	rawSize    int64
	damaged    bool
	damageNote string
	exists     bool
}

// readExternalStackSidecar collects process-isolated sidecars into one bounded
// capture member. Complete records survive a torn final line; malformed
// complete records reject the sidecar so they can never be labeled as valid.
func readExternalStackSidecar(base string, targetPID uint64) (stackSidecarRead, error) {
	var out stackSidecarRead
	if base == "" {
		return out, nil
	}
	if info, err := os.Stat(base); err == nil && info.IsDir() {
		return stackSidecarRead{exists: true}, fmt.Errorf("sidecar path is a directory")
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return stackSidecarRead{exists: true}, fmt.Errorf("stat sidecar: %w", err)
	}
	paths := []string{}
	if _, err := os.Stat(base); err == nil {
		paths = append(paths, base)
	} else if !errors.Is(err, os.ErrNotExist) {
		return stackSidecarRead{exists: true}, fmt.Errorf("stat sidecar: %w", err)
	}
	entries, err := os.ReadDir(filepath.Dir(base))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return stackSidecarRead{exists: true}, fmt.Errorf("list sidecars: %w", err)
	}
	prefix := filepath.Base(base) + "."
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), prefix) {
			continue
		}
		pidText := strings.TrimPrefix(entry.Name(), prefix)
		pid, parseErr := strconv.ParseUint(pidText, 10, 64)
		if parseErr != nil || pid == 0 || entry.IsDir() {
			continue
		}
		paths = append(paths, filepath.Join(filepath.Dir(base), entry.Name()))
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		return out, nil
	}
	out.exists = true
	table := cudaprof.ExternalStackTable{Version: 1}
	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			return out, fmt.Errorf("stat %s: %w", filepath.Base(path), err)
		}
		if !info.Mode().IsRegular() {
			return out, fmt.Errorf("%s is not a regular file", filepath.Base(path))
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return out, fmt.Errorf("read %s: %w", filepath.Base(path), err)
		}
		out.rawSize += int64(len(data))
		fallbackPID := uint64(0)
		if strings.HasPrefix(filepath.Base(path), prefix) {
			fallbackPID, _ = strconv.ParseUint(strings.TrimPrefix(filepath.Base(path), prefix), 10, 64)
		}
		part, damaged, err := decodeStackSidecar(data, fallbackPID)
		if err != nil {
			return out, fmt.Errorf("decode %s: %w", filepath.Base(path), err)
		}
		out.damaged = out.damaged || damaged
		if damaged {
			out.damageNote = "trailing partial sidecar line omitted"
		}
		for i := range part.Stacks {
			if part.Stacks[i].Process == 0 {
				part.Stacks[i].Process = targetPID
			}
		}
		table.Losses += part.Losses
		table.LossesLowerBound = table.LossesLowerBound || part.LossesLowerBound
		table.Stacks = append(table.Stacks, part.Stacks...)
	}
	out.data, err = json.Marshal(table)
	if err != nil {
		return out, fmt.Errorf("encode combined sidecar: %w", err)
	}
	return out, nil
}

func decodeStackSidecar(data []byte, fallbackPID uint64) (cudaprof.ExternalStackTable, bool, error) {
	var table cudaprof.ExternalStackTable
	var discriminator struct {
		Kind string `json:"kind"`
	}
	if json.Unmarshal(data, &table) == nil && table.Version != 0 && json.Unmarshal(data, &discriminator) == nil && discriminator.Kind == "" {
		for i := range table.Stacks {
			if table.Stacks[i].Process == 0 {
				table.Stacks[i].Process = fallbackPID
			}
		}
		return table, false, nil
	}
	lines := bytes.Split(data, []byte{'\n'})
	damaged := len(data) != 0 && data[len(data)-1] != '\n'
	if damaged {
		lines = lines[:len(lines)-1]
	}
	if len(lines) == 0 {
		return table, damaged, fmt.Errorf("no complete sidecar records")
	}
	table.Version = 1
	for n, line := range lines {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var row stackSidecarLine
		if err := json.Unmarshal(line, &row); err != nil {
			return table, false, err
		}
		if row.Version != 1 {
			return table, false, fmt.Errorf("unsupported version %d", row.Version)
		}
		if n == 0 && row.Kind == "header" {
			if row.Losses != "" {
				losses, err := strconv.ParseUint(row.Losses, 10, 64)
				if err != nil {
					return table, false, fmt.Errorf("invalid loss count: %w", err)
				}
				table.Losses += losses
			}
			table.LossesLowerBound = table.LossesLowerBound || row.LowerBound
			continue
		}
		if row.Kind != "stack" || row.ExternalID == 0 || len(row.Frames) == 0 {
			return table, false, fmt.Errorf("invalid record kind or fields %q", row.Kind)
		}
		if row.Process == 0 {
			row.Process = fallbackPID
		}
		table.Stacks = append(table.Stacks, cudaprof.ExternalStackRecord{ExternalKind: row.ExternalKind, ExternalID: row.ExternalID, Process: row.Process, Frames: row.Frames})
	}
	return table, damaged, nil
}

func labelExternalStacks(p *profile.Profile, activities []cudacapture.ActivityRecord, sampleRecords []int, data []byte, targetPID uint64) (*profile.Profile, error) {
	var table cudaprof.ExternalStackTable
	if len(data) != 0 {
		if err := json.Unmarshal(data, &table); err != nil {
			return nil, fmt.Errorf("decode normalized external stack sidecar: %w", err)
		}
	}
	type stackKey struct {
		process uint64
		kind    uint32
		id      uint64
	}
	stacks := make(map[stackKey][][]string)
	for _, record := range table.Stacks {
		stacks[stackKey{record.Process, record.ExternalKind, record.ExternalID}] = append(stacks[stackKey{record.Process, record.ExternalKind, record.ExternalID}], record.Frames)
	}
	byCorrelation := make(map[[2]uint64][]cudacapture.ActivityRecord)
	for _, activity := range activities {
		if activity.Subtype != "external_correlation" {
			continue
		}
		process := activity.Process
		if process == 0 {
			process = targetPID
		}
		corr := uint64(activity.Correlation)
		if corr == 0 {
			corr = uint64(activity.RuntimeCorrelation)
		}
		if corr != 0 {
			key := [2]uint64{process, corr}
			byCorrelation[key] = append(byCorrelation[key], activity)
		}
	}
	out := p.Copy()
	for i, index := range sampleRecords {
		if index < 0 || index >= len(activities) || i >= len(out.Sample) {
			return nil, fmt.Errorf("sample record index %d is out of range", index)
		}
		activity := activities[index]
		process := activity.Process
		if process == 0 {
			process = targetPID
		}
		labels := out.Sample[i].Label
		ids, kinds := labels["cupti.external.id"], labels["cupti.external.kind"]
		if len(ids) != 1 || len(kinds) != 1 {
			continue
		}
		id, err := strconv.ParseUint(ids[0], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid external ID label: %w", err)
		}
		kind, err := strconv.ParseUint(kinds[0], 10, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid external kind label: %w", err)
		}
		corr := uint64(activity.Correlation)
		if corr == 0 {
			corr = uint64(activity.RuntimeCorrelation)
		}
		linked := false
		for _, external := range byCorrelation[[2]uint64{process, corr}] {
			if external.ExternalID == id && external.ExternalKind == uint32(kind) {
				linked = true
				break
			}
		}
		if !linked {
			setStackLabel(out.Sample[i], "cupti.external.stack_status", "missing")
			continue
		}
		matches := stacks[stackKey{process, uint32(kind), id}]
		if len(matches) > 1 {
			setStackLabel(out.Sample[i], "cupti.external.stack_status", "ambiguous")
			continue
		}
		if len(matches) == 0 {
			setStackLabel(out.Sample[i], "cupti.external.stack_status", "missing")
			continue
		}
		setStackLabel(out.Sample[i], "cupti.external.stack", strings.Join(matches[0], ";"))
		setStackLabel(out.Sample[i], "cupti.external.stack_status", "matched")
	}
	if table.Losses > 0 {
		qualifier := "exact"
		if table.LossesLowerBound {
			qualifier = "lower bound"
		}
		out.Comments = append(out.Comments, fmt.Sprintf("cudaprof: external stack losses %s: %d", qualifier, table.Losses))
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
	var table cudaprof.ExternalStackTable
	if err := json.Unmarshal(data, &table); err != nil {
		return nil
	}
	kinds := make(map[uint32]bool)
	for _, record := range table.Stacks {
		kinds[record.ExternalKind] = true
	}
	return kinds
}
