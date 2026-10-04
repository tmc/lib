package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
	Epoch        string   `json:"epoch,omitempty"`
	ExternalKind uint32   `json:"external_kind,omitempty"`
	ExternalID   uint64   `json:"external_id,omitempty"`
	Frames       []string `json:"frames,omitempty"`
}

type stackSidecarRead struct {
	data        []byte
	rawSize     int64
	damaged     bool
	damageNote  string
	damageFiles []string
	exists      bool
}

const maxStackSidecarBytes = 16 << 20
const maxStackSidecarRowBytes = 64 << 10
const maxStackSidecarRows = 16384
const maxStackSidecarReadBytes = 64 << 20

var readExternalStackFile = func(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, limit))
	if len(data) > maxStackSidecarBytes || int64(len(data)) == limit {
		return data, fmt.Errorf("sidecar exceeds byte budget")
	}
	return data, err
}

// readExternalStackSidecar collects process-isolated sidecars into one bounded
// capture member. Newline framing permits recovery after malformed records.
func readExternalStackSidecar(base string, targetPID uint64, expectedEpoch ...string) (stackSidecarRead, error) {
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
	directory, err := os.Open(filepath.Dir(base))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return stackSidecarRead{exists: true}, fmt.Errorf("list sidecars: %w", err)
	}
	var entries []os.DirEntry
	if err == nil {
		defer directory.Close()
		entries, err = directory.ReadDir(4096)
		if err != nil && !errors.Is(err, io.EOF) {
			return out, fmt.Errorf("list sidecars: %w", err)
		}
		if len(entries) == 4096 {
			out.damaged = true
			out.damageNote = "sidecar directory entry budget reached"
		}
	}
	prefix := filepath.Base(base) + "."
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), prefix) {
			continue
		}
		pidText := strings.TrimPrefix(entry.Name(), prefix)
		pid, parseErr := strconv.ParseUint(pidText, 10, 64)
		if parseErr != nil || pid == 0 {
			continue
		}
		paths = append(paths, filepath.Join(filepath.Dir(base), entry.Name()))
		if len(paths) >= 256 {
			out.damaged = true
			out.damageNote = "sidecar process file budget reached"
			break
		}
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		return out, nil
	}
	out.exists = true
	table := cudaprof.ExternalStackTable{Version: 1}
	var baseErr error
	var readBytes int64
	markLostFile := func() {
		table.DiscardedRecordsLowerBound = true
		table.LossesLowerBound = true
	}
	if out.damaged {
		markLostFile()
	}
	for _, path := range paths {
		baseFile := filepath.Clean(path) == filepath.Clean(base)
		info, err := os.Lstat(path)
		if err != nil {
			if baseFile {
				baseErr = fmt.Errorf("stat %s: %w", filepath.Base(path), err)
			}
			out.damaged = true
			out.damageFiles = append(out.damageFiles, filepath.Base(path))
			markLostFile()
			continue
		}
		if !info.Mode().IsRegular() {
			if baseFile {
				baseErr = fmt.Errorf("%s is not a regular file", filepath.Base(path))
			}
			out.damaged = true
			out.damageFiles = append(out.damageFiles, filepath.Base(path))
			markLostFile()
			continue
		}
		remainingBytes := int64(maxStackSidecarReadBytes) - readBytes
		if remainingBytes == 0 || info.Size() > maxStackSidecarBytes || info.Size() > remainingBytes || len(table.Stacks) >= maxStackSidecarRows {
			out.damaged = true
			out.damageFiles = append(out.damageFiles, filepath.Base(path))
			markLostFile()
			continue
		}
		// Reserve the maximum read before opening the file. A growing file or
		// a failed read keeps its reservation, so errors cannot reset the
		// aggregate budget. Successful reads refund only unused allowance.
		allowance := min(int64(maxStackSidecarBytes+1), remainingBytes)
		readBytes += allowance
		data, err := readExternalStackFile(path, allowance)
		if err == nil && int64(len(data)) < allowance {
			readBytes -= allowance - int64(len(data))
		} else if err == nil {
			err = fmt.Errorf("sidecar exceeds read allowance")
		}
		if err != nil {
			if baseFile {
				baseErr = fmt.Errorf("read %s: %w", filepath.Base(path), err)
			}
			out.damaged = true
			out.damageFiles = append(out.damageFiles, filepath.Base(path))
			markLostFile()
			continue
		}
		out.rawSize += int64(len(data))
		fallbackPID := uint64(0)
		if strings.HasPrefix(filepath.Base(path), prefix) {
			fallbackPID, _ = strconv.ParseUint(strings.TrimPrefix(filepath.Base(path), prefix), 10, 64)
		}
		part, damaged, err := decodeStackSidecar(data, fallbackPID, expectedEpoch...)
		if err != nil {
			if baseFile {
				baseErr = fmt.Errorf("decode %s: %w", filepath.Base(path), err)
			}
			out.damaged = true
			out.damageFiles = append(out.damageFiles, filepath.Base(path))
			if part.DiscardedRecords > 0 {
				if ^uint64(0)-table.DiscardedRecords < part.DiscardedRecords {
					table.DiscardedRecords = ^uint64(0)
					table.DiscardedRecordsLowerBound = true
				} else {
					table.DiscardedRecords += part.DiscardedRecords
				}
				table.DiscardedRecordsLowerBound = table.DiscardedRecordsLowerBound || part.DiscardedRecordsLowerBound
				table.LossesLowerBound = true
			} else {
				markLostFile()
			}
			continue
		}
		out.damaged = out.damaged || damaged
		if damaged {
			out.damageNote = "invalid or partial sidecar records omitted"
			out.damageFiles = append(out.damageFiles, filepath.Base(path))
		}
		for i := range part.Stacks {
			if part.Stacks[i].Process == 0 {
				part.Stacks[i].Process = targetPID
			}
		}
		if ^uint64(0)-table.Losses < part.Losses {
			table.Losses = ^uint64(0)
			table.LossesLowerBound = true
		} else {
			table.Losses += part.Losses
		}
		table.LossesLowerBound = table.LossesLowerBound || part.LossesLowerBound
		if ^uint64(0)-table.DiscardedRecords < part.DiscardedRecords {
			table.DiscardedRecords = ^uint64(0)
			table.DiscardedRecordsLowerBound = true
		} else {
			table.DiscardedRecords += part.DiscardedRecords
		}
		table.DiscardedRecordsLowerBound = table.DiscardedRecordsLowerBound || part.DiscardedRecordsLowerBound
		remaining := maxStackSidecarRows - len(table.Stacks)
		if len(part.Stacks) > remaining {
			out.damaged = true
			table.LossesLowerBound = true
			discarded := uint64(len(part.Stacks) - remaining)
			if ^uint64(0)-table.DiscardedRecords < discarded {
				table.DiscardedRecords = ^uint64(0)
				table.DiscardedRecordsLowerBound = true
			} else {
				table.DiscardedRecords += discarded
			}
			part.Stacks = part.Stacks[:remaining]
		}
		table.Stacks = append(table.Stacks, part.Stacks...)
	}
	if baseErr != nil && len(table.Stacks) == 0 {
		return out, baseErr
	}
	out.data, err = json.Marshal(table)
	if err != nil {
		return out, fmt.Errorf("encode combined sidecar: %w", err)
	}
	return out, nil
}

func decodeStackSidecar(data []byte, fallbackPID uint64, expectedEpoch ...string) (cudaprof.ExternalStackTable, bool, error) {
	if len(data) > maxStackSidecarBytes {
		return cudaprof.ExternalStackTable{}, true, fmt.Errorf("sidecar exceeds byte budget")
	}
	var table cudaprof.ExternalStackTable
	var discriminator struct {
		Kind string `json:"kind"`
	}
	if json.Unmarshal(data, &table) == nil && table.Version != 0 && json.Unmarshal(data, &discriminator) == nil && discriminator.Kind == "" {
		if table.Version != 1 || len(table.Stacks) > maxStackSidecarRows {
			return cudaprof.ExternalStackTable{}, true, fmt.Errorf("unsupported or oversized stack table")
		}
		for i := range table.Stacks {
			if table.Stacks[i].ExternalID == 0 || !validStackSidecarFrames(table.Stacks[i].Frames) || (fallbackPID != 0 && table.Stacks[i].Process != 0 && table.Stacks[i].Process != fallbackPID) {
				return cudaprof.ExternalStackTable{}, true, fmt.Errorf("invalid stack table record")
			}
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
		if len(data) > 0 {
			table = cudaprof.ExternalStackTable{Version: 1, DiscardedRecords: 1, DiscardedRecordsLowerBound: true, LossesLowerBound: true}
		}
		return table, damaged, fmt.Errorf("no complete sidecar records")
	}
	table.Version = 1
	validRow := false
	var discarded uint64
	if damaged {
		discarded++
	}
	bad := func() { discarded++; damaged = true }
	for n, line := range lines {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var row stackSidecarLine
		if len(line) > maxStackSidecarRowBytes || len(table.Stacks) >= maxStackSidecarRows {
			bad()
			continue
		}
		if err := decodeStackSidecarLine(line, &row); err != nil {
			bad()
			continue
		}
		if row.Version != 1 || (fallbackPID != 0 && row.Process != 0 && row.Process != fallbackPID) || (len(expectedEpoch) > 0 && row.Epoch != "" && row.Epoch != expectedEpoch[0]) {
			bad()
			continue
		}
		if n == 0 && row.Kind == "header" {
			if row.Losses != "" {
				losses, err := strconv.ParseUint(row.Losses, 10, 64)
				if err != nil {
					bad()
					continue
				}
				table.Losses += losses
			}
			table.LossesLowerBound = table.LossesLowerBound || row.LowerBound
			validRow = true
			continue
		}
		if row.Kind == "losses" {
			losses, err := strconv.ParseUint(row.Losses, 10, 64)
			if err != nil {
				bad()
				continue
			}
			if losses < table.Losses {
				bad()
				continue
			}
			table.Losses = losses
			table.LossesLowerBound = table.LossesLowerBound || row.LowerBound
			validRow = true
			continue
		}
		if row.Kind != "stack" || row.ExternalID == 0 || !validStackSidecarFrames(row.Frames) {
			bad()
			continue
		}
		if row.Process == 0 {
			row.Process = fallbackPID
		}
		table.Stacks = append(table.Stacks, cudaprof.ExternalStackRecord{ExternalKind: row.ExternalKind, ExternalID: row.ExternalID, Process: row.Process, Frames: row.Frames})
		validRow = true
	}
	if discarded > 0 {
		if ^uint64(0)-table.DiscardedRecords < discarded {
			table.DiscardedRecords = ^uint64(0)
		} else {
			table.DiscardedRecords += discarded
		}
		table.DiscardedRecordsLowerBound = len(data) != 0 && data[len(data)-1] != '\n'
		table.LossesLowerBound = true
	}
	if !validRow {
		return table, true, fmt.Errorf("no valid complete sidecar records")
	}
	return table, damaged, nil
}

func decodeStackSidecarLine(data []byte, row *stackSidecarLine) error {
	d := json.NewDecoder(bytes.NewReader(data))
	token, err := d.Token()
	if err != nil || token != json.Delim('{') {
		return fmt.Errorf("invalid sidecar object")
	}
	seen := make(map[string]bool)
	for d.More() {
		token, err = d.Token()
		if err != nil {
			return err
		}
		key, ok := token.(string)
		if !ok || seen[key] {
			return fmt.Errorf("duplicate sidecar key")
		}
		seen[key] = true
		switch key {
		case "version", "kind", "losses", "losses_lower_bound", "process", "epoch", "external_kind", "external_id", "frames":
		default:
			return fmt.Errorf("unknown sidecar key %q", key)
		}
		var value json.RawMessage
		if err := d.Decode(&value); err != nil {
			return err
		}
	}
	if _, err = d.Token(); err != nil {
		return err
	}
	if _, err = d.Token(); err != io.EOF {
		return fmt.Errorf("trailing sidecar data")
	}
	return json.Unmarshal(data, row)
}

func validStackSidecarFrames(frames []string) bool {
	if len(frames) == 0 || len(frames) > 65 {
		return false
	}
	var size int
	for _, f := range frames {
		if f == "" || len(f) > 4096 {
			return false
		}
		size += len(f)
		if size > 8192 {
			return false
		}
	}
	return true
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
	if table.DiscardedRecords > 0 || table.DiscardedRecordsLowerBound {
		qualifier := "exact"
		if table.DiscardedRecordsLowerBound {
			qualifier = "lower bound"
		}
		out.Comments = append(out.Comments, fmt.Sprintf("cudaprof: discarded sidecar records %s: %d; not proven lost stacks", qualifier, table.DiscardedRecords))
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
