package main

import (
	"archive/tar"
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"io"
	"math"
	"path"
	"strings"
)

// inspectTraceArchive validates supported alignment metadata without extraction.
// Historical receipts lack member hashes: their qualification is unavailable.
func inspectTraceArchive(data []byte) (map[string][]byte, error) {
	return inspectTraceArchiveLimits(data, defaultTraceInputLimits())
}
func inspectTraceArchiveLimits(data []byte, limits traceInputLimits) (map[string][]byte, error) {
	reader := bytes.NewReader(data)
	tr := tar.NewReader(reader)
	members := make(map[string][]byte)
	total := int64(0)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			remaining, readErr := io.ReadAll(reader)
			if readErr != nil {
				return nil, readErr
			}
			for _, b := range remaining {
				if b != 0 {
					return nil, errors.New("trace archive: trailing nonzero data")
				}
			}
			break
		}
		if err != nil {
			return nil, fmt.Errorf("trace archive: %w", err)
		}
		if h.Typeflag != tar.TypeReg && h.Typeflag != tar.TypeRegA {
			return nil, errors.New("trace archive: nonregular member")
		}
		if h.Name == "" || path.IsAbs(h.Name) || path.Clean(h.Name) != h.Name || (h.Name == ".." || strings.HasPrefix(h.Name, "../")) || bytes.Contains([]byte(h.Name), []byte("\\")) {
			return nil, errors.New("trace archive: unsafe member path")
		}
		if _, ok := members[h.Name]; ok {
			return nil, errors.New("trace archive: duplicate member")
		}
		if h.Size < 0 || h.Size > limits.ArchiveBytes || h.Size > limits.ArchiveBytes-total || len(members) >= limits.ArchiveMembers {
			return nil, errors.New("trace archive: expansion limit exceeded")
		}
		total += h.Size
		contents, err := io.ReadAll(io.LimitReader(tr, h.Size+1))
		if err != nil {
			return nil, err
		}
		if int64(len(contents)) != h.Size {
			return nil, errors.New("trace archive: truncated member")
		}
		members[h.Name] = contents
	}
	if data, ok := members[".cudaprof-bindings.json"]; ok {
		var bindings tracecontract.Bindings
		if err := json.Unmarshal(data, &bindings); err != nil {
			return nil, fmt.Errorf("trace archive bindings: %w", err)
		}
		if err := bindings.Verify(members); err != nil {
			return nil, err
		}
	}
	var receipt alignmentReceipt
	if err := json.Unmarshal(members[".alignment.json"], &receipt); err != nil {
		return nil, fmt.Errorf("trace archive alignment receipt: %w", err)
	}
	if receipt.Version != 1 || receipt.TimeScale != 1 {
		return nil, errors.New("trace archive: unsupported alignment version or time scale")
	}
	var manifest struct {
		Manifest struct {
			Version int                          `json:"version"`
			Time    struct{ Clock, File string } `json:"trace_time"`
			Files   []struct {
				Path   string
				Clocks struct {
					Clock  string
					Offset int64                        `json:"offset_ns"`
					Sync   struct{ File, Clock string } `json:"sync_to"`
				}
			} `json:"files"`
		} `json:"perfetto_manifest"`
	}
	if err := json.Unmarshal(members["perfetto_manifest.json"], &manifest); err != nil {
		return nil, fmt.Errorf("trace archive manifest: %w", err)
	}
	m := manifest.Manifest
	if receipt.Arms[0].Trace == receipt.Arms[1].Trace {
		return nil, errors.New("trace archive: arm trace paths must differ")
	}
	if m.Version != 1 || m.Time.Clock != "BOOTTIME" || m.Time.File != receipt.Arms[0].Trace || len(m.Files) != 2 {
		return nil, errors.New("trace archive: unsupported manifest clock or members")
	}
	for i, a := range receipt.Arms {
		if _, err := hex.DecodeString(a.SHA256); err != nil || len(a.SHA256) != 64 {
			return nil, errors.New("trace archive: invalid capture identity")
		}
		if m.Files[i].Path != a.Trace || !validTracePackets(members[a.Trace]) {
			return nil, errors.New("trace archive: missing or malformed trace member")
		}
		if i == 1 {
			if receipt.Arms[0].Offset > 0 && a.Offset < math.MinInt64+receipt.Arms[0].Offset || receipt.Arms[0].Offset < 0 && a.Offset > math.MaxInt64+receipt.Arms[0].Offset {
				return nil, errors.New("trace archive: alignment offset difference overflows")
			}
			c := m.Files[i].Clocks
			if c.Clock != "BOOTTIME" || c.Sync.Clock != "BOOTTIME" || c.Sync.File != receipt.Arms[0].Trace || c.Offset != a.Offset-receipt.Arms[0].Offset {
				return nil, errors.New("trace archive: alignment clock receipt mismatch")
			}
		}
	}
	if data, ok := members[".cudaprof-bindings.json"]; ok {
		var bindings tracecontract.Bindings
		if err := json.Unmarshal(data, &bindings); err != nil {
			return nil, err
		}
		traces := 0
		for _, member := range bindings.Members {
			if member.Kind != "trace" {
				continue
			}
			traces++
			matched := false
			for _, arm := range receipt.Arms {
				if member.Path == arm.Trace && member.CaptureSHA256 == arm.SHA256 {
					matched = true
				}
			}
			if !matched {
				return nil, errors.New("trace archive: trace binding disagrees with alignment source identity")
			}
		}
		if traces != 2 {
			return nil, errors.New("trace archive: expected exactly two bound trace members")
		}
	}
	return members, nil
}
