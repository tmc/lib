package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	languagestack "github.com/tmc/lib/nvidia/x/cudaprof/stack"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

type languageSamples struct {
	Journal       string                 `json:"journal"`
	SHA256        string                 `json:"sha256"`
	Receipt       string                 `json:"receipt,omitempty"`
	ReceiptSHA256 string                 `json:"receipt_sha256,omitempty"`
	Samples       []languagestack.Sample `json:"samples"`
	GoTrace       *languagestack.GoTrace `json:"go_trace,omitempty"`
	Unavailable   string                 `json:"unavailable,omitempty"`
}

func readLanguageSamples(cap *cudacapture.Capture) ([]languageSamples, error) {
	var names []string
	for name := range cap.Files {
		if strings.HasPrefix(name, "stacks-python-wall-") && strings.HasSuffix(name, ".jsonl") || strings.HasPrefix(name, "stacks-go-") && strings.HasSuffix(name, "-host.json") {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	var out []languageSamples
	var total int64
	for _, name := range names {
		data := cap.Files[name]
		total += int64(len(data))
		if total > 32<<20 {
			return nil, fmt.Errorf("host stack samples exceed total byte limit")
		}
		digest := sha256.Sum256(data)
		group := languageSamples{Journal: name, SHA256: hex.EncodeToString(digest[:])}
		if strings.HasSuffix(name, ".jsonl") {
			samples, err := languagestack.ReadSamples(bytes.NewReader(data), languagestack.Limits{})
			if err != nil {
				return nil, fmt.Errorf("read %s: %w", name, err)
			}
			group.Samples = samples
		} else {
			var identity struct {
				PID        int    `json:"pid"`
				Process    string `json:"process"`
				Session    string `json:"session"`
				Generation uint64 `json:"generation"`
				GOOS       string `json:"goos"`
				GoVersion  string `json:"go_version"`
			}
			if err := json.Unmarshal(data, &identity); err != nil {
				return nil, fmt.Errorf("read host receipt %s: %w", name, err)
			}
			traceName := strings.TrimSuffix(name, "-host.json") + ".trace"
			trace, ok := cap.Files[traceName]
			if !ok {
				group.Unavailable = "runtime trace attachment missing"
				out = append(out, group)
				continue
			}
			group.Receipt, group.ReceiptSHA256 = group.Journal, group.SHA256
			group.Journal = traceName
			traceDigest := sha256.Sum256(trace)
			group.SHA256 = hex.EncodeToString(traceDigest[:])
			total += int64(len(trace))
			if total > 32<<20 {
				return nil, fmt.Errorf("host stack samples exceed total byte limit")
			}
			goTrace, err := analyzeRetainedGoTrace(trace, languagestack.Record{PID: identity.PID, Process: identity.Process, Session: identity.Session, Generation: identity.Generation})
			if err != nil {
				group.Unavailable = err.Error()
			} else {
				group.GoTrace = goTrace
				if err := languagestack.QualifyGoMonotonic(goTrace, identity.GOOS, identity.GoVersion); err != nil {
					group.Unavailable = err.Error()
				}
				group.Samples = goTrace.Samples
				metadata := *goTrace
				metadata.Samples = nil
				group.GoTrace = &metadata
			}
		}
		out = append(out, group)
	}
	return out, nil
}

func nativeStackClocks(records []traceRecord) map[stackReference]bool {
	clocks := make(map[stackReference]bool)
	for _, t := range records {
		clocks[stackReference{session: t.session, pid: int(t.process), generation: t.generation}] = t.clocksUnified
	}
	return clocks
}

func appendLanguageSamples(cap *cudacapture.Capture, records []traceRecord) ([]traceRecord, error) {
	groups, err := readLanguageSamples(cap)
	if err != nil {
		return nil, err
	}
	clocks := nativeStackClocks(records)
	var count int
	for _, group := range groups {
		for i := range group.Samples {
			s := group.Samples[i]
			if s.Type != "sample" {
				continue
			}
			count++
			if count > 100000 {
				return nil, fmt.Errorf("host stack samples exceed total record limit")
			}
			key := stackReference{session: s.Session, pid: s.PID, generation: s.Generation}
			if !clocks[key] || s.Generation == 0 || s.Clock != "monotonic" || s.Time == 0 || s.TID > math.MaxUint32 {
				continue
			}
			s.Status = "native_clock_verified"
			annotations := []perfetto.DebugAnnotation{
				perfetto.NewStringAnnotation("cudaprof_stack_language", s.Language),
				perfetto.NewStringAnnotation("cudaprof_stack_attribution", s.Provenance),
				perfetto.NewStringAnnotation("cudaprof_stack_status", s.Status),
				perfetto.NewStringAnnotation("cudaprof_stack_journal", group.Journal),
				perfetto.NewStringAnnotation("cudaprof_stack_journal_sha256", group.SHA256),
				perfetto.NewUintAnnotation("cudaprof_stack_goroutine", s.Goroutine),
				perfetto.NewBoolAnnotation("cudaprof_stack_truncated", s.Truncated),
			}
			records = append(records, traceRecord{journal: group.Journal, sequence: uint64(i + 1), annotations: annotations, sample: &s, session: s.Session, process: uint64(s.PID), generation: s.Generation, clocksUnified: true,
				rec: cuptiactivity.Record{Kind: cupti.ActivityKindOverhead, Name: s.Language + " " + s.Provenance, Start: s.Time, End: s.Time, Complete: true, ProcessID: uint32(s.PID), ThreadID: uint32(s.TID)}, kind: trackStackSample, group: [2]uint32{uint32(s.PID), uint32(s.TID)}})
		}
	}
	return records, nil
}

func analyzeRetainedGoTrace(data []byte, identity languagestack.Record) (*languagestack.GoTrace, error) {
	return languagestack.AnalyzeGoTraceBytes(context.Background(), "go", data, identity, languagestack.Limits{})
}
