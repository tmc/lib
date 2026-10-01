package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	languagestack "github.com/tmc/lib/nvidia/x/cudaprof/stack"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// retainStackCorrelations keeps each external namespace independently. The
// ordinary framework projection chooses one namespace; stack attribution must
// not discard an application's stack namespace in favor of a framework ID.
func retainStackCorrelations(records []traceRecord, external []cuptiactivity.Record) error {
	if len(records) == 0 || len(external) == 0 {
		return nil
	}
	var scoped []cuptiactivity.ScopedRecord
	r := records[0]
	process := fmt.Sprintf("%d/%d", r.process, r.generation)
	add := func(record cuptiactivity.Record) {
		scoped = append(scoped, cuptiactivity.ScopedRecord{Session: r.session, Run: r.session, Process: process, Record: record})
	}
	for _, t := range records {
		add(t.rec)
	}
	kinds := make(map[cupti.ExternalCorrelationKind]bool)
	for _, e := range external {
		add(e)
		kinds[e.ExternalKind] = true
	}
	for kind := range kinds {
		matches, err := cuptiactivity.MatchExternal(scoped, kind)
		if err != nil {
			return err
		}
		for _, match := range matches {
			if match.Index >= len(records) {
				continue
			}
			t := &records[match.Index]
			if t.external == nil {
				t.external = make(map[uint32]cuptiactivity.ExternalMatch)
			}
			t.external[uint32(kind)] = match
		}
	}
	return nil
}

type stackReference struct {
	session    string
	pid        int
	generation uint64
	kind       uint32
	id         uint64
}

type languageStackJournal struct {
	Name    string                 `json:"journal"`
	SHA256  string                 `json:"sha256"`
	Records []languagestack.Record `json:"records"`
	Final   bool                   `json:"final"`
}

func haveLanguageStacks(cap *cudacapture.Capture) bool {
	for name := range cap.Files {
		if strings.HasPrefix(name, "stacks-") && strings.HasSuffix(name, ".jsonl") && !strings.Contains(name, "python-wall-") {
			return true
		}
	}
	return false
}

func readLanguageStackJournals(cap *cudacapture.Capture) ([]languageStackJournal, error) {
	var names []string
	for name := range cap.Files {
		if strings.HasPrefix(name, "stacks-") && strings.HasSuffix(name, ".jsonl") && !strings.Contains(name, "python-wall-") {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	var out []languageStackJournal
	var total int64
	var count int
	for _, name := range names {
		total += int64(len(cap.Files[name]))
		if total > 32<<20 {
			return nil, fmt.Errorf("language stack journals exceed total byte limit")
		}
		records, err := languagestack.Read(bytes.NewReader(cap.Files[name]), languagestack.Limits{})
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", name, err)
		}
		count += len(records)
		if count > 100000 {
			return nil, fmt.Errorf("language stack journals exceed total reference limit")
		}
		digest := sha256.Sum256(cap.Files[name])
		journal := languageStackJournal{Name: name, SHA256: hex.EncodeToString(digest[:]), Records: records}
		for _, record := range records {
			if record.Type == "coverage" && record.Final {
				journal.Final = true
			}
		}
		out = append(out, journal)
	}
	return out, nil
}

func attachLanguageStacks(cap *cudacapture.Capture, records []traceRecord) error {
	journals, err := readLanguageStackJournals(cap)
	if err != nil {
		return err
	}

	refs := make(map[stackReference]*languagestack.Record)
	namespaces := make(map[stackReference]bool)
	sources := make(map[stackReference]string)
	hashes := make(map[stackReference]string)
	finals := make(map[stackReference]bool)
	for _, journal := range journals {
		name, hash, entries := journal.Name, journal.SHA256, journal.Records

		for i := range entries {
			r := &entries[i]
			if r.Type == "coverage" {
				continue
			}
			key := stackReference{r.Session, r.PID, r.Generation, r.ExternalKind, r.ExternalID}
			if _, ok := refs[key]; ok {
				return fmt.Errorf("duplicate language stack reference in %s", name)
			}
			refs[key] = r
			namespace := key
			namespace.id = 0
			namespaces[namespace] = true
			if len(refs) > 100000 {
				return fmt.Errorf("language stack journals exceed total reference limit")
			}
			sources[key] = name
			finals[key] = journal.Final
			hashes[key] = hash
		}
	}
	if len(journals) == 0 {
		return nil
	}
	for i := range records {
		t := &records[i]
		status := "not_observed"
		if t.session == "" || t.process == 0 || t.generation == 0 {
			t.annotations = append(t.annotations, perfetto.NewStringAnnotation("cudaprof_stack_status", "identity_unavailable"))
			continue
		}
		for kind, match := range t.external {
			if match.Status == "ambiguous" && namespaces[stackReference{t.session, int(t.process), t.generation, kind, 0}] {
				t.stack = nil
				status = "ambiguous"
				continue
			}
			if match.Status != "matched" {
				continue
			}
			r := refs[stackReference{t.session, int(t.process), t.generation, kind, match.ExternalID}]
			if r == nil {
				continue
			}
			if status == "ambiguous" {
				continue
			}
			if t.stack != nil {
				t.stack = nil
				status = "ambiguous"
				continue
			}
			status = r.Status
			if status == "attached" {
				t.stack = r
			}
		}
		t.annotations = append(t.annotations, perfetto.NewStringAnnotation("cudaprof_stack_status", status))
		if r := t.stack; r != nil {
			key := stackReference{r.Session, r.PID, r.Generation, r.ExternalKind, r.ExternalID}
			t.annotations = append(t.annotations,
				perfetto.NewStringAnnotation("cudaprof_stack_journal", sources[key]),
				perfetto.NewBoolAnnotation("cudaprof_stack_final", finals[key]),
				perfetto.NewStringAnnotation("cudaprof_stack_journal_sha256", hashes[key]),
				perfetto.NewStringAnnotation("cudaprof_stack_language", r.Language),
				perfetto.NewStringAnnotation("cudaprof_stack_attribution", r.Attribution),
				perfetto.NewStringAnnotation("cudaprof_stack_process", r.Process),
				perfetto.NewStringAnnotation("cudaprof_stack_external_id", strconv.FormatUint(r.ExternalID, 10)),
				perfetto.NewUintAnnotation("cudaprof_stack_external_kind", uint64(r.ExternalKind)),
				perfetto.NewBoolAnnotation("cudaprof_stack_truncated", r.Truncated))
		}
	}
	return nil
}

func internLanguageStack(pw *perfetto.Writer, r *languagestack.Record) (uint64, error) {
	if r == nil {
		return 0, nil
	}
	return internLanguageFrames(pw, r.Frames)
}

func internLanguageFrames(pw *perfetto.Writer, source []languagestack.Frame) (uint64, error) {
	frames := make([]perfetto.StackFrame, len(source))
	for i, f := range source {
		frames[i] = perfetto.StackFrame{Function: f.Function, File: f.File, Line: f.Line}
	}
	return pw.InternCallstack(frames)
}
