package main

import (
	"context"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	languagestack "github.com/tmc/lib/nvidia/x/cudaprof/stack"
)

type languageStackMatch struct {
	Journal  string                `json:"journal"`
	Sequence uint64                `json:"raw_record_sequence"`
	Kind     string                `json:"activity_kind"`
	Status   string                `json:"status"`
	Stack    *languagestack.Record `json:"stack,omitempty"`
}

type languageStackView struct {
	State    string                 `json:"state"`
	Journals []languageStackJournal `json:"journals,omitempty"`
	Matches  []languageStackMatch   `json:"matches,omitempty"`
	Groups   []languageSamples      `json:"groups,omitempty"`
	Limits   string                 `json:"limits"`
}

func nativeLanguageStackView(ctx context.Context, cap *cudacapture.Capture, name string, goTrace goTraceOptions) (any, error) {
	journals, err := requireJournals(cap)
	if err != nil {
		return nil, err
	}
	maxRecords, maxBytes := decodeLimits(cap, 0, 0)
	records, _, _, err := decodeTraceJournals(journals, maxRecords, maxBytes, name == "language-stacks")
	if err != nil {
		return nil, err
	}
	if name == "host-stacks" {
		groups, err := readLanguageSamples(ctx, cap, goTrace)
		if err != nil {
			return nil, err
		}
		clocks := nativeStackClocks(records)
		for i := range groups {
			for j := range groups[i].Samples {
				sample := &groups[i].Samples[j]
				if sample.Type != "sample" || sample.Generation == 0 || sample.Clock != "monotonic" {
					continue
				}
				if clocks[stackReference{session: sample.Session, pid: sample.PID, generation: sample.Generation}] {
					sample.Status = "native_clock_verified"
				}
			}
		}
		state := "unavailable"
		if len(groups) > 0 {
			state = "observed"
		}
		return languageStackView{State: state, Groups: groups, Limits: "wall snapshots include blocked threads; CPU samples are statistical; neither establishes submission causality"}, nil
	}
	stackJournals, err := readLanguageStackJournals(cap)
	if err != nil {
		return nil, err
	}
	if err := attachLanguageStacks(cap, records); err != nil {
		return nil, err
	}
	var matches []languageStackMatch
	for _, record := range records {
		status := "not_observed"
		for _, annotation := range record.annotations {
			if annotation.Name == "cudaprof_stack_status" {
				status = annotation.StringValue
			}
		}
		matches = append(matches, languageStackMatch{record.journal, record.sequence, traceRecordKind(record), status, record.stack})
	}
	state := "unavailable"
	if len(stackJournals) > 0 {
		state = "observed"
	}
	return languageStackView{State: state, Journals: stackJournals, Matches: matches, Limits: "scope entry and synchronous submission attribution are distinct; graph replay stacks do not recover node creation origins"}, nil
}
