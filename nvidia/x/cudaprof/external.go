package cudaprof

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// LabelExternal copies p and labels device activity samples with exact native
// external-correlation status, namespace and ID. A matched ID does not establish
// a Python operator or frame; that requires a separately retained scope producer.
// Unmatched and ambiguous samples remain in the profile with explicit status.
// Values, locations and existing labels are unchanged.
//
// recordIndices[i] identifies the native record from which p.Sample[i] was
// projected. The exporter must retain this association while building samples;
// it must not reconstruct it by timestamp or name proximity. Every concurrent
// kernel, memcpy and memset record must appear exactly once. API and other
// records must not be bound to a device sample. Labels use the cupti.external.*
// and cupti.collection.* namespaces; existing labels in those namespaces cause
// an error. The caller must preserve original records and loss information.
func LabelExternal(p *profile.Profile, records []cuptiactivity.ScopedRecord, recordIndices []int, kind cupti.ExternalCorrelationKind) (*profile.Profile, error) {
	if p == nil || len(p.Sample) != len(recordIndices) {
		return nil, fmt.Errorf("cudaprof: sample/record association mismatch")
	}
	if err := p.CheckValid(); err != nil {
		return nil, fmt.Errorf("cudaprof: invalid profile: %w", err)
	}
	matches, err := cuptiactivity.MatchExternal(records, kind)
	if err != nil {
		return nil, err
	}
	links := make(map[int]cuptiactivity.ExternalMatch)
	for _, m := range matches {
		switch records[m.Index].Record.Kind {
		case cupti.ActivityKindConcurrentKernel, cupti.ActivityKindMemcpy, cupti.ActivityKindMemset:
			links[m.Index] = m
		}
	}
	out := p.Copy()
	for i, index := range recordIndices {
		m, ok := links[index]
		if !ok {
			return nil, fmt.Errorf("cudaprof: sample %d has invalid or reused native record %d", i, index)
		}
		delete(links, index)
		r := records[index]
		s := out.Sample[i]
		for name := range s.Label {
			if externalLabel(name) {
				return nil, fmt.Errorf("cudaprof: sample %d already has external correlation labels", i)
			}
		}
		for name := range s.NumLabel {
			if externalLabel(name) {
				return nil, fmt.Errorf("cudaprof: sample %d already has numeric external correlation labels", i)
			}
		}
		if s.Label == nil {
			s.Label = make(map[string][]string)
		}
		s.Label["cupti.external.status"] = []string{m.Status}
		s.Label["cupti.external.kind"] = []string{strconv.FormatInt(int64(kind), 10)}
		s.Label["cupti.collection.session"] = []string{r.Session}
		s.Label["cupti.collection.run"] = []string{r.Run}
		s.Label["cupti.collection.process"] = []string{r.Process}
		if m.Status == "matched" {
			s.Label["cupti.external.id"] = []string{strconv.FormatUint(m.ExternalID, 10)}
		}
	}
	if len(links) != 0 {
		return nil, fmt.Errorf("cudaprof: %d device records have no sample", len(links))
	}
	return out, nil
}

func externalLabel(name string) bool {
	return strings.HasPrefix(name, "cupti.external.") || strings.HasPrefix(name, "cupti.collection.")
}
