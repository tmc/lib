package tracecontract

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// Track identifies a semantic track independently of its display name. Arm
// scopes independently aligned runs; Journal scopes process generations.
// Device is optional so device zero remains distinct from unavailable identity.
// Canonical marks the sole counting representation for one scope/aggregation;
// child hierarchy tracks and raw references must not repeat that total.
// The zero value lacks the identity required by Write.
type Track struct {
	UUID          uint64  `json:"uuid"`
	Parent        uint64  `json:"parent_uuid,omitempty"`
	Name          string  `json:"name"`
	Role          string  `json:"role"`
	Clock         string  `json:"clock"`
	Machine       string  `json:"machine,omitempty"`
	Arm           string  `json:"arm,omitempty"`
	CaptureSHA256 string  `json:"capture_sha256,omitempty"`
	Journal       string  `json:"journal,omitempty"`
	Device        *uint32 `json:"device,omitempty"`
	Scope         string  `json:"scope_id,omitempty"`
	Level         int     `json:"hierarchy_level"`
	Provenance    string  `json:"evidence_provenance"`
	Aggregation   string  `json:"aggregation"`
	Canonical     bool    `json:"canonical"`
}

// Write registers the descriptor and one SQL-visible metadata instant. The
// official TrackEvent debug_annotations encode cudaprof.track.v1 on this UUID.
// Timestamp must be a valid position in the caller's exported clock domain.
func (t Track) Write(writer *perfetto.Writer, timestamp uint64) error {
	if writer == nil {
		return fmt.Errorf("trace track: missing writer")
	}
	if err := t.validate(); err != nil {
		return err
	}
	if err := writer.WriteTrackDescriptor(t.UUID, t.Name, t.Parent); err != nil {
		return err
	}
	a := []perfetto.DebugAnnotation{number("cudaprof_schema_version", 1), text("cudaprof_schema", "cudaprof.track.v1"), text("cudaprof_role", t.Role), text("cudaprof_arm", t.Arm), text("cudaprof_capture_sha256", t.CaptureSHA256), text("cudaprof_journal", t.Journal), text("cudaprof_scope_id", t.Scope), number("cudaprof_hierarchy_level", uint64(t.Level)), text("cudaprof_evidence_provenance", t.Provenance), text("cudaprof_aggregation", t.Aggregation), {Name: "cudaprof_canonical", Type: perfetto.AnnotationTypeBool, BoolValue: t.Canonical}}
	if t.Device != nil {
		a = append(a, number("cudaprof_device", uint64(*t.Device)))
	}
	a = append(a, text("cudaprof_clock", t.Clock), text("cudaprof_machine", t.Machine))
	return writer.WriteFlowInstant(t.UUID, timestamp, "cudaprof.track.v1", "cudaprof.semantic", a, nil, nil)
}

func (t Track) validate() error {
	if len(t.Name)+len(t.Role)+len(t.Clock)+len(t.Machine)+len(t.Arm)+len(t.CaptureSHA256)+len(t.Journal)+len(t.Scope)+len(t.Provenance)+len(t.Aggregation) > 64<<10 {
		return fmt.Errorf("trace track: metadata exceeds 65536 bytes")
	}
	if t.UUID == 0 || t.UUID == t.Parent || t.Name == "" || t.Role == "" || t.Level < 0 || t.Provenance == "" || t.Clock == "" {
		return fmt.Errorf("trace track: missing or invalid identity")
	}
	if t.CaptureSHA256 != "" && !validHash(t.CaptureSHA256) {
		return fmt.Errorf("trace track: invalid capture hash")
	}
	switch t.Aggregation {
	case "none", "elapsed", "host_union", "gpu_union", "kernel_sum":
	default:
		return fmt.Errorf("trace track: unsupported aggregation %q", t.Aggregation)
	}
	if t.Canonical && (t.Aggregation == "none" || t.Scope == "") {
		return fmt.Errorf("trace track: canonical total lacks a scope or aggregation")
	}
	return nil
}

// ValidateTracks refuses duplicate UUIDs, missing parents and duplicated
// canonical representations of the same arm/journal/device/scope aggregation.
func ValidateTracks(tracks []Track) error {
	ids := map[uint64]bool{}
	type scope struct{ capture, arm, journal, device, name, aggregation string }
	canonical := map[scope]bool{}
	for _, t := range tracks {
		if err := t.validate(); err != nil {
			return err
		}
		if ids[t.UUID] {
			return fmt.Errorf("trace track: duplicate uuid %d", t.UUID)
		}
		ids[t.UUID] = true
		if t.Canonical {
			device := "unavailable"
			if t.Device != nil {
				device = fmt.Sprint(*t.Device)
			}
			k := scope{t.CaptureSHA256, t.Arm, t.Journal, device, t.Scope, t.Aggregation}
			if canonical[k] {
				return fmt.Errorf("trace track: duplicated canonical aggregation")
			}
			canonical[k] = true
		}
	}
	for _, t := range tracks {
		if t.Parent != 0 && !ids[t.Parent] {
			return fmt.Errorf("trace track: missing parent %d", t.Parent)
		}
	}
	parents := make(map[uint64]uint64, len(tracks))
	for _, t := range tracks {
		parents[t.UUID] = t.Parent
	}
	finished := map[uint64]bool{}
	for _, t := range tracks {
		seen := map[uint64]bool{}
		chain := []uint64{}
		for id := t.UUID; id != 0 && !finished[id]; id = parents[id] {
			if seen[id] {
				return fmt.Errorf("trace track: parent cycle")
			}
			seen[id] = true
			chain = append(chain, id)
		}
		for _, id := range chain {
			finished[id] = true
		}
	}
	return nil
}

func text(name, value string) perfetto.DebugAnnotation {
	return perfetto.DebugAnnotation{Name: name, Type: perfetto.AnnotationTypeString, StringValue: value}
}
func number(name string, value uint64) perfetto.DebugAnnotation {
	return perfetto.DebugAnnotation{Name: name, Type: perfetto.AnnotationTypeUint, UintValue: value}
}
