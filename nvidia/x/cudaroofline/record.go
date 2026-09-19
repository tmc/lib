package cudaroofline

import (
	"encoding/json"
	"fmt"
	"io"
)

// RecordVersion is the version of the encoded floor record. It is written into
// every record and checked on decode. Bump it when a field changes meaning; a
// consumer that does not recognize the version must refuse the record rather
// than read the fields it happens to know.
const RecordVersion = 1

// GoldenVectorPath is where this package's golden floor vector is committed,
// relative to the package directory.
//
// The vector exists because the floor has consumers that cannot import this
// package: a fleet exporter in another module, a search harness in a third.
// Agreement between modules that cannot see each other is otherwise a claim
// each side makes separately, and two independent implementations of the same
// formula agree right up until one of them is edited. A file both sides read is
// not a claim; it is a fixture, and a consumer whose floor disagrees with it
// fails its own test.
const GoldenVectorPath = "testdata/floor-vector.json"

// Record is the crossing contract: a device identity, and the floor and verdict
// for each operation judged against it.
//
// The encoding is deliberately plain -- strings, numbers, and nothing that has
// to be constructed by this package -- so that a consumer in another module can
// decode it with the standard library and no dependency on NVIDIA anything. The
// architecture and residency names are NVIDIA facts, but they cross as opaque
// strings that a decoder compares rather than interprets.
type Record struct {
	Version int          `json:"version"`
	Device  DeviceRecord `json:"device"`
	Triples []Triple     `json:"triples"`
}

// DeviceRecord identifies what the floors in a Record were computed against.
// The peaks travel with the floors because a floor is only meaningful against
// the numbers that produced it, and the provenances travel because a rejection
// against a datasheet figure is a weaker statement than one against a measured
// bandwidth.
type DeviceRecord struct {
	Architecture      string  `json:"architecture"`
	Residency         string  `json:"residency"`
	PeakTFLOPS        float64 `json:"peak_tflops"`
	PeakMemoryGBps    float64 `json:"peak_memory_gbps"`
	ComputeProvenance string  `json:"compute_provenance"`
	MemoryProvenance  string  `json:"memory_provenance"`

	// PeakPrecision names the format PeakTFLOPS describes. Without it the
	// number is two numbers.
	PeakPrecision string `json:"peak_precision"`
}

// Triple is one (operation, floor, verdict) row: the operation as claimed, the
// floor the ceilings impose on it, and what judging a duration against that
// floor concluded.
type Triple struct {
	Name       string  `json:"name"`
	FLOPs      uint64  `json:"flops"`
	Bytes      uint64  `json:"bytes"`
	Precision  string  `json:"precision"`
	DurationNs uint64  `json:"duration_ns"`
	FloorNs    float64 `json:"floor_ns"`

	// MemoryFloorNs and ComputeFloorNs are the per-axis minima, with
	// FloorUnavailable for an axis that produced none. A decoder must treat
	// the sentinel as absence and not as a floor of -1 nanoseconds.
	MemoryFloorNs  float64 `json:"memory_floor_ns"`
	ComputeFloorNs float64 `json:"compute_floor_ns"`

	Basis             string `json:"basis"`
	Axis              string `json:"axis"`
	Verdict           string `json:"verdict"`
	CeilingProvenance string `json:"ceiling_provenance"`
	Reason            string `json:"reason,omitempty"`
}

// NewRecord assembles a record from judgments produced by m. It does not judge;
// pass it what Model.Judge returned.
func NewRecord(m *Model, judgments []Judgment) Record {
	r := Record{
		Version: RecordVersion,
		Device: DeviceRecord{
			Architecture:      m.Ceiling.Architecture,
			Residency:         string(m.Ceiling.Residency),
			PeakTFLOPS:        m.Ceiling.PeakTFLOPS,
			PeakMemoryGBps:    m.Ceiling.PeakMemoryGBps,
			ComputeProvenance: string(m.Ceiling.ComputeProvenance),
			MemoryProvenance:  string(m.Ceiling.MemoryProvenance),
			PeakPrecision:     string(FP32),
		},
		Triples: make([]Triple, 0, len(judgments)),
	}
	for _, j := range judgments {
		r.Triples = append(r.Triples, Triple{
			Name:              j.Operation.Name,
			FLOPs:             j.Operation.FLOPs,
			Bytes:             j.Operation.Bytes,
			Precision:         string(j.Operation.Precision),
			DurationNs:        j.DurationNs,
			FloorNs:           j.FloorNs,
			MemoryFloorNs:     j.MemoryFloorNs,
			ComputeFloorNs:    j.ComputeFloorNs,
			Basis:             string(j.Basis),
			Axis:              string(j.Axis),
			Verdict:           string(j.Verdict),
			CeilingProvenance: string(j.CeilingProvenance),
			Reason:            j.Reason,
		})
	}
	return r
}

// Encode writes r as indented JSON with a trailing newline, so a committed
// vector reads as a diff rather than as one line.
func (r Record) Encode(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

// DecodeRecord reads a record and checks its version and shape. It rejects an
// unknown version rather than reading the fields it recognizes.
func DecodeRecord(rd io.Reader) (Record, error) {
	var r Record
	dec := json.NewDecoder(rd)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		return Record{}, fmt.Errorf("cudaroofline: decoding floor record: %w", err)
	}
	if err := r.Validate(); err != nil {
		return Record{}, err
	}
	return r, nil
}

// Validate reports whether a decoded record is usable: a version this package
// understands, a named device, and no triple whose verdict it cannot name.
//
// It does not require the record to contain any particular verdict. A record of
// judgments that all passed is a legitimate record; it is the committed golden
// vector that must exercise both outcomes, and that is asserted where the
// vector is, not here.
func (r Record) Validate() error {
	if r.Version != RecordVersion {
		return fmt.Errorf("cudaroofline: floor record version %d, want %d", r.Version, RecordVersion)
	}
	if r.Device.Architecture == "" || r.Device.Residency == "" {
		return fmt.Errorf("cudaroofline: floor record names no device")
	}
	if len(r.Triples) == 0 {
		return fmt.Errorf("cudaroofline: floor record holds no triples")
	}
	for _, t := range r.Triples {
		switch Verdict(t.Verdict) {
		case VerdictAccepted, VerdictRejected, VerdictUnjudged:
		default:
			return fmt.Errorf("cudaroofline: triple %q: unknown verdict %q", t.Name, t.Verdict)
		}
		switch FloorBasis(t.Basis) {
		case FloorFromBothAxes, FloorFromMemoryOnly, FloorFromComputeOnly, FloorAbsent:
		default:
			return fmt.Errorf("cudaroofline: triple %q: unknown floor basis %q", t.Name, t.Basis)
		}
	}
	return nil
}
