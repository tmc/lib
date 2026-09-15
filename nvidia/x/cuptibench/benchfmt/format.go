package benchfmt

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

// Run is a set of independently retained external measurements. Config contains
// the same file headers as the Go arm (for example goos, goarch, pkg, cpu).
// Source-specific provenance belongs in Samples, not Config.
type Run struct {
	Version int               `json:"version"` // Must be 1.
	Config  map[string]string `json:"config"`
	Samples []Sample          `json:"samples"`
}

// Sample is one measurement of a named workload. Name is the exact Go benchmark
// name, including its subbenchmark path and processor suffix when present.
// ID and Source identify the original sample; repeated IDs for the same source
// and name are rejected. Scope distinguishes full, isolated, or attributed work.
type Sample struct {
	Name         string            `json:"name"`
	ID           string            `json:"sample_id"`
	Source       string            `json:"source"`
	Operations   int64             `json:"operations"`
	Scope        string            `json:"scope"`
	Provenance   map[string]string `json:"provenance"`
	Measurements []Measurement     `json:"measurements"`
}

// Measurement is an explicitly classified number. Value is required; a nil
// pointer represents missing data and cannot become a measured zero.
//
// Basis is total, per-operation, or raw. Total divides by Sample.Operations;
// the other bases preserve Value. Total applies only to additive counters and
// elapsed durations and requires an output unit ending in /op.
//
// Kind is counter, host-elapsed, device-elapsed, kernel-duration-sum, rate, or
// ratio. Timing units are respectively ns/op, gpu-ns/op, and kernel-sum-ns/op.
// These labels describe producer assertions, not validation of its instruments.
type Measurement struct {
	Unit  string   `json:"unit"`
	Value *float64 `json:"value"`
	Basis string   `json:"basis"`
	Kind  string   `json:"kind"`
}

const maxInputBytes = 64 << 20

// Decode reads one canonical JSON Run, rejecting unknown fields, unsupported
// versions, trailing documents, and input larger than 64 MiB (including trailing
// whitespace). Input is bounded while reading, before decoding can accumulate
// an unbounded document. Write performs semantic validation before output.
func Decode(r io.Reader) (Run, error) {
	return decodeLimit(r, maxInputBytes)
}

func decodeLimit(r io.Reader, limit int64) (Run, error) {
	var run Run
	input := &io.LimitedReader{R: r, N: limit + 1}
	d := json.NewDecoder(input)
	d.DisallowUnknownFields()
	err := d.Decode(&run)
	if input.N == 0 {
		return Run{}, fmt.Errorf("benchfmt: input exceeds %d bytes", limit)
	}
	if err != nil {
		return Run{}, fmt.Errorf("benchfmt: decode run: %w", err)
	}
	var extra any
	err = d.Decode(&extra)
	if input.N == 0 {
		return Run{}, fmt.Errorf("benchfmt: input exceeds %d bytes", limit)
	}
	if err != io.EOF {
		if err == nil {
			err = fmt.Errorf("trailing JSON document")
		}
		return Run{}, fmt.Errorf("benchfmt: decode run: %w", err)
	}
	if run.Version != 1 {
		return Run{}, fmt.Errorf("benchfmt: unsupported format version %d (want 1)", run.Version)
	}
	return run, nil
}

// Write validates all samples and emits Go benchmark text. Invalid input writes
// nothing. Writer failures can leave partial output. Each original sample emits
// exactly one row; no repetitions or uncertainty estimates are manufactured.
func Write(w io.Writer, run Run) error {
	if run.Version != 1 {
		return fmt.Errorf("benchfmt: unsupported format version %d (want 1)", run.Version)
	}
	var out bytes.Buffer
	for _, key := range slices.Sorted(maps.Keys(run.Config)) {
		if !configKey(key) || !text(run.Config[key]) {
			return fmt.Errorf("benchfmt: invalid configuration %q", key)
		}
		fmt.Fprintf(&out, "%s: %s\n", key, run.Config[key])
	}
	if len(run.Samples) == 0 {
		return fmt.Errorf("benchfmt: no samples")
	}
	seen := make(map[string]bool)
	for _, s := range run.Samples {
		if !strings.HasPrefix(s.Name, "Benchmark") || len(s.Name) == len("Benchmark") || !token(s.Name) {
			return fmt.Errorf("benchfmt: invalid benchmark name %q", s.Name)
		}
		if s.Operations <= 0 {
			return fmt.Errorf("benchfmt: %s has nonpositive operation count", s.Name)
		}
		if !text(s.ID) || !text(s.Source) || !text(s.Scope) {
			return fmt.Errorf("benchfmt: %s requires sample_id, source, and scope", s.Name)
		}
		if s.Scope != "full" && s.Scope != "isolated" && s.Scope != "attributed" {
			return fmt.Errorf("benchfmt: invalid scope %q", s.Scope)
		}
		for k, v := range s.Provenance {
			if !text(k) || !text(v) {
				return fmt.Errorf("benchfmt: invalid provenance for %s", s.Name)
			}
		}
		identity := s.Source + "\x00" + s.ID + "\x00" + s.Name
		if seen[identity] {
			return fmt.Errorf("benchfmt: repeated source sample %q for %s", s.ID, s.Name)
		}
		seen[identity] = true
		if len(s.Measurements) == 0 {
			return fmt.Errorf("benchfmt: %s has no measurements", s.Name)
		}
		units := make(map[string]bool)
		var row strings.Builder
		fmt.Fprintf(&row, "%s %d", s.Name, s.Operations)
		for _, m := range s.Measurements {
			if units[m.Unit] {
				return fmt.Errorf("benchfmt: duplicate unit %q", m.Unit)
			}
			units[m.Unit] = true
			value, err := measurementValue(s, m)
			if err != nil {
				return fmt.Errorf("benchfmt: %s: %w", s.Name, err)
			}
			fmt.Fprintf(&row, " %s %s", strconv.FormatFloat(value, 'g', -1, 64), m.Unit)
		}
		// A JSON comment retains original values and provenance without introducing
		// file-level keys that would split C and Go into different benchstat tables.
		data, err := json.Marshal(s)
		if err != nil {
			return fmt.Errorf("benchfmt: provenance: %w", err)
		}
		fmt.Fprintf(&out, "# cuptibench.sample %s\n%s\n", data, row.String())
	}
	_, err := out.WriteTo(w)
	return err
}

func measurementValue(s Sample, m Measurement) (float64, error) {
	if !token(m.Unit) || m.Value == nil {
		return 0, fmt.Errorf("invalid unit or missing measurement value")
	}
	value := *m.Value
	if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
		return 0, fmt.Errorf("invalid value for %s", m.Unit)
	}
	timing := false
	switch m.Kind {
	case "host-elapsed":
		if m.Unit != "ns/op" {
			return 0, fmt.Errorf("host elapsed time requires ns/op")
		}
		timing = true
	case "device-elapsed":
		if m.Unit != "gpu-ns/op" {
			return 0, fmt.Errorf("device elapsed time requires gpu-ns/op")
		}
		timing = true
	case "kernel-duration-sum":
		if m.Unit != "kernel-sum-ns/op" {
			return 0, fmt.Errorf("kernel duration sum requires kernel-sum-ns/op")
		}
		timing = true
	case "counter", "rate", "ratio":
		if m.Unit == "ns/op" || m.Unit == "gpu-ns/op" || m.Unit == "kernel-sum-ns/op" {
			return 0, fmt.Errorf("reserved timing unit %s", m.Unit)
		}
	default:
		return 0, fmt.Errorf("unknown measurement kind %q", m.Kind)
	}
	if timing && value == 0 {
		return 0, fmt.Errorf("unresolved zero timing for %s", m.Unit)
	}
	switch m.Basis {
	case "total":
		if nonAdditiveUnit(m.Unit) {
			return 0, fmt.Errorf("rate or ratio unit cannot be normalized per operation")
		}
		if m.Kind == "rate" || m.Kind == "ratio" || !strings.HasSuffix(m.Unit, "/op") {
			return 0, fmt.Errorf("total requires additive semantics and /op unit")
		}
		value /= float64(s.Operations)
	case "per-operation":
		if nonAdditiveUnit(m.Unit) {
			return 0, fmt.Errorf("rate or ratio unit cannot be normalized per operation")
		}
		if m.Kind == "rate" || m.Kind == "ratio" || !strings.HasSuffix(m.Unit, "/op") {
			return 0, fmt.Errorf("per-operation requires additive semantics and /op unit")
		}
	case "raw":
		if timing || strings.HasSuffix(m.Unit, "/op") {
			return 0, fmt.Errorf("raw measurement cannot claim per-operation timing or counters")
		}
	default:
		return 0, fmt.Errorf("unknown measurement basis %q", m.Basis)
	}
	if *m.Value > 0 && value == 0 {
		return 0, fmt.Errorf("measurement normalization underflow")
	}
	return value, nil
}

func token(s string) bool { return text(s) && strings.IndexFunc(s, unicode.IsSpace) < 0 }
func text(s string) bool {
	return strings.TrimSpace(s) != "" && strings.IndexFunc(s, unicode.IsControl) < 0
}
func configKey(s string) bool {
	if s == "" || s[0] < 'a' || s[0] > 'z' {
		return false
	}
	for _, r := range s {
		if !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-') {
			return false
		}
	}
	return true
}

// This catches recognizable mislabeling without inferring additivity for other
// units. The producer still has to substantiate its declared measurement kind.
func nonAdditiveUnit(unit string) bool {
	u := strings.ToLower(strings.TrimSuffix(unit, "/op"))
	return u == "%" || u == "pct" || u == "percent" || u == "ratio" ||
		strings.HasSuffix(u, "/s") || strings.HasSuffix(u, "-ratio") ||
		strings.Contains(u, ".pct") || strings.HasPrefix(u, "pct_")
}
