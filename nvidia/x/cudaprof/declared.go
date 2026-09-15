package cudaprof

import (
	"fmt"
	"maps"
	"slices"
	"strings"
)

// SampleType declares an additive measurement. Name identifies its exact
// semantics and Unit identifies its physical unit. The producer is responsible
// for establishing additivity; ratios and rates are not additive measurements.
type SampleType struct {
	Name string
	Unit string
}

// SetDeclaredSampleTypes declares a complete custom schema, replacing the
// builtin schema. Call it before recording. Names must be unique and must not
// collide with builtin names. All recorded samples must supply every value.
func (b *Builder) SetDeclaredSampleTypes(types ...SampleType) error {
	if len(b.samples) != 0 {
		return fmt.Errorf("cudaprof: sample schema is frozen after recording")
	}
	if len(types) == 0 {
		return fmt.Errorf("cudaprof: empty declared sample schema")
	}
	seen := make(map[string]bool)
	for _, t := range types {
		if strings.TrimSpace(t.Name) == "" || strings.TrimSpace(t.Unit) == "" {
			return fmt.Errorf("cudaprof: sample name and unit are required")
		}
		if seen[t.Name] {
			return fmt.Errorf("cudaprof: duplicate sample type %q", t.Name)
		}
		for _, builtin := range profileSampleTypes {
			if t.Name == builtin.name {
				return fmt.Errorf("cudaprof: sample type %q collides with builtin", t.Name)
			}
		}
		seen[t.Name] = true
	}
	b.declaredTypes = slices.Clone(types)
	b.sampleTypes = nil
	b.defaultType = types[0].Name
	return nil
}

// AddDeclaredSample records a complete sample in declaration order. Frames are
// ordered leaf to root. Values and labels are copied. Missing values, empty
// frames, and negative counts return an error without recording a sample.
func (b *Builder) AddDeclaredSample(values []int64, frames []Frame, labels map[string]string) error {
	if len(b.declaredTypes) == 0 {
		return fmt.Errorf("cudaprof: sample types are not declared")
	}
	if len(values) != len(b.declaredTypes) {
		return fmt.Errorf("cudaprof: got %d sample values, want %d", len(values), len(b.declaredTypes))
	}
	if len(frames) == 0 {
		return fmt.Errorf("cudaprof: sample requires frames")
	}
	for _, v := range values {
		if v < 0 {
			return fmt.Errorf("cudaprof: negative additive sample value")
		}
	}
	for _, s := range b.samples {
		if !s.declared {
			return fmt.Errorf("cudaprof: mixed builtin and declared samples")
		}
	}
	b.AddStackSample(SampleValues{}, frames, maps.Clone(labels))
	s := &b.samples[len(b.samples)-1]
	s.values = slices.Clone(values)
	s.declared = true
	return nil
}
