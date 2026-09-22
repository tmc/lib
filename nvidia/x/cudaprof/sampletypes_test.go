package cudaprof_test

import (
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// TestBuilderSampleTypes pins what a producer reads to check its own values
// against: the names actually written, in write order, under each of the three
// ways the schema gets set.
func TestBuilderSampleTypes(t *testing.T) {
	tests := []struct {
		name  string
		setup func(*cudaprof.Builder) error
		want  []string
	}{
		{
			name:  "default",
			setup: func(b *cudaprof.Builder) error { return nil },
			want:  cudaprof.SampleTypeNames(),
		},
		{
			name: "selected",
			setup: func(b *cudaprof.Builder) error {
				return b.SetSampleTypes("gpu_time", "stall_mem")
			},
			// Selection order is the write order, not the declaration order.
			want: []string{"gpu_time", "stall_mem"},
		},
		{
			name: "declared",
			setup: func(b *cudaprof.Builder) error {
				return b.SetDeclaredSampleTypes(
					cudaprof.SampleType{Name: "tokens", Unit: "count"},
					cudaprof.SampleType{Name: "prefill", Unit: "nanoseconds"})
			},
			want: []string{"tokens", "prefill"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := cudaprof.NewBuilder()
			if err := tt.setup(b); err != nil {
				t.Fatal(err)
			}
			if got := b.SampleTypes(); !slices.Equal(got, tt.want) {
				t.Errorf("SampleTypes() = %v, want %v", got, tt.want)
			}
		})
	}
}
