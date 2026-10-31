package cudadiag_test

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// aggregateIdentityInput uses the public producer and clock-stamping paths.
// Rank is a caller assertion about this aggregate export, not a GPU identity.
func aggregateIdentityInput(t *testing.T, beforeRank, afterRank *int) cudadiag.ServingInput {
	t.Helper()
	c := newServingCase()
	c.movement = nil
	c.window = false
	in := c.input(t)
	read := func(name string, rank *int, at int64) *nvtelemetry.WorkloadExport {
		t.Helper()
		raw, err := os.ReadFile("../nvtelemetry/testdata/workload/" + name)
		if err != nil {
			t.Fatal(err)
		}
		source := nvtelemetry.WorkloadSource{Provider: "triton", Version: "25.06", RunID: "run", Phase: "decode", Rank: rank, Units: map[string]string{"supplied": "count"}}
		export, err := nvtelemetry.ReadServingExposition(context.Background(), bytes.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
		if err != nil {
			t.Fatal(err)
		}
		export, err = nvtelemetry.StampWorkloadSnapshot(export, at, "test_monotonic", "test_epoch")
		if err != nil {
			t.Fatal(err)
		}
		return &export
	}
	in.Before = read("triton-before.prom", beforeRank, 0)
	in.After = read("triton-after.prom", afterRank, 100)
	in.WindowStartNanos, in.WindowEndNanos = 0, 100
	in.WindowClockDomain, in.WindowClockEpoch = "test_monotonic", "test_epoch"
	return in
}

// aggregateIdentityWire decodes through a test-local wire view so this
// regression also compiles against the predecessor, which omits the field.
func aggregateIdentityWire(t *testing.T, summary cudadiag.ServingSummary) ([]byte, *nvtelemetry.WorkloadSource) {
	t.Helper()
	raw, err := json.Marshal(summary)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		AggregateSource *nvtelemetry.WorkloadSource `json:"aggregate_source"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	return raw, wire.AggregateSource
}

func TestServingAggregateIdentity(t *testing.T) {
	for _, rank := range []int{0, 1} {
		name := "rank-zero"
		if rank == 1 {
			name = "rank-one"
		}
		t.Run(name, func(t *testing.T) {
			in := aggregateIdentityInput(t, &rank, &rank)
			summary, err := cudadiag.SummarizeServing(context.Background(), in)
			if err != nil {
				t.Fatal(err)
			}
			_, source := aggregateIdentityWire(t, summary)
			if source == nil || source.Rank == nil || *source.Rank != rank || !reflect.DeepEqual(*source, in.Before.Source) {
				t.Fatalf("aggregate source = %+v, want supplied rank %d and source %+v", source, rank, in.Before.Source)
			}
			if summary.Source.Provider != "application" || summary.Source.Rank != nil || len(summary.Aggregate) == 0 || len(summary.AggregateRaw) != 2 {
				t.Fatalf("aggregate provenance changed application identity or results: %+v", summary)
			}
		})
	}
	t.Run("unknown-rank", func(t *testing.T) {
		summary, err := cudadiag.SummarizeServing(context.Background(), aggregateIdentityInput(t, nil, nil))
		if err != nil {
			t.Fatal(err)
		}
		_, source := aggregateIdentityWire(t, summary)
		if source == nil || source.Rank != nil {
			t.Fatalf("unknown rank became absent source or explicit zero: %+v", source)
		}
	})
	t.Run("different-ranks-distinguishable", func(t *testing.T) {
		zero, one := 0, 1
		a, err := cudadiag.SummarizeServing(context.Background(), aggregateIdentityInput(t, &zero, &zero))
		if err != nil {
			t.Fatal(err)
		}
		b, err := cudadiag.SummarizeServing(context.Background(), aggregateIdentityInput(t, &one, &one))
		if err != nil {
			t.Fatal(err)
		}
		aRaw, _ := aggregateIdentityWire(t, a)
		bRaw, _ := aggregateIdentityWire(t, b)
		if bytes.Equal(aRaw, bRaw) {
			t.Fatal("aggregate rank change disappeared from the serialized summary")
		}
		if !reflect.DeepEqual(a.Aggregate, b.Aggregate) || !reflect.DeepEqual(a.AggregateRaw, b.AggregateRaw) || !reflect.DeepEqual(a.Source, b.Source) || !reflect.DeepEqual(a.Outcomes, b.Outcomes) {
			t.Fatal("preserving rank changed arithmetic, raw bytes, application source or outcomes")
		}
	})
	t.Run("mismatched-window-refused", func(t *testing.T) {
		zero, one := 0, 1
		summary, err := cudadiag.SummarizeServing(context.Background(), aggregateIdentityInput(t, &zero, &one))
		if err == nil || !strings.Contains(err.Error(), "workload source changed across window") {
			t.Fatalf("mismatched rank = %+v, %v; want unchanged window refusal", summary, err)
		}
		_, source := aggregateIdentityWire(t, summary)
		if source != nil {
			t.Fatalf("failed window published source %+v", source)
		}
	})
	t.Run("no-window", func(t *testing.T) {
		in := aggregateIdentityInput(t, nil, nil)
		in.Before, in.After = nil, nil
		summary, err := cudadiag.SummarizeServing(context.Background(), in)
		if err != nil {
			t.Fatal(err)
		}
		_, source := aggregateIdentityWire(t, summary)
		if source != nil || len(summary.Aggregate) != 0 || len(summary.AggregateRaw) != 0 {
			t.Fatalf("missing window produced aggregate provenance: %+v", summary)
		}
	})
	t.Run("detached", func(t *testing.T) {
		rank := 0
		in := aggregateIdentityInput(t, &rank, &rank)
		summary, err := cudadiag.SummarizeServing(context.Background(), in)
		if err != nil {
			t.Fatal(err)
		}
		before, source := aggregateIdentityWire(t, summary)
		if source == nil {
			t.Fatal("missing aggregate source")
		}
		*in.Before.Source.Rank = 99
		in.Before.Source.Units["supplied"] = "changed"
		after, _ := aggregateIdentityWire(t, summary)
		if !bytes.Equal(before, after) {
			t.Fatal("caller mutation rewrote retained aggregate provenance")
		}
	})
}
