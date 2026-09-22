package cudaprof_test

import (
	"encoding/json"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func TestSamplingProvenanceRetention(t *testing.T) {
	raw := validProfileSnapshot()
	raw.Sampling = &cudaprof.SamplingProvenance{RunID: "run", ModuleGeneration: 7, ModuleCRC: 12, KernelName: "kernel", FunctionIndex: 3, CorrelationID: 4, RangeID: 5, CUPTIVersion: 130001, CollectionMode: 2, TotalSamples: 100, NonUserSamples: 10}
	raw.Losses = 9
	s, err := cudaprof.NewProfileSnapshot(raw)
	if err != nil {
		t.Fatal(err)
	}
	raw.Sampling.ModuleGeneration = 99
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	var decoded cudaprof.ProfileSnapshot
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	g, err := decoded.Guidance(profileTarget(&decoded, "work", "", "device"))
	if err != nil {
		t.Fatal(err)
	}
	if g.Sampling == nil || *g.Sampling != *s.Sampling || g.Sampling.ModuleGeneration != 7 {
		t.Fatalf("lost native provenance: %+v", g.Sampling)
	}
	g.Sampling.RunID = "changed"
	if decoded.Sampling.RunID != "run" {
		t.Fatal("guidance aliases provenance")
	}
}

func TestSamplingProvenanceRefusals(t *testing.T) {
	for _, kind := range []string{"run", "generation", "kernel", "version", "mode", "nonuser", "losses"} {
		t.Run(kind, func(t *testing.T) {
			s := validProfileSnapshot()
			s.Sampling = &cudaprof.SamplingProvenance{RunID: "run", ModuleGeneration: 1, KernelName: "k", CUPTIVersion: 130001, CollectionMode: 2, TotalSamples: 10}
			switch kind {
			case "run":
				s.Sampling.RunID = ""
			case "generation":
				s.Sampling.ModuleGeneration = 0
			case "kernel":
				s.Sampling.KernelName = ""
			case "version":
				s.Sampling.CUPTIVersion = 0
			case "mode":
				s.Sampling.CollectionMode = 99
			case "nonuser":
				s.Sampling.NonUserSamples = 11
			case "losses":
				s.Losses = 11
			}
			if _, err := cudaprof.NewProfileSnapshot(s); err == nil {
				t.Fatal("accepted invalid provenance")
			}
		})
	}
}
