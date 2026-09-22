package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprofagent"
)

// publishedCounts is what a target writes into a capture: counts naming the
// image they were measured against by digest.
func publishedCounts(t *testing.T, image []byte, counts ...cudaprofagent.Count) []byte {
	t.Helper()
	sum := sha256.Sum256(image)
	data, err := json.Marshal(cudaprofagent.Counts{
		Module: hex.EncodeToString(sum[:]),
		Metric: "smsp__sass_inst_executed",
		Counts: counts,
	})
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// TestProjectExecuted joins published counts to the module whose image they
// name. Two builds of one kernel are two images, and the counts of one are
// not charged to the other.
func TestProjectExecuted(t *testing.T) {
	first, second := []byte("CUBIN-FIRST"), []byte("CUBIN-SECOND")
	c := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample:     []*profile.Sample{{Value: []int64{100}}},
		},
		Modules: []cudacapture.Module{
			{ID: "module-0000-id22", DriverID: 22, Code: cudacapture.CodeFiles{CUBIN: "module-0000-id22.cubin"}},
			{ID: "module-0001-id23", DriverID: 23, Code: cudacapture.CodeFiles{CUBIN: "module-0001-id23.cubin"}},
		},
	}
	attachments := map[string][]byte{
		"module-0000-id22.cubin": first,
		"module-0001-id23.cubin": second,
		"count-fused_gate.json":  publishedCounts(t, second, cudaprofagent.Count{Function: "fused_gate", Offset: 0x10, Value: 4096}),
		// Counts of an image this capture never saw bind to no module.
		"count-other.json": publishedCounts(t, []byte("CUBIN-ELSEWHERE"), cudaprofagent.Count{Function: "other", Offset: 0, Value: 7}),
	}
	projectExecuted(c, attachments)

	last := len(c.Profile.SampleType) - 1
	if c.Profile.SampleType[last].Type != executedType {
		t.Fatalf("sample types = %v, want %s last", c.Profile.SampleType, executedType)
	}
	if got := c.Profile.Sample[0].Value; len(got) != 2 || got[1] != 0 {
		t.Errorf("the kernel-time sample gained %v in the executed dimension, want a zero", got)
	}
	if len(c.Profile.Sample) != 2 {
		t.Fatalf("got %d samples, want 2: the counts of an unknown image must be dropped", len(c.Profile.Sample))
	}
	s := c.Profile.Sample[1]
	if s.Value[last] != 4096 {
		t.Errorf("executed = %v, want 4096", s.Value)
	}
	if ids := s.Label["module_id"]; len(ids) != 1 || ids[0] != "23" {
		t.Errorf("counts were charged to module %v, want the build whose image they name", ids)
	}
	if f := deviceFrames(s); f.kernel != "fused_gate" {
		t.Errorf("frames name kernel %q, want fused_gate", f.kernel)
	}
}

// A capture nobody published counts to gains no dimension: an empty column
// of exact counts reads as a kernel that executed nothing.
func TestProjectExecutedWithoutCounts(t *testing.T) {
	c := &cudacapture.Capture{Profile: &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
	}}
	projectExecuted(c, map[string][]byte{"module-0000-id22.cubin": []byte("CUBIN")})
	if len(c.Profile.SampleType) != 1 {
		t.Errorf("sample types = %v, want the capture's own", c.Profile.SampleType)
	}
}
