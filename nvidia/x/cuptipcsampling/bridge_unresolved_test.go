package cuptipcsampling_test

import (
	"bytes"
	"context"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// recordOffsets records one sample per offset through a bridge whose resolver
// knows only the kernels in known, and returns the parsed profile.
func recordOffsets(t *testing.T, known map[string]cuptipcsampling.CodeLocation, kernel string, offsets ...uint64) *profile.Profile {
	t.Helper()
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 130001,
		Reasons:      []cuptipcsampling.StallReason{{Index: 2, Name: "membar"}},
	}
	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)
	session.SetMappingFile("kernels.cubin", "")
	bridge := &cuptipcsampling.ProfileBridge{
		Session:   session,
		Inventory: inv,
		Resolver:  &mockResolver{locations: known},
	}
	for _, off := range offsets {
		batch := cuptipcsampling.Batch{Valid: true, TotalSamples: 8, PCs: []cuptipcsampling.PC{{
			CRC:     0x12345678,
			Name:    kernel,
			Offset:  off,
			Reasons: []cuptipcsampling.StallSamples{{Index: 2, Samples: 8}},
		}}}
		if err := bridge.RecordBatch(context.Background(), batch); err != nil {
			t.Fatalf("RecordBatch: %v", err)
		}
	}
	if err := session.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}
	return p
}

// TestUnresolvedPCsStayDistinct pins the datum an unresolved sample would
// otherwise lose. An unresolved leaf carries address 0 and line 0, so every
// unresolved PC in a kernel is one pprof location no matter how many distinct
// addresses the hardware sampled: the sample count stays honest while the
// distinctness disappears, and nothing in the profile says it happened. The
// offset is carried as a label so the measurement survives.
func TestUnresolvedPCsStayDistinct(t *testing.T) {
	offsets := []uint64{0x20, 0x40, 0x60, 0x80}
	p := recordOffsets(t, nil, "ds4_rms", offsets...)

	// The collapse itself is not a defect to fix here: without a resolved
	// address there is no honest location to distinguish. Asserting it keeps
	// the test's premise true, so a future change that starts distinguishing
	// locations makes this fail rather than leaving the label untested.
	if len(p.Location) != 1 {
		t.Errorf("unresolved locations = %d, want 1 (the collapse this label compensates for)", len(p.Location))
	}

	got := make(map[string]int)
	for _, s := range p.Sample {
		for _, v := range s.Label["pc_offset"] {
			got[v]++
		}
	}
	want := []string{"0x20", "0x40", "0x60", "0x80"}
	if len(got) != len(want) {
		t.Fatalf("distinct pc_offset labels = %d %v, want %d: %d sampled addresses are indistinguishable in the profile",
			len(got), got, len(want), len(offsets))
	}
	for _, w := range want {
		if got[w] != 1 {
			t.Errorf("pc_offset %q appears %d times, want 1", w, got[w])
		}
	}
}

// TestResolvedPCsCarryNoOffsetLabel is the other half of the pair. A resolved
// sample's leaf address already locates the PC, so a pc_offset label would be a
// second spelling of the same fact and an unbounded label on the hot path.
// Without this arm, a change that labelled every sample would pass the test
// above while doubling the label cardinality of a working capture.
func TestResolvedPCsCarryNoOffsetLabel(t *testing.T) {
	known := map[string]cuptipcsampling.CodeLocation{"ds4_rms": {
		KernelName:  "ds4_rms",
		SourceFile:  "cuda_probe.cu",
		SourceLine:  516,
		SASSFile:    "ds4_math.sass",
		SASSLine:    30,
		CUBINOffset: 0x1000,
	}}
	p := recordOffsets(t, known, "ds4_rms", 0x20, 0x40, 0x60, 0x80)

	// Count LEAF locations, not every location: a resolved sample also carries
	// a CUDA source frame, and all four samples share the one source line, so
	// the profile holds five locations for four addresses.
	leaves := make(map[uint64]bool)
	for _, s := range p.Sample {
		if len(s.Location) == 0 {
			t.Fatal("sample with no location")
		}
		leaves[s.Location[0].Address] = true
	}
	if len(leaves) != 4 {
		t.Errorf("distinct resolved leaf addresses = %d, want 4: distinct addresses must stay distinct locations", len(leaves))
	}
	for _, s := range p.Sample {
		if v, ok := s.Label["pc_offset"]; ok {
			t.Errorf("resolved sample carries pc_offset = %v; the leaf address already locates the PC", v)
		}
	}
}
