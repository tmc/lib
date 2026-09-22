package cuptipcsampling_test

import (
	"bytes"
	"context"
	"io"
	"slices"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// TestStallSampleTypes checks the published set against cudaprof's own
// declaration rather than against a second copy of the list. A list and a test
// of that list written together agree by construction and cannot notice a type
// neither mentions, which is how stall_other came to be counted here and
// absent from every capture the examples produced.
//
// The comparison runs in both directions: a stall type cudaprof carries and
// this set omits is a column callers would never select, and one this set
// names and cudaprof does not carry is a name SetSampleTypes rejects.
func TestStallSampleTypes(t *testing.T) {
	got := cuptipcsampling.StallSampleTypes()
	var want []string
	for _, name := range cudaprof.SampleTypeNames() {
		if strings.HasPrefix(name, "stall_") {
			want = append(want, name)
		}
	}
	if !slices.Equal(got, want) {
		t.Errorf("StallSampleTypes() = %v, want %v", got, want)
	}
	// The set publishes what the classifier fills and nothing else. The values
	// RecordBatch writes beside the stalls are named in the doc comment, not
	// returned, because a caller that records launches adds types this package
	// knows nothing about and one that does not must not declare them.
	for _, unwanted := range []string{"samples", "pc_samples", "inst_executed", "gpu_time", "modeled_cost"} {
		if slices.Contains(got, unwanted) {
			t.Errorf("StallSampleTypes() = %v, want no %q", got, unwanted)
		}
	}
}

// TestStallSampleTypesAreSelectable pins that the set is usable as it stands:
// SetSampleTypes rejects an unknown or duplicate name, so this fails if the
// set drifts from cudaprof's vocabulary in either way.
func TestStallSampleTypesAreSelectable(t *testing.T) {
	prof := cudaprof.NewSession(nil)
	if err := prof.SetSampleTypes(append(cuptipcsampling.StallSampleTypes(), "pc_samples", "inst_executed")...); err != nil {
		t.Fatalf("SetSampleTypes: %v", err)
	}
}

// reasonsFillingEveryType names one reason per sample type RecordBatch counts
// into, so one batch over it exercises every arm at once.
var reasonsFillingEveryType = []string{
	"smsp__pcsamp_warps_issue_stalled_selected",           // inst_executed
	"smsp__pcsamp_warps_issue_stalled_short_scoreboard",   // stall_scoreboard
	"smsp__pcsamp_warps_issue_stalled_membar",             // stall_wait
	"smsp__pcsamp_warps_issue_stalled_math_pipe_throttle", // stall_math
	"smsp__pcsamp_warps_issue_stalled_lg_throttle",        // stall_mem
	"smsp__pcsamp_warps_issue_stalled_branch_resolving",   // stall_branch
	"smsp__pcsamp_warps_issue_stalled_no_instructions",    // stall_fetch
	"smsp__pcsamp_warps_issue_stalled_drain",              // stall_drain
	"smsp__pcsamp_warps_issue_stalled_dispatch_stall",     // stall_dispatch
	"smsp__pcsamp_warps_issue_stalled_not_selected",       // stall_not_selected
	"sleeping", // stall_other, classified nowhere
}

// batchFillingEveryType returns an inventory and a batch that put samples in
// every type RecordBatch writes.
func batchFillingEveryType() (cuptipcsampling.ReasonInventory, cuptipcsampling.Batch) {
	var inv cuptipcsampling.ReasonInventory
	inv.CUPTIVersion = 130001
	pc := cuptipcsampling.PC{CRC: 1, Name: "kernel"}
	for i, name := range reasonsFillingEveryType {
		inv.Reasons = append(inv.Reasons, cuptipcsampling.StallReason{Index: uint32(i + 1), Name: name})
		pc.Reasons = append(pc.Reasons, cuptipcsampling.StallSamples{Index: uint32(i + 1), Samples: 5})
	}
	return inv, cuptipcsampling.Batch{Valid: true, TotalSamples: 55, PCs: []cuptipcsampling.PC{pc}}
}

// writtenTypes returns the sample types a profile of batch carries nonzero
// values for, read back from the encoded profile rather than from the bridge.
func writtenTypes(t *testing.T, inv cuptipcsampling.ReasonInventory, batch cuptipcsampling.Batch) []string {
	t.Helper()
	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)
	session.SetMappingFile("kernels.cubin", "")
	bridge := &cuptipcsampling.ProfileBridge{Session: session, Inventory: inv}
	if err := bridge.RecordBatch(context.Background(), batch); err != nil {
		t.Fatalf("RecordBatch: %v", err)
	}
	if got := bridge.DroppedSampleTypes(); len(got) != 0 {
		t.Errorf("DroppedSampleTypes() = %v for a session carrying every type, want none", got)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}
	var written []string
	for i, st := range p.SampleType {
		var sum int64
		for _, s := range p.Sample {
			sum += s.Value[i]
		}
		if sum != 0 {
			written = append(written, st.Type)
		}
	}
	return written
}

// droppedTypes returns the bridge's own account of what it filled, read out
// through a session carrying a type no PC sample fills so that everything
// filled is dropped.
func droppedTypes(t *testing.T, inv cuptipcsampling.ReasonInventory, batch cuptipcsampling.Batch) []string {
	t.Helper()
	prof := cudaprof.NewSession(io.Discard)
	if err := prof.SetSampleTypes("vram_alloc"); err != nil {
		t.Fatal(err)
	}
	bridge := &cuptipcsampling.ProfileBridge{Session: prof, Inventory: inv}
	if err := bridge.RecordBatch(context.Background(), batch); err != nil {
		t.Fatalf("RecordBatch: %v", err)
	}
	return bridge.DroppedSampleTypes()
}

// TestBridgeNotesAllItFills checks the name-to-field pairing in note against
// the profile itself. The bridge's account of what it filled comes from a
// hand-written list, so each reason is recorded alone and the account compared
// with the nonzero types of a profile carrying every type. One reason at a
// time is what makes the pairing testable: over a batch that fills every
// stall, a field paired with another stall's name leaves both sets unchanged.
func TestBridgeNotesAllItFills(t *testing.T) {
	var everyType []string
	for i, reason := range reasonsFillingEveryType {
		t.Run(reason, func(t *testing.T) {
			inv := cuptipcsampling.ReasonInventory{
				CUPTIVersion: 130001,
				Reasons:      []cuptipcsampling.StallReason{{Index: uint32(i + 1), Name: reason}},
			}
			batch := cuptipcsampling.Batch{
				Valid:        true,
				TotalSamples: 5,
				PCs: []cuptipcsampling.PC{{
					CRC:     1,
					Name:    "kernel",
					Reasons: []cuptipcsampling.StallSamples{{Index: uint32(i + 1), Samples: 5}},
				}},
			}
			written := writtenTypes(t, inv, batch)
			if got := droppedTypes(t, inv, batch); !slices.Equal(got, written) {
				t.Errorf("DroppedSampleTypes() = %v, profile wrote %v", got, written)
			}
			for _, name := range written {
				if !slices.Contains(everyType, name) {
					everyType = append(everyType, name)
				}
			}
		})
	}

	// The reasons above are chosen to fill every stall type, so the published
	// set has to be a subset of what they wrote. A stall type the classifier
	// can no longer reach fails here and in TestStallSampleTypes together.
	for _, name := range cuptipcsampling.StallSampleTypes() {
		if !slices.Contains(everyType, name) {
			t.Errorf("StallSampleTypes names %q, which no reason filled", name)
		}
	}
}

// TestDroppedSampleTypesNamesTheOmission is the case L4 exists for: a caller
// that selects the stalls and forgets what RecordBatch writes beside them.
func TestDroppedSampleTypesNamesTheOmission(t *testing.T) {
	inv, batch := batchFillingEveryType()
	prof := cudaprof.NewSession(io.Discard)
	if err := prof.SetSampleTypes(cuptipcsampling.StallSampleTypes()...); err != nil {
		t.Fatal(err)
	}
	bridge := &cuptipcsampling.ProfileBridge{Session: prof, Inventory: inv}
	if before := bridge.DroppedSampleTypes(); before != nil {
		t.Errorf("DroppedSampleTypes() = %v before any batch, want none", before)
	}
	if err := bridge.RecordBatch(context.Background(), batch); err != nil {
		t.Fatalf("RecordBatch: %v", err)
	}
	want := []string{"pc_samples", "inst_executed"}
	if got := bridge.DroppedSampleTypes(); !slices.Equal(got, want) {
		t.Errorf("DroppedSampleTypes() = %v, want %v", got, want)
	}
}
