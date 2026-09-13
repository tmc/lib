package cuptipcsampling_test

import (
	"bytes"
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

type mockResolver struct {
	locations map[string]cuptipcsampling.CodeLocation
}

func (m *mockResolver) ResolvePC(mod cuptipcsampling.ModuleID, kernelName string, pcOffset uint64) cuptipcsampling.CodeLocation {
	if loc, ok := m.locations[kernelName]; ok {
		loc.CUBINOffset += pcOffset
		loc.Resolved = true
		return loc
	}
	return cuptipcsampling.CodeLocation{
		KernelName: kernelName,
	}
}

func TestProfileBridge(t *testing.T) {
	// Modern verbatim names from CUPTI 13 native reason inventory
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 130001,
		Reasons: []cuptipcsampling.StallReason{
			{Index: 1, Name: "selected"},
			{Index: 2, Name: "membar"},
			{Index: 3, Name: "short_scoreboard"},
			{Index: 4, Name: "math_pipe_throttle"},
			{Index: 5, Name: "lg_throttle"},
			{Index: 6, Name: "sleeping"},
			{Index: 7, Name: "selected_not_issued"},
		},
	}

	resolver := &mockResolver{
		locations: map[string]cuptipcsampling.CodeLocation{
			"ds4_iq2_simd": {
				KernelName:  "ds4_iq2_simd",
				SourceFile:  "cuda_experts_q81.cu",
				SourceLine:  42,
				PTXFile:     "cuda_experts_q81.ptx",
				PTXLine:     105,
				SASSFile:    "cuda_experts_q81.sass",
				SASSLine:    30,
				CUBINOffset: 0x1000,
			},
		},
	}

	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)
	session.SetMappingFile("kernels.cubin", "")

	bridge := &cuptipcsampling.ProfileBridge{
		Session:   session,
		Inventory: inv,
		Resolver:  resolver,
	}

	// 42 = the six base reasons; the companion at index 7 counts a subset of
	// the cycles index 1 already counted, which is why CUPTI's own total for a
	// batch matches the base reasons alone.
	batch := cuptipcsampling.Batch{
		Valid:        true,
		TotalSamples: 42,
		PCs: []cuptipcsampling.PC{
			{
				CRC:    0x12345678,
				Name:   "ds4_iq2_simd",
				Offset: 0x20,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 1, Samples: 10}, // selected -> InstExecuted
					{Index: 2, Samples: 8},  // membar -> StallWait (must not be swallowed by mem)
					{Index: 3, Samples: 6},  // short_scoreboard -> StallScoreboard
					{Index: 4, Samples: 7},  // math_pipe_throttle -> StallMath
					{Index: 5, Samples: 9},  // lg_throttle -> StallMem
					{Index: 6, Samples: 2},  // sleeping -> UnmappedStalls
					{Index: 7, Samples: 4},  // selected_not_issued -> discarded, neither issued nor unmapped
				},
			},
		},
	}

	ctx := context.Background()
	if err := bridge.RecordBatch(ctx, batch); err != nil {
		t.Fatalf("RecordBatch: %v", err)
	}

	// Verify unmapped stall reasons are preserved and not silently discarded
	if got := bridge.UnmappedStalls["sleeping"]; got != 2 {
		t.Errorf("UnmappedStalls[\"sleeping\"] = %d, want 2", got)
	}
	// The companion of selected is a subset of cycles selected already counted,
	// so it adds to neither inst_executed nor the unmapped tally: a reason this
	// package models must not be reported as one it failed to model.
	if got, ok := bridge.UnmappedStalls["selected_not_issued"]; ok {
		t.Errorf("UnmappedStalls[\"selected_not_issued\"] = %d, want absent", got)
	}

	if err := session.Close(); err != nil {
		t.Fatalf("session.Close: %v", err)
	}

	// Verify profile output can be parsed and contains expected structure
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}

	if len(p.Sample) != 1 {
		t.Fatalf("got %d samples, want 1", len(p.Sample))
	}

	sampleTypes := make(map[string]int)
	for i, st := range p.SampleType {
		sampleTypes[st.Type] = i
	}
	// PC samples land in pc_samples, not samples: the latter counts launches,
	// and one type cannot mean both without meaning neither.
	if idx, ok := sampleTypes["pc_samples"]; ok {
		if got := p.Sample[0].Value[idx]; got != 42 {
			t.Errorf("pc_samples = %d, want 42 (base reasons only)", got)
		}
	} else {
		t.Error("missing pc_samples type in profile")
	}
	if idx, ok := sampleTypes["inst_executed"]; ok {
		if got := p.Sample[0].Value[idx]; got != 10 {
			t.Errorf("inst_executed = %d, want 10 (selected_not_issued must not contribute)", got)
		}
	} else {
		t.Error("missing inst_executed type in profile")
	}

	// SASS Address should be DefaultMappingStart + 0x1000 + 0x20 = 0x7f001020
	leafLoc := p.Location[0]
	wantAddr := cudaprof.DefaultMappingStart + 0x1020
	if leafLoc.Address != wantAddr {
		t.Errorf("leaf location address = %#x, want %#x", leafLoc.Address, wantAddr)
	}

	// Verify line numbers in stack frames
	if len(leafLoc.Line) == 0 {
		t.Fatal("leaf location missing lines")
	}
	if gotLine := leafLoc.Line[0].Line; gotLine != 30 {
		t.Errorf("SASS line = %d, want 30", gotLine)
	}
}

func TestProfileBridgeErrors(t *testing.T) {
	bridge := &cuptipcsampling.ProfileBridge{}
	ctx := context.Background()

	// Nil session
	if err := bridge.RecordBatch(ctx, cuptipcsampling.Batch{Valid: true}); err == nil {
		t.Error("expected error for nil session, got nil")
	}

	// Invalid batch
	var buf bytes.Buffer
	bridge.Session = cudaprof.NewSession(&buf)
	if err := bridge.RecordBatch(ctx, cuptipcsampling.Batch{Valid: false}); err == nil {
		t.Error("expected error for invalid batch, got nil")
	}
}

// CUPTI reports each stall reason alongside a _not_issued companion counting a
// subset of the same cycles, so adding both counts those samples twice. On a
// GB10 capture that inflated the stall totals to about 1.9x the samples
// actually collected. The companion of a reason this package does not model
// still has to stay visible as unmapped.
func TestProfileBridgeDoesNotDoubleCountNotIssued(t *testing.T) {
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 130001,
		Reasons: []cuptipcsampling.StallReason{
			{Index: 1, Name: "short_scoreboard"},
			{Index: 2, Name: "short_scoreboard_not_issued"},
			{Index: 3, Name: "sleeping"},
			{Index: 4, Name: "sleeping_not_issued"},
		},
	}
	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)
	session.SetMappingFile("kernels.cubin", "")
	bridge := &cuptipcsampling.ProfileBridge{Session: session, Inventory: inv}
	batch := cuptipcsampling.Batch{
		Valid:        true,
		TotalSamples: 23,
		PCs: []cuptipcsampling.PC{{
			CRC:  1,
			Name: "kernel",
			Reasons: []cuptipcsampling.StallSamples{
				{Index: 1, Samples: 23},
				{Index: 2, Samples: 23},
				{Index: 3, Samples: 4},
				{Index: 4, Samples: 4},
			},
		}},
	}
	if err := bridge.RecordBatch(context.Background(), batch); err != nil {
		t.Fatalf("RecordBatch: %v", err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("session.Close: %v", err)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}
	types := make(map[string]int)
	for i, st := range p.SampleType {
		types[st.Type] = i
	}
	if got := p.Sample[0].Value[types["stall_scoreboard"]]; got != 23 {
		t.Errorf("stall_scoreboard = %d, want 23 counted once, not 46", got)
	}
	// sleeping is not modelled, so the base reason is unmapped under its own
	// name and the companion under its own, neither folded into a stall bucket.
	if got := bridge.UnmappedStalls["sleeping"]; got != 4 {
		t.Errorf("UnmappedStalls[sleeping] = %d, want 4", got)
	}
	if got := bridge.UnmappedStalls["sleeping_not_issued"]; got != 4 {
		t.Errorf("UnmappedStalls[sleeping_not_issued] = %d, want 4", got)
	}
}

// A PTX frame with no address leaves pprof's address granularity blind to the
// only samples that have real instruction addresses.
func TestProfileBridgePTXAddress(t *testing.T) {
	ptx := "// line 1\n// line 2\n// line 3\n"
	offsets := cudaprof.PTXLineOffsets(ptx)
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 130001,
		Reasons:      []cuptipcsampling.StallReason{{Index: 1, Name: "short_scoreboard"}},
	}
	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)
	session.SetMappingFile("kernels.cubin", "")
	session.AddMapping("kernels.ptx", "", cudaprof.PTXMappingStart, cudaprof.PTXMappingStart+uint64(len(ptx)))
	bridge := &cuptipcsampling.ProfileBridge{
		Session:    session,
		Inventory:  inv,
		PTXOffsets: offsets,
		Resolver: &mockResolver{locations: map[string]cuptipcsampling.CodeLocation{
			"kernel": {KernelName: "kernel", PTXFile: "kernels.ptx", PTXLine: 3, SASSLine: 1},
		}},
	}
	batch := cuptipcsampling.Batch{
		Valid:        true,
		TotalSamples: 1,
		PCs:          []cuptipcsampling.PC{{CRC: 1, Name: "kernel", Reasons: []cuptipcsampling.StallSamples{{Index: 1, Samples: 1}}}},
	}
	if err := bridge.RecordBatch(context.Background(), batch); err != nil {
		t.Fatalf("RecordBatch: %v", err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("session.Close: %v", err)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}
	want := cudaprof.PTXMappingStart + offsets[3]
	var found bool
	for _, loc := range p.Location {
		for _, line := range loc.Line {
			if line.Function != nil && line.Function.Name == "kernel.PTX" {
				found = true
				if loc.Address != want {
					t.Errorf("PTX address = %#x, want %#x (line 3 at byte %d)", loc.Address, want, offsets[3])
				}
			}
		}
	}
	if !found {
		t.Fatal("no PTX frame in the profile")
	}
}

func TestProfileBridgeLabelsAndHostFrames(t *testing.T) {
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 130001,
		Reasons:      []cuptipcsampling.StallReason{{Index: 1, Name: "short_scoreboard"}},
	}
	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)
	session.SetMappingFile("kernels.cubin", "")
	bridge := &cuptipcsampling.ProfileBridge{
		Session:   session,
		Inventory: inv,
		Resolver: &mockResolver{locations: map[string]cuptipcsampling.CodeLocation{
			"test_kernel": {KernelName: "test_kernel", SourceFile: "kernel.cu", SourceLine: 10, SASSLine: 1},
		}},
		HostFrames: func(loc cuptipcsampling.CodeLocation) []cudaprof.Frame {
			if loc.KernelName != "test_kernel" {
				return nil
			}
			return []cudaprof.Frame{
				{FunctionName: "main.runWorkload", FileName: "workload.go", Line: 50},
				{FunctionName: "main.main", FileName: "main.go", Line: 100},
			}
		},
		LabelFunc: func(loc cuptipcsampling.CodeLocation) map[string]string {
			return map[string]string{"phase": "compute"}
		},
	}
	batch := cuptipcsampling.Batch{
		Valid:        true,
		TotalSamples: 1,
		PCs:          []cuptipcsampling.PC{{CRC: 1, Name: "test_kernel", Reasons: []cuptipcsampling.StallSamples{{Index: 1, Samples: 1}}}},
	}
	if err := bridge.RecordBatch(context.Background(), batch); err != nil {
		t.Fatalf("RecordBatch: %v", err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("session.Close: %v", err)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}
	if len(p.Sample) != 1 {
		t.Fatalf("got %d samples, want 1", len(p.Sample))
	}
	s := p.Sample[0]
	if len(s.Label["kernel"]) != 1 || s.Label["kernel"][0] != "test_kernel" {
		t.Errorf("kernel label = %v, want [test_kernel]", s.Label["kernel"])
	}
	if len(s.Label["phase"]) != 1 || s.Label["phase"][0] != "compute" {
		t.Errorf("phase label = %v, want [compute]", s.Label["phase"])
	}
	// Verify host frames are present
	var hasRunWorkload, hasMain bool
	for _, loc := range s.Location {
		for _, line := range loc.Line {
			if line.Function != nil {
				if line.Function.Name == "main.runWorkload" {
					hasRunWorkload = true
				}
				if line.Function.Name == "main.main" {
					hasMain = true
				}
			}
		}
	}
	if !hasRunWorkload || !hasMain {
		t.Errorf("host frames missing: hasRunWorkload=%v, hasMain=%v", hasRunWorkload, hasMain)
	}
}

// A native inventory names every reason in full, and the selected reason is
// issued work rather than a stall. Reported under its native name it was
// landing in the unmapped tally, where a manifest calls it an unmapped stall
// and nothing counts the warps that did issue.
func TestProfileBridgeNativeReasonNames(t *testing.T) {
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 130001,
		Reasons: []cuptipcsampling.StallReason{
			{Index: 1, Name: "smsp__pcsamp_warps_issue_stalled_selected"},
			{Index: 2, Name: "smsp__pcsamp_warps_issue_stalled_not_selected"},
			{Index: 3, Name: "smsp__pcsamp_warps_issue_stalled_short_scoreboard"},
			{Index: 4, Name: "smsp__pcsamp_warps_issue_stalled_short_scoreboard_not_issued"},
			{Index: 5, Name: "smsp__pcsamp_warps_issue_stalled_selected_not_issued"},
		},
	}
	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)
	session.SetMappingFile("kernels.cubin", "")
	bridge := &cuptipcsampling.ProfileBridge{Session: session, Inventory: inv}
	batch := cuptipcsampling.Batch{
		Valid:        true,
		TotalSamples: 226,
		PCs: []cuptipcsampling.PC{{
			CRC:  1,
			Name: "kernel",
			Reasons: []cuptipcsampling.StallSamples{
				{Index: 1, Samples: 206},
				{Index: 2, Samples: 18},
				{Index: 3, Samples: 2},
				{Index: 4, Samples: 2},
				{Index: 5, Samples: 5},
			},
		}},
	}
	if err := bridge.RecordBatch(context.Background(), batch); err != nil {
		t.Fatalf("RecordBatch: %v", err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("session.Close: %v", err)
	}
	if got, ok := bridge.UnmappedStalls["smsp__pcsamp_warps_issue_stalled_selected"]; ok {
		t.Errorf("selected counted as an unmapped stall (%d)", got)
	}
	// classifyStall has no selected case, so the companion of selected reaches
	// the unmapped tally unless the suffix comes off before the test. The
	// identity still closes either way, which is what makes it easy to miss:
	// the only symptom is a modelled reason reported as unmodelled.
	if got, ok := bridge.UnmappedStalls["smsp__pcsamp_warps_issue_stalled_selected_not_issued"]; ok {
		t.Errorf("selected_not_issued counted as an unmapped stall (%d)", got)
	}
	// not_selected has its own type, and the suffix test must not mistake it
	// for the selected reason: 18 warps that were ready while another issued
	// are not 18 warps that issued.
	if got, ok := bridge.UnmappedStalls["smsp__pcsamp_warps_issue_stalled_not_selected"]; ok {
		t.Errorf("not_selected counted as an unmapped stall (%d)", got)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}
	types := make(map[string]int)
	for i, st := range p.SampleType {
		types[st.Type] = i
	}
	if got := p.Sample[0].Value[types["inst_executed"]]; got != 206 {
		t.Errorf("inst_executed = %d, want 206", got)
	}
	if got := p.Sample[0].Value[types["stall_not_selected"]]; got != 18 {
		t.Errorf("stall_not_selected = %d, want 18", got)
	}
	// The batch total CUPTI reports is the base reasons; the profile's own
	// count has to agree with it or the manifest beside it reads 1.9x off.
	if got := p.Sample[0].Value[types["pc_samples"]]; got != int64(batch.TotalSamples) {
		t.Errorf("pc_samples = %d, want %d, the batch total", got, batch.TotalSamples)
	}
}

// A kernel whose launch stack is unknown gets no host frames rather than
// another kernel's, because HostFrames is asked per location.
func TestProfileBridgeHostFramesPerKernel(t *testing.T) {
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 130001,
		Reasons:      []cuptipcsampling.StallReason{{Index: 1, Name: "short_scoreboard"}},
	}
	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)
	session.SetMappingFile("kernels.cubin", "")
	stacks := map[string][]cudaprof.Frame{
		"known": {{FunctionName: "main.launchKnown", FileName: "workload.go", Line: 7}},
	}
	var asked []string
	bridge := &cuptipcsampling.ProfileBridge{
		Session:   session,
		Inventory: inv,
		HostFrames: func(loc cuptipcsampling.CodeLocation) []cudaprof.Frame {
			asked = append(asked, loc.KernelName)
			return stacks[loc.KernelName]
		},
	}
	batch := cuptipcsampling.Batch{
		Valid:        true,
		TotalSamples: 2,
		PCs: []cuptipcsampling.PC{
			{CRC: 1, Name: "known", Reasons: []cuptipcsampling.StallSamples{{Index: 1, Samples: 1}}},
			{CRC: 1, Name: "unknown", Reasons: []cuptipcsampling.StallSamples{{Index: 1, Samples: 1}}},
		},
	}
	if err := bridge.RecordBatch(context.Background(), batch); err != nil {
		t.Fatalf("RecordBatch: %v", err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("session.Close: %v", err)
	}
	if !slices.Equal(asked, []string{"known", "unknown"}) {
		t.Errorf("HostFrames asked for %v, want one call per PC", asked)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}
	depth := make(map[string]int)
	for _, s := range p.Sample {
		var kernel string
		for _, loc := range s.Location {
			for _, line := range loc.Line {
				if line.Function != nil && strings.HasSuffix(line.Function.Name, ".SASS") {
					kernel = strings.TrimSuffix(line.Function.Name, ".SASS")
				}
			}
		}
		depth[kernel] = len(s.Location)
	}
	if depth["known"] != depth["unknown"]+1 {
		t.Errorf("stack depths known=%d unknown=%d, want the known kernel to carry one more frame", depth["known"], depth["unknown"])
	}
}

// Every reason CUPTI collected has to land in a sample type or the profile
// shows less than the hardware measured: a reason left unmodelled appears only
// in the manifest's unmapped tally, where no pprof view can reach it. On a GB10
// capture that was 172 of 2765 samples across five reasons.
func TestProfileBridgeFrontEndAndIssueReasons(t *testing.T) {
	for _, tt := range []struct {
		name   string
		reason string
		want   string
	}{
		{"branch", "smsp__pcsamp_warps_issue_stalled_branch_resolving", "stall_branch"},
		{"fetch", "smsp__pcsamp_warps_issue_stalled_no_instructions", "stall_fetch"},
		{"drain", "smsp__pcsamp_warps_issue_stalled_drain", "stall_drain"},
		{"dispatch", "smsp__pcsamp_warps_issue_stalled_dispatch_stall", "stall_dispatch"},
		{"not_selected", "smsp__pcsamp_warps_issue_stalled_not_selected", "stall_not_selected"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			inv := cuptipcsampling.ReasonInventory{
				CUPTIVersion: 130001,
				Reasons:      []cuptipcsampling.StallReason{{Index: 1, Name: tt.reason}},
			}
			var buf bytes.Buffer
			session := cudaprof.NewSession(&buf)
			session.SetMappingFile("kernels.cubin", "")
			bridge := &cuptipcsampling.ProfileBridge{Session: session, Inventory: inv}
			batch := cuptipcsampling.Batch{
				Valid:        true,
				TotalSamples: 5,
				PCs: []cuptipcsampling.PC{{
					CRC:     1,
					Name:    "kernel",
					Reasons: []cuptipcsampling.StallSamples{{Index: 1, Samples: 5}},
				}},
			}
			if err := bridge.RecordBatch(context.Background(), batch); err != nil {
				t.Fatalf("RecordBatch: %v", err)
			}
			if err := session.Close(); err != nil {
				t.Fatalf("session.Close: %v", err)
			}
			if len(bridge.UnmappedStalls) != 0 {
				t.Errorf("UnmappedStalls = %v, want empty", bridge.UnmappedStalls)
			}
			p, err := profile.ParseData(buf.Bytes())
			if err != nil {
				t.Fatalf("ParseData: %v", err)
			}
			for i, st := range p.SampleType {
				got := p.Sample[0].Value[i]
				switch {
				case st.Type == tt.want && got != 5:
					t.Errorf("%s = %d, want 5", st.Type, got)
				case st.Type != tt.want && st.Type != "pc_samples" && got != 0:
					t.Errorf("%s = %d, want 0", st.Type, got)
				}
			}
		})
	}
}

// Routing is the audit the identity cannot perform: a fetch stall counted as a
// memory stall leaves every sum closed and sends a reader to the wrong hardware
// unit. The table is the full reason set a GB10 inventory reports, including the
// reasons this workload never sampled.
func TestReasonInventoryRouting(t *testing.T) {
	const prefix = "smsp__pcsamp_warps_issue_stalled_"
	want := map[string]string{
		"selected":           "inst_executed",
		"not_selected":       "stall_not_selected",
		"long_scoreboard":    "stall_scoreboard",
		"short_scoreboard":   "stall_scoreboard",
		"wait":               "stall_wait",
		"barrier":            "stall_wait",
		"membar":             "stall_wait",
		"math_pipe_throttle": "stall_math",
		"mio_throttle":       "stall_mem",
		"lg_throttle":        "stall_mem",
		"tex_throttle":       "stall_mem",
		"imc_miss":           "stall_mem",
		"drain":              "stall_drain",
		"dispatch_stall":     "stall_dispatch",
		"branch_resolving":   "stall_branch",
		"no_instructions":    "stall_fetch",
		"misc":               "unmapped",
		"sleeping":           "unmapped",
	}
	var inv cuptipcsampling.ReasonInventory
	for reason := range want {
		inv.Reasons = append(inv.Reasons,
			cuptipcsampling.StallReason{Name: prefix + reason},
			cuptipcsampling.StallReason{Name: prefix + reason + "_not_issued"})
	}
	routing := inv.Routing()
	for reason, wantType := range want {
		if got := routing[prefix+reason]; got != wantType {
			t.Errorf("Routing[%s] = %q, want %q", reason, got, wantType)
		}
		// A companion is a subset of the cycles its base reason already
		// counted, so it is discarded rather than added anywhere -- unless the
		// base reason is unmodelled, in which case both stay visible as
		// unmapped instead of vanishing.
		wantCompanion := "companion"
		if wantType == "unmapped" {
			wantCompanion = "unmapped"
		}
		if got := routing[prefix+reason+"_not_issued"]; got != wantCompanion {
			t.Errorf("Routing[%s_not_issued] = %q, want %q", reason, got, wantCompanion)
		}
	}
}
