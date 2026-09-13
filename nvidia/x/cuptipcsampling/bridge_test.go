package cuptipcsampling_test

import (
	"bytes"
	"context"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

type mockResolver struct {
	locations map[string]cuptipcsampling.CodeLocation
}

func (m *mockResolver) ResolvePC(kernelName string, pcOffset uint64) cuptipcsampling.CodeLocation {
	if loc, ok := m.locations[kernelName]; ok {
		loc.Offset += pcOffset
		return loc
	}
	return cuptipcsampling.CodeLocation{
		KernelName: kernelName,
		Offset:     pcOffset,
	}
}

func TestProfileBridge(t *testing.T) {
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 130001,
		Reasons: []cuptipcsampling.StallReason{
			{Index: 1, Name: "stall_memory_dependency"},
			{Index: 2, Name: "stall_math_pipe_throttle"},
			{Index: 3, Name: "stall_barrier"},
			{Index: 4, Name: "stall_scoreboard"},
		},
	}

	resolver := &mockResolver{
		locations: map[string]cuptipcsampling.CodeLocation{
			"ds4_iq2_simd": {
				KernelName: "ds4_iq2_simd",
				SourceFile: "cuda_experts_q81.cu",
				SourceLine: 42,
				PTXFile:    "cuda_experts_q81.ptx",
				PTXLine:    105,
				SASSFile:   "cuda_experts_q81.sass",
				SASSLine:   30,
				Offset:     0x1000,
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

	batch := cuptipcsampling.Batch{
		Valid:        true,
		TotalSamples: 15,
		PCs: []cuptipcsampling.PC{
			{
				Name:   "ds4_iq2_simd",
				Offset: 0x20,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 1, Samples: 10}, // memory dependency -> StallMem
					{Index: 2, Samples: 5},  // math pipe throttle -> StallMath
				},
			},
		},
	}


	ctx := context.Background()
	if err := bridge.RecordBatch(ctx, batch); err != nil {
		t.Fatalf("RecordBatch: %v", err)
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
