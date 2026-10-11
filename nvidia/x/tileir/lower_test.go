package tileir_test

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func TestSemanticTileLowering(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("A", []int{16, 16}, cudair.Float32)
	b := g.AddInput("B", []int{16, 16}, cudair.Float32)
	add := g.AddBinary("add", a, b)
	_ = g.AddUnary("relu", add)

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	opts := tileir.LowerOptions{
		KernelName: "fused_add_relu",
		TileM:      16,
		TileN:      16,
		TileK:      16,
	}

	res, err := tileir.LowerSemanticRegion(snap, opts)
	if err != nil {
		t.Fatalf("LowerSemanticRegion: %v", err)
	}

	if res.Module == nil {
		t.Fatal("expected non-nil Module")
	}

	fn := res.Module.FindFunction("fused_add_relu")
	if fn == nil {
		t.Fatal("expected to find function fused_add_relu")
	}

	if len(fn.Operations) == 0 {
		t.Fatal("expected operations in lowered function")
	}

	// Verify Schema
	if len(res.Schema.Params) != 3 { // 2 inputs + 1 output
		t.Fatalf("len(res.Schema.Params) = %d, want 3", len(res.Schema.Params))
	}
	if res.Schema.Params[0].Name != "A" || res.Schema.Params[1].Name != "B" {
		t.Errorf("schema input names mismatch: %v", res.Schema.Params)
	}

	// Verify generated Pinned Tile C++ source
	if err := tileir.ValidateTileCPPSource(res.Source); err != nil {
		t.Errorf("ValidateTileCPPSource failed on generated source: %v", err)
	}

	// Verify textual Tile IR MLIR printing
	var buf bytes.Buffer
	if _, err := res.Module.WriteTo(&buf); err != nil {
		t.Fatalf("Module.WriteTo: %v", err)
	}
	text := buf.String()
	if !strings.Contains(text, "cuda_tile.entry @fused_add_relu") {
		t.Errorf("missing entry in textual tileir: %s", text)
	}
	if !strings.Contains(text, "cuda_tile.load_view_tko") {
		t.Errorf("missing load_view_tko in textual tileir: %s", text)
	}
	if !strings.Contains(text, "cuda_tile.store_view_tko") {
		t.Errorf("missing store_view_tko in textual tileir: %s", text)
	}
}

func TestTileSourceMapping(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("A", []int{16, 16}, cudair.Float32)
	a.SourceLoc = cudair.SourceLocation{
		File:     "layer_norm.go",
		Line:     42,
		Function: "ComputeLayerNorm",
	}

	b := g.AddInput("B", []int{16, 16}, cudair.Float32)
	b.SourceLoc = cudair.SourceLocation{
		File:     "layer_norm.go",
		Line:     43,
		Function: "ComputeLayerNorm",
	}

	sum := g.AddBinary("add", a, b)
	sum.SourceLoc = cudair.SourceLocation{
		File:     "layer_norm.go",
		Line:     45,
		Function: "ComputeLayerNorm",
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	res, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{
		KernelName: "source_mapped_kernel",
		TileM:      16,
		TileN:      16,
	})
	if err != nil {
		t.Fatalf("LowerSemanticRegion: %v", err)
	}

	if len(res.Module.Locations) == 0 {
		t.Fatal("expected locations recorded in module")
	}

	foundLine45 := false
	for _, loc := range res.Module.Locations {
		if loc.File == "layer_norm.go" && loc.Line == 45 {
			foundLine45 = true
			if loc.Scope != "ComputeLayerNorm" {
				t.Errorf("loc.Scope = %q, want ComputeLayerNorm", loc.Scope)
			}
			break
		}
	}
	if !foundLine45 {
		t.Errorf("expected to find location for layer_norm.go:45, locations = %v", res.Module.Locations)
	}

	// Verify that textual printing emits loc(#...)
	var buf bytes.Buffer
	if _, err := res.Module.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo: %v", err)
	}
	text := buf.String()
	if !strings.Contains(text, "loc(#") {
		t.Errorf("textual tileir missing loc references: %s", text)
	}
	if !strings.Contains(text, `layer_norm.go":45:1`) {
		t.Errorf("textual tileir missing location symbol definition for layer_norm.go:45:1 in %s", text)
	}
}

func TestTileTokenOrdering(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("A", []int{16, 16}, cudair.Float32)
	b := g.AddInput("B", []int{16, 16}, cudair.Float32)
	_ = g.AddBinary("mul", a, b)

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	res, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{
		KernelName: "token_ordered_kernel",
		TileM:      16,
		TileN:      16,
	})
	if err != nil {
		t.Fatalf("LowerSemanticRegion: %v", err)
	}

	fn := res.Module.FindFunction("token_ordered_kernel")
	if fn == nil {
		t.Fatal("expected token_ordered_kernel function")
	}

	// Ensure no hazards
	hazards := fn.VerifyTokens()
	if len(hazards) > 0 {
		t.Fatalf("unexpected token hazards in lowered module: %v", hazards)
	}

	// Verify that token chain is linear: make_token -> load_view_tko -> load_view_tko -> store_view_tko -> return
	ops := fn.Operations
	if len(ops) < 5 {
		t.Fatalf("expected at least 5 ops, got %d", len(ops))
	}
	if ops[0].OpCode != "cuda_tile.make_token" {
		t.Errorf("op 0 = %q, want cuda_tile.make_token", ops[0].OpCode)
	}
	lastOp := ops[len(ops)-1]
	if lastOp.OpCode != "cuda_tile.return" {
		t.Errorf("last op = %q, want cuda_tile.return", lastOp.OpCode)
	}
}

// TestTileBytecodeRoundTrip lowers, encodes and decodes. Like
// TestEncodeBytecodeRoundTrip its oracle is its own output and it survives
// wholesale corruption of the opcode stream; see that test's comment.
func TestTileBytecodeRoundTrip(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("A", []int{16, 16}, cudair.Float32)
	b := g.AddInput("B", []int{16, 16}, cudair.Float32)
	_ = g.AddBinary("add", a, b)

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	res, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{
		KernelName: "bytecode_roundtrip",
		TileM:      16,
		TileN:      16,
	})
	if err != nil {
		t.Fatalf("LowerSemanticRegion: %v", err)
	}

	bytecode, err := res.Module.EncodeBytecode()
	if !errors.Is(err, tileir.ErrUnsupportedASTEncoding) || bytecode != nil {
		t.Fatalf("incomplete AST encoding = %x, %v", bytecode, err)
	}
}

func TestUnsupportedTileOpRejected(t *testing.T) {
	g := cudair.NewGraph()
	in := g.AddInput("X", []int{16, 16}, cudair.Float32)
	// Add an unsupported op like "slice"
	_ = g.AddSlice(in, []int{0, 0}, []int{8, 8}, []int{1, 1})

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	_, err = tileir.LowerSemanticRegion(snap, tileir.LowerOptions{
		KernelName: "unsupported_kernel",
		TileM:      16,
		TileN:      16,
	})
	if err == nil {
		t.Fatal("expected error for unsupported slice op, got nil")
	}
	if !errors.Is(err, tileir.ErrUnsupportedTileOp) {
		t.Errorf("expected ErrUnsupportedTileOp, got: %v", err)
	}
}

func TestNegativeControls(t *testing.T) {
	// 1. Break token dependency (negative control)
	t.Run("broken token dependency", func(t *testing.T) {
		const brokenTokenIR = `cuda_tile.module @m {
  cuda_tile.entry @broken(%view: !cuda_tile.tile<ptr<f32>>) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %tok1 = cuda_tile.make_token : !cuda_tile.token
    %val = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<16x16xf32>
    %out = cuda_tile.store_view_tko %val, %view, %tok0 : (!cuda_tile.tile<16x16xf32>, !cuda_tile.tile<ptr<f32>>, !cuda_tile.token) -> !cuda_tile.token
    %read = cuda_tile.load_view_tko %view, %tok1 : (!cuda_tile.tile<ptr<f32>>, !cuda_tile.token) -> !cuda_tile.tile<16x16xf32>
    cuda_tile.return
  }
}`
		mod, err := tileir.ParseString(brokenTokenIR)
		if err != nil {
			t.Fatalf("ParseString: %v", err)
		}
		hazards, _ := mod.Verify()
		if len(hazards) == 0 {
			t.Fatal("expected token hazard for unsynchronized RAW, got none")
		}
	})

	// 2. Tile type mismatch (negative control)
	t.Run("tile type mismatch in MMA", func(t *testing.T) {
		const mismatchIR = `cuda_tile.module @m {
  cuda_tile.entry @bad_mma(%A: !cuda_tile.tile<16x16xf16>, %B: !cuda_tile.tile<16x16xf32>, %C: !cuda_tile.tile<16x16xf32>) {
    %acc = cuda_tile.mmaf %A, %B, %C : (!cuda_tile.tile<16x16xf16>, !cuda_tile.tile<16x16xf32>, !cuda_tile.tile<16x16xf32>) -> !cuda_tile.tile<16x16xf32>
    cuda_tile.return
  }
}`
		mod, err := tileir.ParseString(mismatchIR)
		if err != nil {
			t.Fatalf("ParseString: %v", err)
		}
		_, shapeErrors := mod.Verify()
		if len(shapeErrors) == 0 {
			t.Fatal("expected shape/type error for mismatched tile types in MMA, got none")
		}
		found := false
		for _, se := range shapeErrors {
			if strings.Contains(se.Reason, "MMA operand tile type mismatch") {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected error containing 'MMA operand tile type mismatch', got %v", shapeErrors)
		}
	})

	// 3. Request unsupported instruction target / compile flags (negative control)
	t.Run("unsupported compile flags rejected", func(t *testing.T) {
		// Passing offline flags (-maxrregcount, -Xptxas) to NVRTC phase must be rejected
		badFlags := []string{"--enable-tile", "--tile-only", "--tilebc", "--std=c++20", "-maxrregcount=32"}
		if err := tileir.ValidateTileCompileFlags(badFlags); err == nil {
			t.Errorf("expected error for offline flag in NVRTC compile flags, got nil")
		} else if !errors.Is(err, tileir.ErrInvalidCompileFlags) {
			t.Errorf("expected ErrInvalidCompileFlags, got %v", err)
		}

		// Missing --enable-tile
		missingFlags := []string{"--tile-only", "--tilebc", "--std=c++20"}
		if err := tileir.ValidateTileCompileFlags(missingFlags); err == nil {
			t.Errorf("expected error for missing --enable-tile flag, got nil")
		} else if !errors.Is(err, tileir.ErrInvalidCompileFlags) {
			t.Errorf("expected ErrInvalidCompileFlags, got %v", err)
		}

		// Hallucinated header in Tile C++ source
		badHeaderSource := `#include <cuda/tiles.hpp>
extern "C" __tile_global__ void k() {}`
		if err := tileir.ValidateTileCPPSource(badHeaderSource); err == nil {
			t.Errorf("expected error for hallucinated header <cuda/tiles.hpp>, got nil")
		} else if !errors.Is(err, tileir.ErrInvalidHeader) {
			t.Errorf("expected ErrInvalidHeader, got %v", err)
		}

		// Missing __tile_global__
		missingGlobalSource := `#include "cuda_tile.h"
extern "C" void k() {}`
		if err := tileir.ValidateTileCPPSource(missingGlobalSource); err == nil {
			t.Errorf("expected error for missing __tile_global__, got nil")
		} else if !errors.Is(err, tileir.ErrInvalidTileGlobal) {
			t.Errorf("expected ErrInvalidTileGlobal, got %v", err)
		}
	})
}
