package tileir_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileir"
)

func TestLookupDoc(t *testing.T) {
	tests := []struct {
		name         string
		query        string
		wantMnemonic string
		wantCategory tileir.Category
		wantLowering string
	}{
		{
			name:         "mma_f with prefix",
			query:        "cuda_tile.mma_f",
			wantMnemonic: "mma_f",
			wantCategory: tileir.CatComputeMMA,
			wantLowering: "wgmma.mma_async / mma.sync",
		},
		{
			name:         "mma_f without prefix",
			query:        "mma_f",
			wantMnemonic: "mma_f",
			wantCategory: tileir.CatComputeMMA,
			wantLowering: "wgmma.mma_async / mma.sync",
		},
		{
			name:         "load_view_tko case insensitive",
			query:        "CUDA_TILE.LOAD_VIEW_TKO",
			wantMnemonic: "load_view_tko",
			wantCategory: tileir.CatTileMemory,
			wantLowering: "cp.async.bulk.tensor",
		},
		{
			name:         "make_token sync",
			query:        "make_token",
			wantMnemonic: "make_token",
			wantCategory: tileir.CatSyncToken,
		},
		{
			name:         "scaled mma on Blackwell",
			query:        "mmaf_scaled",
			wantMnemonic: "mmaf_scaled",
			wantCategory: tileir.CatComputeMMA,
			wantLowering: "tcgen05.mma",
		},
		{
			name:         "convert_layout",
			query:        "convert_layout",
			wantMnemonic: "convert_layout",
			wantCategory: tileir.CatShapeLayout,
			wantLowering: "stmatrix / ldmatrix",
		},
		{
			name:         "pointer arithmetic fallback",
			query:        "int_to_ptr",
			wantMnemonic: "int_to_ptr",
			wantCategory: tileir.CatTypeCast,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc := tileir.LookupDoc(tt.query)
			if doc == nil {
				t.Fatalf("LookupDoc(%q) returned nil", tt.query)
			}
			if doc.Mnemonic != tt.wantMnemonic {
				t.Errorf("Mnemonic = %q, want %q", doc.Mnemonic, tt.wantMnemonic)
			}
			if doc.Category != tt.wantCategory {
				t.Errorf("Category = %v, want %v", doc.Category, tt.wantCategory)
			}
			if tt.wantLowering != "" && doc.HardwareLowering != tt.wantLowering {
				t.Errorf("HardwareLowering = %q, want %q", doc.HardwareLowering, tt.wantLowering)
			}
		})
	}
}

func TestLookupDocUnknown(t *testing.T) {
	if doc := tileir.LookupDoc("not_a_valid_opcode"); doc != nil {
		t.Errorf("LookupDoc(not_a_valid_opcode) = %+v, want nil", doc)
	}
	if doc := tileir.LookupDoc(""); doc != nil {
		t.Errorf("LookupDoc(\"\") = %+v, want nil", doc)
	}
}

func TestCategoryProperties(t *testing.T) {
	tests := []struct {
		cat         tileir.Category
		name        string
		isCompute   bool
		isMemory    bool
	}{
		{tileir.CatComputeMMA, "Tensor Core MMA", true, false},
		{tileir.CatComputeElement, "Compute Elementwise", true, false},
		{tileir.CatComputeInteger, "Compute Integer", true, false},
		{tileir.CatReduction, "Reduction", true, false},
		{tileir.CatTileMemory, "Tile Memory", false, true},
		{tileir.CatSyncToken, "Sync & Tokens", false, false},
		{tileir.CatControlFlow, "Control Flow", false, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.cat.String(); got != tt.name {
				t.Errorf("String() = %q, want %q", got, tt.name)
			}
			if got := tt.cat.IsCompute(); got != tt.isCompute {
				t.Errorf("IsCompute() = %v, want %v", got, tt.isCompute)
			}
			if got := tt.cat.IsMemory(); got != tt.isMemory {
				t.Errorf("IsMemory() = %v, want %v", got, tt.isMemory)
			}
		})
	}
}

func TestExplainAndHoverInfo(t *testing.T) {
	exp := tileir.Explain("cuda_tile.mma_f")
	if !strings.Contains(exp, "Tensor Core MMA") {
		t.Errorf("Explain missing category: %s", exp)
	}
	if !strings.Contains(exp, "wgmma") {
		t.Errorf("Explain missing lowering: %s", exp)
	}

	hover := tileir.HoverInfo("load_view_tko")
	if !strings.Contains(hover, "### `cuda_tile.load_view_tko`") {
		t.Errorf("HoverInfo missing heading: %s", hover)
	}
	if !strings.Contains(hover, "cp.async.bulk.tensor") {
		t.Errorf("HoverInfo missing hardware lowering: %s", hover)
	}

	// Unknown opcode hover
	hoverUnknown := tileir.HoverInfo("unknown_op")
	if !strings.Contains(hoverUnknown, "Unknown Tile IR opcode") {
		t.Errorf("HoverInfo for unknown did not report unknown: %s", hoverUnknown)
	}
}
