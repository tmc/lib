package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// llama3_8B is Meta-Llama-3-8B's shape.
var llama3_8B = ModelConfig{
	HiddenSize:        4096,
	IntermediateSize:  14336,
	NumHiddenLayers:   32,
	NumAttentionHeads: 32,
	NumKeyValueHeads:  8,
	VocabSize:         128256,
	MaxSeqLen:         2048,
	RmsNormEps:        1e-5,
	RopeTheta:         500000,
}

func TestBuildEgraphPlans(t *testing.T) {
	p, err := buildEgraphPlans(&llama3_8B, false, egraphOptions{Profile: mustProfile(t, "GB10"), Attn: true, PrefillRows: -1})
	if err != nil {
		t.Fatal(err)
	}
	count := func(p *hopper.Plan) (skinny, tile, row int) {
		for _, k := range p.Kernels {
			switch {
			case k.GEMM != nil:
				skinny++
			case k.Tile != nil:
				tile++
				if k.Tile.WGMMA {
					t.Errorf("%s: wgmma is not available on sm_121", k.Tile.Name)
				}
			case k.Row != nil:
				row++
			default:
				t.Errorf("%s: unexpected kernel %+v", p.Name, k)
			}
		}
		return
	}
	tests := []struct {
		plan              *hopper.Plan
		skinny, tile, row int
	}{
		// rmsnorm row, qkv gemm.
		{p.attnIn, 1, 0, 1},
		{p.pf.attnIn, 0, 1, 1},
		// o_proj+x gemm reading F32 attn, rmsnorm row, gate_up gemm with
		// the SwiGLU epilogue, down_proj+h gemm. Prefill keeps the swiglu
		// row: its tile gemms have no paired epilogue.
		{p.post, 3, 0, 1},
		{p.pf.post, 0, 3, 2},
		{p.head, 1, 0, 1},
	}
	for _, tt := range tests {
		skinny, tile, row := count(tt.plan)
		if skinny != tt.skinny || tile != tt.tile || row != tt.row {
			t.Errorf("%s: %d skinny gemms, %d tile gemms, %d row kernels; want %d, %d, %d",
				tt.plan.Name, skinny, tile, row, tt.skinny, tt.tile, tt.row)
		}
	}
	for _, pl := range []*hopper.Plan{p.pf.attnIn, p.pf.post} {
		for _, k := range pl.Kernels {
			if k.Tile != nil && k.Tile.SharedBytes() > 99*1024 {
				t.Errorf("%s: %d bytes of shared memory, more than sm_121's 99 KiB", k.Tile.Name, k.Tile.SharedBytes())
			}
		}
	}
	if p.pf.rows != llama3_8B.MaxSeqLen {
		t.Errorf("prefill chunk = %d rows, want the cache length %d", p.pf.rows, llama3_8B.MaxSeqLen)
	}
	if p.pf.qtok != 15 {
		t.Errorf("prefill attention groups = %d rows, want 15 (G=4, D=128 in 48 KiB of static shared memory)", p.pf.qtok)
	}
}

func mustProfile(t *testing.T, name string) cudadevicetable.Profile {
	t.Helper()
	p, err := cudadevicetable.LookupProfile(name)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// TestEgraphPlansTarget checks that the profile, not a constant, sets the
// attention target, the prefill tile and the decode attention splits.
func TestEgraphPlansTarget(t *testing.T) {
	small := llama3_8B
	small.NumHiddenLayers = 1
	tests := []struct {
		target  string
		splits  int // llama3_8B has 8 KV heads
		bm      int
		maxSmem int
	}{
		{"sm_121a", 6, 64, 99 << 10},
		{"sm_103a", 20, 128, 227 << 10},
		{"sm_100a", 18, 128, 227 << 10},
		{"sm_90a", 16, 128, 227 << 10},
		{"sm_120a", 23, 64, 99 << 10},
	}
	for _, tt := range tests {
		t.Run(tt.target, func(t *testing.T) {
			p, err := buildEgraphPlans(&small, false, egraphOptions{Profile: mustProfile(t, tt.target), Attn: true, PrefillRows: -1, AttnSplits: -1})
			if err != nil {
				t.Fatal(err)
			}
			if p.sdpa.Target != tt.target || p.sdpa.Splits != tt.splits {
				t.Errorf("attention target %s, %d splits; want %s, %d", p.sdpa.Target, p.sdpa.Splits, tt.target, tt.splits)
			}
			if _, err := p.sdpa.CUDA(); err != nil {
				t.Error(err)
			}
			for _, pl := range []*hopper.Plan{p.pf.attnIn, p.pf.post} {
				for _, k := range pl.Kernels {
					if k.Tile == nil {
						continue
					}
					if k.Tile.BM != tt.bm || k.Tile.SharedBytes() > tt.maxSmem {
						t.Errorf("%s: BM %d, %d bytes of shared memory; want BM %d within %d", k.Tile.Name, k.Tile.BM, k.Tile.SharedBytes(), tt.bm, tt.maxSmem)
					}
				}
			}
		})
	}
	if _, err := buildEgraphPlans(&small, false, egraphOptions{Attn: true}); err == nil {
		t.Error("built plans without a target profile")
	}
}

func TestEmitEgraph(t *testing.T) {
	dir := t.TempDir()
	cfg := llama3_8B
	if err := emitEgraph(dir, "GB300", &cfg, egraphOptions{Attn: true, PrefillRows: -1, AttnSplits: -1}); err != nil {
		t.Fatal(err)
	}
	src, err := os.ReadFile(filepath.Join(dir, "sm_103a", "llama_egraph.cu"))
	if err != nil {
		t.Fatal(err)
	}
	names, err := os.ReadFile(filepath.Join(dir, "sm_103a", "names.txt"))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range strings.Fields(string(names)) {
		if !strings.Contains(string(src), strings.TrimSuffix(name, "<1>")) {
			t.Errorf("names.txt lists %s, which the source does not define", name)
		}
	}
	if !strings.Contains(string(src), "sm_103a register-accumulated") {
		t.Error("attention is not emitted for sm_103a")
	}
}
