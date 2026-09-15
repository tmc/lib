//go:build !windows

package moonep

import (
	"encoding/json"
	"os"
	"testing"
)

//go:generate python3 gen_golden.py ../../../../../MoonshotAI/MoonEP

// goldenFile is the layout oracle: MoonEP's own _create_context and
// pad_dim0_for_alignment arithmetic, executed rather than transcribed, by
// gen_golden.py against a MoonEP checkout. Regenerate it with go:generate
// after changing a case or moving to a newer MoonEP revision; Revision
// records which one the checked-in values came from.
const goldenFile = "testdata/layout_golden.json"

type goldenSet struct {
	Source   string
	Revision string
	Cases    []goldenCase
}

type goldenCase struct {
	Name   string
	Config struct {
		S, H, K, E, R, B int
		TokenPadding     int `json:"token_padding"`
	}
	VMMGranularity uintptr `json:"vmm_granularity"`
	McGranularity  uintptr `json:"mc_granularity"`
	Want           struct {
		N                int
		NvS              int
		NvSPadded        int `json:"NvS_padded"`
		NumVBlocks       int `json:"num_vblocks"`
		WeightsOff       int `json:"WEIGHTS_OFF"`
		TpeOff           int `json:"TPE_OFF"`
		PlanOff          int `json:"PLAN_OFF"`
		PlanElems        int `json:"planning_out_elems"`
		Topk0Off         int `json:"TOPK0_OFF"`
		OrderOff         int `json:"ORDER_OFF"`
		Order0Off        int `json:"ORDER0_OFF"`
		BarrierOff       int `json:"BARRIER_OFF"`
		SrcInfoOff       int `json:"SRC_INFO_OFF"`
		MetaChunkLogical int `json:"meta_chunk_logical"`
		MetaChunkPadded  int `json:"meta_chunk_padded"`
		B                int
		NumSMs           int `json:"num_sms"`
	}
}

func loadGolden(t *testing.T) goldenSet {
	t.Helper()
	data, err := os.ReadFile(goldenFile)
	if err != nil {
		t.Fatal(err)
	}
	var g goldenSet
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Cases) == 0 {
		t.Fatalf("%s: no cases", goldenFile)
	}
	return g
}

func TestNewLayoutGolden(t *testing.T) {
	g := loadGolden(t)
	t.Logf("oracle: %s %s", g.Source, g.Revision)
	for _, tt := range g.Cases {
		t.Run(tt.Name, func(t *testing.T) {
			c := tt.Config
			l, err := NewLayout(Config{
				S: c.S, H: c.H, K: c.K, E: c.E, R: c.R,
				B: c.B, TokenPadding: c.TokenPadding,
			}, tt.VMMGranularity, tt.McGranularity)
			if err != nil {
				t.Fatal(err)
			}
			w := tt.Want
			for _, c := range []struct {
				name      string
				got, want int
			}{
				{"N", l.N, w.N},
				{"NvS", l.NvS, w.NvS},
				{"NvSPadded", l.NvSPadded, w.NvSPadded},
				{"NumVBlocks", l.NumVBlocks, w.NumVBlocks},
				{"B", l.B, w.B},
				{"NumSMs", l.NumSMs, w.NumSMs},
				{"WeightsOff", l.WeightsOff, w.WeightsOff},
				{"TpeOff", l.TpeOff, w.TpeOff},
				{"PlanOff", l.PlanOff, w.PlanOff},
				{"PlanElems", l.PlanElems, w.PlanElems},
				{"Topk0Off", l.Topk0Off, w.Topk0Off},
				{"OrderOff", l.OrderOff, w.OrderOff},
				{"Order0Off", l.Order0Off, w.Order0Off},
				{"BarrierOff", l.BarrierOff, w.BarrierOff},
				{"SrcInfoOff", l.SrcInfoOff, w.SrcInfoOff},
				{"MetaChunkLogical", l.MetaChunkLogical, w.MetaChunkLogical},
				{"MetaChunkPadded", l.MetaChunkPadded, w.MetaChunkPadded},
			} {
				if c.got != c.want {
					t.Errorf("%s = %d, want %d", c.name, c.got, c.want)
				}
			}
			// Derived byte sizes, checked against the oracle's element
			// counts rather than against our own fields.
			if got, want := l.HiddenChunkBytes(), uintptr(w.NvSPadded*c.H*bf16Bytes); got != want {
				t.Errorf("HiddenChunkBytes = %d, want %d", got, want)
			}
			if got, want := l.MetaChunkBytes(), uintptr(w.MetaChunkPadded*int32Bytes); got != want {
				t.Errorf("MetaChunkBytes = %d, want %d", got, want)
			}
		})
	}
}

func TestNewLayoutInvariants(t *testing.T) {
	l, err := NewLayout(Config{S: 1000, H: 1536, K: 6, E: 96, R: 8}, 1<<21, 1<<22)
	if err != nil {
		t.Fatal(err)
	}
	// Defaults.
	if l.B != l.E/l.R {
		t.Errorf("B default = %d, want %d", l.B, l.E/l.R)
	}
	if l.NumSMs != 32 || l.TokenPadding != 128 {
		t.Errorf("defaults: NumSMs=%d TokenPadding=%d", l.NumSMs, l.TokenPadding)
	}
	// Region ordering and 16-byte alignment of vectorized regions.
	if l.TpeOff%4 != 0 || l.PlanOff%4 != 0 || l.Topk0Off%4 != 0 {
		t.Error("vectorized region offsets must be 4-element aligned")
	}
	if !(l.WeightsOff < l.TpeOff && l.TpeOff < l.PlanOff && l.PlanOff < l.Topk0Off &&
		l.Topk0Off < l.OrderOff && l.OrderOff < l.Order0Off && l.Order0Off < l.BarrierOff &&
		l.BarrierOff < l.SrcInfoOff && l.SrcInfoOff < l.MetaChunkLogical) {
		t.Error("meta regions out of order")
	}
	if l.SrcInfoOff != l.BarrierOff+barrierSlots {
		t.Error("src_info must directly follow the 3 barrier slots")
	}
	// Chunk padding satisfies both granularities and covers the logical size.
	if l.MetaChunkPadded < l.MetaChunkLogical {
		t.Error("padded chunk smaller than logical")
	}
	if l.MetaChunkBytes()%l.VMMGran != 0 || l.MetaChunkBytes()%l.McGran != 0 {
		t.Error("meta chunk must be aligned to both granularities")
	}
	if l.HiddenChunkBytes()%l.VMMGran != 0 {
		t.Error("hidden chunk must be VMM aligned")
	}
	if l.NvSPadded < l.NvS {
		t.Error("NvSPadded < NvS")
	}
}

func TestNewLayoutErrors(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		vmm, mc uintptr
	}{
		{"E-not-divisible", Config{S: 8, H: 8, K: 1, E: 7, R: 2}, 4096, 4096},
		{"zero-S", Config{S: 0, H: 8, K: 1, E: 4, R: 2}, 4096, 4096},
		{"zero-gran", Config{S: 8, H: 8, K: 1, E: 4, R: 2}, 0, 4096},
		{"K-too-large", Config{S: 8, H: 8, K: 33, E: 66, R: 2}, 4096, 4096},
		{"R-too-large", Config{S: 8, H: 8, K: 1, E: 258, R: 129}, 4096, 4096},
		// (R-1)*MetaChunkPadded overflows int32.
		{"meta-overflow", Config{S: 1 << 22, H: 128, K: 8, E: 128, R: 64}, 1 << 21, 1 << 21},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewLayout(tt.cfg, tt.vmm, tt.mc); err == nil {
				t.Errorf("NewLayout(%+v): want error", tt.cfg)
			}
		})
	}
}

func TestAlignUp(t *testing.T) {
	for _, tt := range []struct{ n, a, want int }{
		{0, 4, 0}, {1, 4, 4}, {3, 4, 4}, {4, 4, 4}, {5, 4, 8}, {255, 4, 256},
		{1, 1, 1}, {7, 8, 8}, {8, 8, 8}, {9, 8, 16},
	} {
		if got := alignUp(tt.n, tt.a); got != tt.want {
			t.Errorf("alignUp(%d, %d) = %d, want %d", tt.n, tt.a, got, tt.want)
		}
	}
}
