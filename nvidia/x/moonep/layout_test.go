package moonep

import "testing"

// Golden values generated independently from the formulas in MoonEP's
// _create_context (moonep/api.py:254-347) and pad_dim0_for_alignment
// (moonep/buffer.py:36-52).
func TestNewLayoutGolden(t *testing.T) {
	tests := []struct {
		name        string
		cfg         Config
		vmm, mc     uintptr
		want        Layout
		hiddenBytes uintptr
		metaBytes   uintptr
	}{
		{
			name: "production-8rank",
			cfg:  Config{S: 4096, H: 7168, K: 8, E: 256, R: 8},
			vmm:  1 << 21, mc: 1 << 21,
			want: Layout{
				N: 32768, NvS: 40896, NvSPadded: 40960, NumVBlocks: 16,
				TpeOff: 40896, PlanOff: 42944, PlanElems: 13328,
				Topk0Off: 56272, OrderOff: 89040, Order0Off: 121808,
				BarrierOff: 154576, SrcInfoOff: 154579,
				MetaChunkLogical: 195475, MetaChunkPadded: 524288,
			},
			hiddenBytes: 587202560,
			metaBytes:   2097152,
		},
		{
			name: "small-mixed-granularity",
			cfg:  Config{S: 128, H: 512, K: 2, E: 16, R: 4, B: 2, TokenPadding: 4},
			vmm:  65536, mc: 131072,
			want: Layout{
				N: 256, NvS: 280, NvSPadded: 320, NumVBlocks: 1,
				TpeOff: 280, PlanOff: 344, PlanElems: 424,
				Topk0Off: 768, OrderOff: 1024, Order0Off: 1280,
				BarrierOff: 1536, SrcInfoOff: 1539,
				MetaChunkLogical: 1819, MetaChunkPadded: 32768,
			},
			hiddenBytes: 327680,
			metaBytes:   131072,
		},
		{
			name: "no-token-padding",
			cfg:  Config{S: 1024, H: 2048, K: 4, E: 64, R: 8, TokenPadding: 1},
			vmm:  1 << 21, mc: 1 << 22,
			want: Layout{
				N: 4096, NvS: 4096, NvSPadded: 4096, NumVBlocks: 2,
				TpeOff: 4096, PlanOff: 4608, PlanElems: 3344,
				Topk0Off: 7952, OrderOff: 12048, Order0Off: 16144,
				BarrierOff: 20240, SrcInfoOff: 20243,
				MetaChunkLogical: 24339, MetaChunkPadded: 1048576,
			},
			hiddenBytes: 16777216,
			metaBytes:   4194304,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l, err := NewLayout(tt.cfg, tt.vmm, tt.mc)
			if err != nil {
				t.Fatal(err)
			}
			checks := []struct {
				name      string
				got, want int
			}{
				{"N", l.N, tt.want.N},
				{"NvS", l.NvS, tt.want.NvS},
				{"NvSPadded", l.NvSPadded, tt.want.NvSPadded},
				{"NumVBlocks", l.NumVBlocks, tt.want.NumVBlocks},
				{"WeightsOff", l.WeightsOff, 0},
				{"TpeOff", l.TpeOff, tt.want.TpeOff},
				{"PlanOff", l.PlanOff, tt.want.PlanOff},
				{"PlanElems", l.PlanElems, tt.want.PlanElems},
				{"Topk0Off", l.Topk0Off, tt.want.Topk0Off},
				{"OrderOff", l.OrderOff, tt.want.OrderOff},
				{"Order0Off", l.Order0Off, tt.want.Order0Off},
				{"BarrierOff", l.BarrierOff, tt.want.BarrierOff},
				{"SrcInfoOff", l.SrcInfoOff, tt.want.SrcInfoOff},
				{"MetaChunkLogical", l.MetaChunkLogical, tt.want.MetaChunkLogical},
				{"MetaChunkPadded", l.MetaChunkPadded, tt.want.MetaChunkPadded},
			}
			for _, c := range checks {
				if c.got != c.want {
					t.Errorf("%s = %d, want %d", c.name, c.got, c.want)
				}
			}
			if got := l.HiddenChunkBytes(); got != tt.hiddenBytes {
				t.Errorf("HiddenChunkBytes = %d, want %d", got, tt.hiddenBytes)
			}
			if got := l.MetaChunkBytes(); got != tt.metaBytes {
				t.Errorf("MetaChunkBytes = %d, want %d", got, tt.metaBytes)
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

func TestRound4(t *testing.T) {
	for _, tt := range []struct{ n, want int }{
		{0, 0}, {1, 4}, {3, 4}, {4, 4}, {5, 8}, {255, 256},
	} {
		if got := round4(tt.n); got != tt.want {
			t.Errorf("round4(%d) = %d, want %d", tt.n, got, tt.want)
		}
	}
}
