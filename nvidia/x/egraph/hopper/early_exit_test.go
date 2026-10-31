package hopper

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestEarlyExitProbe(t *testing.T) {
	tests := []struct {
		name       string
		c          EarlyExitConfig
		curr, prev []float32
		score      float64
		exit       bool
	}{
		{"cosine same", EarlyExitConfig{Criterion: ExitHiddenCosine, Threshold: 0.99}, []float32{1, 2, 3}, []float32{2, 4, 6}, 1, true},
		{"cosine orthogonal", EarlyExitConfig{Criterion: ExitHiddenCosine, Threshold: 0.99}, []float32{1, 0}, []float32{0, 1}, 0, false},
		{"cosine zero", EarlyExitConfig{Criterion: ExitHiddenCosine, Threshold: 0.99}, []float32{0, 0}, []float32{0, 1}, 1, true},
		{"entropy uniform", EarlyExitConfig{Criterion: ExitProbeEntropy, Threshold: 1}, []float32{3, 3, 3, 3}, nil, math.Log(4), false},
		{"entropy peaked", EarlyExitConfig{Criterion: ExitProbeEntropy, Threshold: 0.01}, []float32{0, 40, 0, 0}, nil, 0, true},
		{"margin wide", EarlyExitConfig{Criterion: ExitTop2Margin, Threshold: 2}, []float32{1, 5, 2.5, -1}, nil, 2.5, true},
		{"margin tie", EarlyExitConfig{Criterion: ExitTop2Margin, Threshold: 0.5}, []float32{5, 1, 5}, nil, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score, exit := tt.c.Probe(tt.curr, tt.prev)
			if math.Abs(score-tt.score) > 1e-9 || exit != tt.exit {
				t.Fatalf("Probe = %v, %v; want %v, %v", score, exit, tt.score, tt.exit)
			}
		})
	}
}

func TestEarlyExitConfigErrors(t *testing.T) {
	ok := EarlyExitConfig{Tokens: 8, Dim: 4, NumLayers: 4}
	tests := []struct {
		name string
		edit func(*EarlyExitConfig)
		want string
	}{
		{"name", func(c *EarlyExitConfig) { c.Name = "1x" }, "bad name"},
		{"tokens", func(c *EarlyExitConfig) { c.Tokens = 0 }, "Tokens"},
		{"layers", func(c *EarlyExitConfig) { c.NumLayers = 0 }, "NumLayers"},
		{"tile", func(c *EarlyExitConfig) { c.TileM = 6 }, "TileM"},
		{"criterion", func(c *EarlyExitConfig) { c.Criterion = 9 }, "criterion"},
		{"margin dim", func(c *EarlyExitConfig) { c.Criterion, c.Dim = ExitTop2Margin, 1 }, "Dim >= 2"},
		{"threshold", func(c *EarlyExitConfig) { c.Threshold = float32(math.NaN()) }, "Threshold"},
	}
	for _, tt := range tests {
		c := ok
		tt.edit(&c)
		if _, err := c.CUDA(); err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: CUDA() error = %v, want %q", tt.name, err, tt.want)
		}
	}
	if _, err := ok.CUDA(); err != nil {
		t.Fatal(err)
	}
}

// TestSimulateAdaptiveDepthLayerOracle is the tile-savings oracle: 390 of
// 512 tokens converge at layer 16 of 32, and the 122 left fit one tile.
func TestSimulateAdaptiveDepthLayerOracle(t *testing.T) {
	const tokens = 512
	cfg := EarlyExitConfig{Tokens: tokens, Dim: 8, NumLayers: 32, MinLayer: 16, CheckpointStep: 4, Criterion: ExitHiddenCosine, Threshold: 0.99}
	r := rand.New(rand.NewPCG(29, 1))
	perm := r.Perm(tokens)
	hard := map[int]bool{}
	for _, t := range perm[:122] {
		hard[t] = true
	}
	curr := make([][]float32, tokens)
	prev := make([][]float32, tokens)
	exited := make([]int, tokens)
	for i := range tokens {
		exited[i] = cfg.NumLayers
		curr[i] = make([]float32, cfg.Dim)
		prev[i] = make([]float32, cfg.Dim)
		for d := range cfg.Dim {
			prev[i][d] = float32(r.NormFloat64())
			curr[i][d] = prev[i][d]
			if hard[i] {
				curr[i][d] = float32(r.NormFloat64())
			}
		}
	}
	st, err := SimulateAdaptiveDepthLayer(curr, prev, exited, 16, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(st.ActiveTokenIdx) != 122 || st.ActiveTiles != 1 || st.CancelledTiles != 3 {
		t.Fatalf("active %d tiles %d cancelled %d, want 122 1 3", len(st.ActiveTokenIdx), st.ActiveTiles, st.CancelledTiles)
	}
	// Bijection: every token is active xor exited at 16, actives in order.
	if !slices.IsSorted(st.ActiveTokenIdx) {
		t.Fatal("active tokens out of order")
	}
	for i, e := range st.ExitedAtLayer {
		active := slices.Contains(st.ActiveTokenIdx, int32(i))
		if hard[i] != active || (e == 16) == active || (e != 16 && e != cfg.NumLayers) {
			t.Fatalf("token %d: hard %v active %v exited at %d", i, hard[i], active, e)
		}
	}
	if len(st.Gather4Quads) != 31 || st.Gather4Quads[30] != [4]int32{st.ActiveTokenIdx[120], st.ActiveTokenIdx[121], -1, -1} {
		t.Fatalf("quads: %d, last %v", len(st.Gather4Quads), st.Gather4Quads[len(st.Gather4Quads)-1])
	}
	// Not a checkpoint: nobody exits.
	if st, err := SimulateAdaptiveDepthLayer(curr, prev, exited, 17, cfg); err != nil || len(st.ActiveTokenIdx) != tokens {
		t.Fatalf("layer 17: %d active, err %v; want %d", len(st.ActiveTokenIdx), err, tokens)
	}
}

// TestEarlyExitNeverSkipsHighEntropy checks the safety property of the
// entropy criterion over random probes: a token whose entropy is above the
// threshold always runs the next layer, and one already exited stays out.
func TestEarlyExitNeverSkipsHighEntropy(t *testing.T) {
	r := rand.New(rand.NewPCG(29, 2))
	for trial := range 200 {
		cfg := EarlyExitConfig{
			Tokens:    1 + r.IntN(300),
			Dim:       2 + r.IntN(30),
			NumLayers: 8,
			Criterion: ExitProbeEntropy,
			Threshold: 3 * r.Float32(),
			TileM:     4 * (1 + r.IntN(32)),
		}
		curr := make([][]float32, cfg.Tokens)
		exited := make([]int, cfg.Tokens)
		for i := range curr {
			exited[i] = cfg.NumLayers
			if r.IntN(4) == 0 {
				exited[i] = r.IntN(3)
			}
			curr[i] = make([]float32, cfg.Dim)
			scale := 4 * r.Float64()
			for d := range curr[i] {
				curr[i][d] = float32(scale * r.NormFloat64())
			}
		}
		layer := 3
		st, err := SimulateAdaptiveDepthLayer(curr, nil, exited, layer, cfg)
		if err != nil {
			t.Fatal(err)
		}
		for i := range cfg.Tokens {
			h, _ := cfg.Probe(curr[i], nil)
			active := slices.Contains(st.ActiveTokenIdx, int32(i))
			switch {
			case exited[i] < cfg.NumLayers && (active || st.ExitedAtLayer[i] != exited[i]):
				t.Fatalf("trial %d token %d: exited at %d but now active %v exit %d", trial, i, exited[i], active, st.ExitedAtLayer[i])
			case exited[i] == cfg.NumLayers && h > float64(cfg.Threshold) && !active:
				t.Fatalf("trial %d token %d: entropy %v > threshold %v but skipped", trial, i, h, cfg.Threshold)
			}
		}
	}
}

// earlyExitToy is a fixture for the emitted loop: layer l adds
// rate[l][t]·u_l to token t's state, with u_l a fixed vector of eighths and
// rate a power of two, so every product is exact and host and device
// agree bit for bit.
type earlyExitToy struct {
	cfg  EarlyExitConfig
	h0   []float32 // [Tokens][Dim]
	rate []float32 // [NumLayers][Tokens]
}

func toyU(layer, d int) float32 {
	return float32(int(treeSpecMix(uint32(layer), uint32(d), 7, 0)%17)-8) * 0.125
}

func newEarlyExitToy(t *testing.T, r *rand.Rand, cfg EarlyExitConfig) *earlyExitToy {
	cfg, err := cfg.norm()
	if err != nil {
		t.Fatal(err)
	}
	toy := &earlyExitToy{cfg: cfg, h0: make([]float32, cfg.Tokens*cfg.Dim), rate: make([]float32, cfg.NumLayers*cfg.Tokens)}
	for i := range toy.h0 {
		toy.h0[i] = float32(r.IntN(33)-16) * 0.0625
	}
	for t := range cfg.Tokens {
		speed := r.IntN(4) // 0 never settles
		for l := range cfg.NumLayers {
			toy.rate[l*cfg.Tokens+t] = float32(math.Ldexp(1, -speed*l))
		}
	}
	return toy
}

type earlyExitResult struct {
	h, kv     []float32
	exit      []int32
	tilesRun  int32
	layers    int32
	minMargin float64
}

// reference runs the loop on the host with SimulateAdaptiveDepthLayer.
func (toy *earlyExitToy) reference(t *testing.T) earlyExitResult {
	t.Helper()
	c := toy.cfg
	T, D, L := c.Tokens, c.Dim, c.NumLayers
	h := slices.Clone(toy.h0)
	hprev := make([]float32, T*D)
	kv := make([]float32, L*T*D)
	exited := slices.Repeat([]int{L}, T)
	active := make([]int32, T)
	for i := range active {
		active[i] = int32(i)
	}
	res := earlyExitResult{minMargin: math.Inf(1)}
	rows := func(x []float32) [][]float32 {
		out := make([][]float32, T)
		for i := range out {
			out[i] = x[i*D : (i+1)*D]
		}
		return out
	}
	for l := range L {
		res.layers++
		res.tilesRun += int32((len(active) + c.TileM - 1) / c.TileM)
		for _, tok := range active {
			r := toy.rate[l*T+int(tok)]
			for d := range D {
				i := int(tok)*D + d
				hprev[i] = h[i]
				h[i] += r * toyU(l, d)
				kv[(l*T)*D+i] = h[i]
			}
		}
		st, err := SimulateAdaptiveDepthLayer(rows(h), rows(hprev), exited, l, c)
		if err != nil {
			t.Fatal(err)
		}
		for tok := range T {
			if exited[tok] < L && c.PropagateSkipKV {
				copy(kv[(l*T+tok)*D:], h[tok*D:(tok+1)*D])
			}
			if exited[tok] == L && c.Checkpoint(l) {
				score, _ := c.Probe(h[tok*D:(tok+1)*D], hprev[tok*D:(tok+1)*D])
				res.minMargin = min(res.minMargin, math.Abs(score-float64(c.Threshold)))
			}
		}
		exited, active = st.ExitedAtLayer, st.ActiveTokenIdx
		if len(active) == 0 && !c.PropagateSkipKV {
			break
		}
	}
	res.h, res.kv = h, kv
	for _, e := range exited {
		res.exit = append(res.exit, int32(e))
	}
	return res
}

func (toy *earlyExitToy) rowSource() string {
	c := toy.cfg
	return treeSpecMixDevice + fmt.Sprintf(`
__device__ float* ee_h;
__device__ float* ee_hprev;
__device__ float* ee_kv;
__device__ const float* ee_rate;
__global__ void %[1]s_bind(float* h, float* hprev, float* kv, const float* rate) {
    ee_h = h;
    ee_hprev = hprev;
    ee_kv = kv;
    ee_rate = rate;
}
__device__ void %[1]s_row(void*, int layer, int32_t t) {
    constexpr int T = %[2]d, D = %[3]d;
    float* x = ee_h + (size_t)t * D;
    float* p = ee_hprev + (size_t)t * D;
    float* k = ee_kv + ((size_t)layer * T + t) * D;
    const float r = ee_rate[layer * T + t];
    for (int d = 0; d < D; ++d) {
        const float u = (float)((int)(eg_tree_spec_mix((uint32_t)layer, (uint32_t)d, 7u, 0u) %% 17u) - 8) * 0.125f;
        p[d] = x[d];
        x[d] = x[d] + r * u;
        k[d] = x[d];
    }
}
`, c.Name, c.Tokens, c.Dim)
}

// initialBuffers returns the device state before layer 0.
func (toy *earlyExitToy) initialBuffers() map[string][]byte {
	c := toy.cfg
	tiles := c.StaticTiles()
	idx := slices.Repeat([]int32{-1}, tiles*c.TileM)
	for i := range c.Tokens {
		idx[i] = int32(i)
	}
	return map[string][]byte{
		"h":          f32le(toy.h0),
		"hprev":      f32le(make([]float32, c.Tokens*c.Dim)),
		"kv":         f32le(make([]float32, c.NumLayers*c.Tokens*c.Dim)),
		"rate":       f32le(toy.rate),
		"exit_layer": specI32Bytes(slices.Repeat([]int32{int32(c.NumLayers)}, c.Tokens)),
		"active_idx": specI32Bytes(idx),
		"ctrl":       specI32Bytes([]int32{0, int32(c.Tokens), int32(tiles)}),
		"tiles_run":  specI32Bytes([]int32{0}),
	}
}

// runEmulated runs the loop under cudahostemu.RunConditionalWhile with
// activeCTAs resident CTAs when cfg.CLC is set.
func (toy *earlyExitToy) runEmulated(t *testing.T, src string, activeCTAs int) (earlyExitResult, error) {
	t.Helper()
	c := toy.cfg
	bufs := toy.initialBuffers()
	var args []cudahostemu.Arg
	for _, n := range []string{"h", "hprev", "kv", "rate", "exit_layer", "active_idx", "ctrl", "tiles_run"} {
		ct := "float"
		if n == "exit_layer" || n == "active_idx" || n == "ctrl" || n == "tiles_run" {
			ct = "int32_t"
		}
		args = append(args, cudahostemu.BufferArg(n, ct, bufs[n]))
	}
	args = append(args, cudahostemu.CondHandleArg("cond_handle", 1))
	tiles := cudahostemu.CondGraphStep{Kernel: c.Name + "_tiles", Grid: cudahostemu.Dim3{X: c.StaticTiles()}, Block: cudahostemu.Dim3{X: 128},
		Args: []string{"active_idx", "ctrl", "tiles_run", "nullptr"}}
	if c.CLC {
		tiles = cudahostemu.CondGraphStep{Kernel: fmt.Sprintf(
			"eg_emu_clc_init(%d, %d); eg_emu_launch(dim3(%d), dim3(128), 0, %s_tiles, active_idx, ctrl, tiles_run, (void*)nullptr); eg_emu_clc_check()",
			activeCTAs, c.StaticTiles(), activeCTAs, c.Name)}
	}
	body := []cudahostemu.CondGraphStep{
		{Kernel: c.Name + "_bind", Grid: cudahostemu.Dim3{X: 1}, Block: cudahostemu.Dim3{X: 1}, Args: []string{"h", "hprev", "kv", "rate"}},
		tiles,
		{Kernel: c.Name + "_probe", Grid: cudahostemu.Dim3{X: 1}, Block: cudahostemu.Dim3{X: 256},
			Args: []string{"h", "hprev", "kv", "exit_layer", "active_idx", "ctrl", "cond_handle"}},
	}
	full := src
	if c.CLC {
		full = cudahostemu.CLCHeader() + src
	}
	res, err := cudahostemu.RunConditionalWhile(t.Context(), full+toy.rowSource(), cudahostemu.CondWhileConfig{
		Handle: 1, InitialCondition: 1, MaxIterations: c.NumLayers + 1, RequireExplicitUpdate: true, DetectDataRace: true,
	}, body, args...)
	if err != nil {
		return earlyExitResult{}, err
	}
	ctrl := specBytesI32(bufs["ctrl"])
	if int(ctrl[0]) != res.IterationsExecuted {
		return earlyExitResult{}, fmt.Errorf("ctrl[0] = %d after %d iterations", ctrl[0], res.IterationsExecuted)
	}
	return earlyExitResult{
		h:        bytesF32(bufs["h"]),
		kv:       bytesF32(bufs["kv"]),
		exit:     specBytesI32(bufs["exit_layer"]),
		tilesRun: specBytesI32(bufs["tiles_run"])[0],
		layers:   ctrl[0],
	}, nil
}

func earlyExitDiff(got, want earlyExitResult) string {
	var d []string
	if got.layers != want.layers {
		d = append(d, fmt.Sprintf("layers %d, want %d", got.layers, want.layers))
	}
	if got.tilesRun != want.tilesRun {
		d = append(d, fmt.Sprintf("tiles run %d, want %d", got.tilesRun, want.tilesRun))
	}
	if !slices.Equal(got.exit, want.exit) {
		d = append(d, fmt.Sprintf("exit layers %v, want %v", got.exit, want.exit))
	}
	if !slices.Equal(got.h, want.h) {
		d = append(d, "final hidden states differ")
	}
	if !slices.Equal(got.kv, want.kv) {
		d = append(d, "KV differs")
	}
	return strings.Join(d, "\n")
}

func TestEarlyExitEmulatedLoop(t *testing.T) {
	if _, err := cudahostemu.Compiler(); errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	tests := []struct {
		cfg        EarlyExitConfig
		activeCTAs int
		mutants    bool
	}{
		{EarlyExitConfig{Name: "ee_cos_clc", Tokens: 300, Dim: 8, NumLayers: 8, MinLayer: 2, Criterion: ExitHiddenCosine, Threshold: 0.98, TileM: 64, PropagateSkipKV: true, CLC: true}, 2, true},
		{EarlyExitConfig{Name: "ee_entropy", Tokens: 200, Dim: 12, NumLayers: 7, MinLayer: 1, CheckpointStep: 2, Criterion: ExitProbeEntropy, Threshold: 2.2, TileM: 32}, 0, false},
		{EarlyExitConfig{Name: "ee_margin", Tokens: 70, Dim: 6, NumLayers: 6, MinLayer: 0, Criterion: ExitTop2Margin, Threshold: 0.6, TileM: 16, PropagateSkipKV: true}, 0, false},
	}
	for i, tt := range tests {
		t.Run(tt.cfg.Name, func(t *testing.T) {
			toy := newEarlyExitToy(t, rand.New(rand.NewPCG(29, uint64(10+i))), tt.cfg)
			want := toy.reference(t)
			if want.minMargin < 1e-4 {
				t.Fatalf("fixture has a probe score %g from the threshold; float rounding may decide it", want.minMargin)
			}
			var exitedEarly int
			for _, e := range want.exit {
				if int(e) < tt.cfg.NumLayers {
					exitedEarly++
				}
			}
			if exitedEarly == 0 || exitedEarly == tt.cfg.Tokens {
				t.Fatalf("fixture: %d of %d tokens exit early; want some but not all", exitedEarly, tt.cfg.Tokens)
			}
			src, err := tt.cfg.Source()
			if err != nil {
				t.Fatal(err)
			}
			got, err := toy.runEmulated(t, src, tt.activeCTAs)
			if err != nil {
				t.Fatal(err)
			}
			if d := earlyExitDiff(got, want); d != "" {
				t.Fatalf("emulated loop differs from reference:\n%s", d)
			}
			t.Logf("%d of %d tokens exited early; %d tiles run of %d static", exitedEarly, tt.cfg.Tokens, want.tilesRun, int(want.layers)*tt.cfg.StaticTiles())
			if !tt.mutants || testing.Short() {
				return
			}
			for _, m := range []struct{ name, old, new string }{
				{"no -1 padding", "active_idx[i] = -1;", "(void)i;"},
				{"compaction rank off by one", "__popc(ballot & ((1u << lane) - 1u))", "__popc(ballot & (2u << lane) - 1u)"},
				{"no KV propagation", "for (int d = 0; d < D; ++d) dst[d] = x[d];", "(void)dst;"},
				{"probe ignores threshold", "exit = cs >= THRESH;", "exit = cs >= 0.0f;"},
			} {
				t.Run(m.name, func(t *testing.T) {
					if !strings.Contains(src, m.old) {
						t.Fatalf("mutant site %q not in source", m.old)
					}
					got, err := toy.runEmulated(t, strings.Replace(src, m.old, m.new, 1), tt.activeCTAs)
					if err == nil && earlyExitDiff(got, want) == "" {
						t.Fatal("mutant survived")
					}
				})
			}
		})
	}
}

func ExampleSimulateAdaptiveDepthLayer() {
	cfg := EarlyExitConfig{Tokens: 6, Dim: 2, NumLayers: 4, MinLayer: 1, Criterion: ExitHiddenCosine, Threshold: 0.99, TileM: 4}
	prev := [][]float32{{1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}}
	// Tokens 1, 2 and 4 turned; the others kept their direction.
	curr := [][]float32{{2, 0}, {0, 1}, {1, 1}, {3, 0}, {-1, 0}, {1, 0}}
	st, err := SimulateAdaptiveDepthLayer(curr, prev, []int{4, 4, 4, 4, 4, 4}, 1, cfg)
	if err != nil {
		panic(err)
	}
	fmt.Println(st.ActiveTokenIdx, st.Gather4Quads, st.ActiveTiles, st.CancelledTiles)
	fmt.Println(st.ExitedAtLayer)
	// Output:
	// [1 2 4] [[1 2 4 -1]] 1 1
	// [1 4 4 1 4 1]
}
