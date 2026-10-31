package hopper

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/tcgen05"
)

// An ExitCriterion selects the per-token test that lets a token leave the
// layer stack early.
type ExitCriterion uint8

const (
	// ExitHiddenCosine exits when cos(h⁽ˡ⁾, h⁽ˡ⁻¹⁾) >= Threshold: the layer
	// barely changed the hidden state.
	ExitHiddenCosine ExitCriterion = iota
	// ExitProbeEntropy exits when the entropy, in nats, of the softmax of a
	// probe head's logits is <= Threshold.
	ExitProbeEntropy
	// ExitTop2Margin exits when the largest probe logit exceeds the second
	// largest by >= Threshold.
	ExitTop2Margin
)

func (c ExitCriterion) String() string {
	switch c {
	case ExitHiddenCosine:
		return "hidden-cosine"
	case ExitProbeEntropy:
		return "probe-entropy"
	case ExitTop2Margin:
		return "top2-margin"
	}
	return fmt.Sprintf("ExitCriterion(%d)", uint8(c))
}

// EarlyExitConfig configures token-adaptive depth (CALM / SkipDecode style
// early exit) decided on the device inside a CUDA graph conditional WHILE
// loop over layers. Each iteration runs two emitted kernels:
//
//	Name_tiles  run layer l for the active tokens, TileM rows per tile
//	Name_probe  test the tokens that ran layer l at checkpoints, propagate
//	            the KV of exited tokens, compact the survivors into
//	            active_idx and set the loop condition
//
// Name_tiles calls a device function the caller defines,
//
//	__device__ void Name_row(void* ctx, int layer, int32_t token);
//
// for every active token of every active tile. It is launched over the
// static grid of all tiles. With CLC it is persistent: a resident CTA
// claims further tiles with clusterlaunchcontrol.try_cancel and stops at
// the first tile past the active count, so the tiles freed by early exit
// are cancelled before they launch. Without CLC every CTA of the static
// grid starts and returns at once if its tile is inactive.
//
// Name_probe runs as one block. It compacts with vote.sync.ballot and popc
// in token order, so active_idx is a stable, order-preserving list whose
// padding up to whole tiles is -1; read four at a time it is the index list
// of TMA .gather4 row loads.
//
// The probe reads Dim floats per token from h (and, for ExitHiddenCosine,
// from hprev). For the probe-head criteria, h holds the probe logits. With
// PropagateSkipKV, the probe copies an exited token's h row into kv for each
// later layer, so later tokens never attend to an unwritten KV slot; the
// copy stands in for a learned projection.
//
// The kernels were run on the host emulator only. The CLC path needs
// sm_100 or later and uses the helpers of package tcgen05; the plain path
// uses no sm_90+ instructions.
type EarlyExitConfig struct {
	Name            string
	Tokens          int // tokens in the batch, 1..1<<20
	Dim             int // floats per token read by the probe, 1..8192
	NumLayers       int
	CheckpointStep  int // test every CheckpointStep layers from MinLayer (0 means 1)
	MinLayer        int // earliest layer after which a token may exit
	Criterion       ExitCriterion
	Threshold       float32
	TileM           int  // rows per tile, a multiple of 4 (0 means 128)
	PropagateSkipKV bool // copy exited tokens' h into later layers' KV
	CLC             bool // persistent Name_tiles with clusterlaunchcontrol
}

func (c EarlyExitConfig) norm() (EarlyExitConfig, error) {
	if c.Name == "" {
		c.Name = "early_exit"
	}
	if !cIdent(c.Name) {
		return c, fmt.Errorf("early exit: bad name %q", c.Name)
	}
	if c.CheckpointStep == 0 {
		c.CheckpointStep = 1
	}
	if c.TileM == 0 {
		c.TileM = 128
	}
	switch {
	case c.Tokens < 1 || c.Tokens > 1<<20:
		return c, fmt.Errorf("early exit %s: Tokens %d outside 1..%d", c.Name, c.Tokens, 1<<20)
	case c.Dim < 1 || c.Dim > 8192:
		return c, fmt.Errorf("early exit %s: Dim %d outside 1..8192", c.Name, c.Dim)
	case c.NumLayers < 1:
		return c, fmt.Errorf("early exit %s: NumLayers %d < 1", c.Name, c.NumLayers)
	case c.CheckpointStep < 1:
		return c, fmt.Errorf("early exit %s: CheckpointStep %d < 1", c.Name, c.CheckpointStep)
	case c.MinLayer < 0:
		return c, fmt.Errorf("early exit %s: MinLayer %d < 0", c.Name, c.MinLayer)
	case c.Criterion > ExitTop2Margin:
		return c, fmt.Errorf("early exit %s: unknown criterion %v", c.Name, c.Criterion)
	case c.Criterion == ExitTop2Margin && c.Dim < 2:
		return c, fmt.Errorf("early exit %s: %v needs Dim >= 2", c.Name, c.Criterion)
	case math.IsNaN(float64(c.Threshold)) || math.IsInf(float64(c.Threshold), 0):
		return c, fmt.Errorf("early exit %s: Threshold %v", c.Name, c.Threshold)
	case c.TileM < 4 || c.TileM > 1024 || c.TileM%4 != 0:
		return c, fmt.Errorf("early exit %s: TileM %d not a multiple of 4 in 4..1024", c.Name, c.TileM)
	}
	return c, nil
}

// StaticTiles returns the number of TileM-row tiles covering all tokens.
func (c EarlyExitConfig) StaticTiles() int {
	c, err := c.norm()
	if err != nil {
		return 0
	}
	return (c.Tokens + c.TileM - 1) / c.TileM
}

// Checkpoint reports whether tokens may exit after layer.
func (c EarlyExitConfig) Checkpoint(layer int) bool {
	step := max(c.CheckpointStep, 1)
	return layer >= c.MinLayer && (layer-c.MinLayer)%step == 0 && layer < c.NumLayers-1
}

// Probe returns the criterion's score for one token, computed in float64,
// and whether it lets the token exit. prev is used only by
// ExitHiddenCosine.
func (c EarlyExitConfig) Probe(curr, prev []float32) (score float64, exit bool) {
	switch c.Criterion {
	case ExitHiddenCosine:
		var dot, n1, n2 float64
		for d, x := range curr {
			y := float64(prev[d])
			dot += float64(x) * y
			n1 += float64(x) * float64(x)
			n2 += y * y
		}
		score = 1
		if n1 > 0 && n2 > 0 {
			score = dot / (math.Sqrt(n1) * math.Sqrt(n2))
		}
		return score, score >= float64(c.Threshold)
	case ExitProbeEntropy:
		m := math.Inf(-1)
		for _, x := range curr {
			m = max(m, float64(x))
		}
		var z, ez float64
		for _, x := range curr {
			e := math.Exp(float64(x) - m)
			z += e
			ez += e * (float64(x) - m)
		}
		score = math.Log(z) - ez/z
		return score, score <= float64(c.Threshold)
	default:
		a, b := math.Inf(-1), math.Inf(-1)
		for _, x := range curr {
			switch x := float64(x); {
			case x > a:
				a, b = x, a
			case x > b:
				b = x
			}
		}
		score = a - b
		return score, score >= float64(c.Threshold)
	}
}

// LayerCompactionState is the active set after one layer's exit test.
type LayerCompactionState struct {
	Layer          int
	ActiveTokenIdx []int32    // tokens still running, in token order
	ExitedAtLayer  []int      // per token: the layer after which it exited, or NumLayers
	Gather4Quads   [][4]int32 // ActiveTokenIdx in fours, the last padded with -1
	ActiveTiles    int        // TileM-row tiles the next layer runs
	CancelledTiles int        // static tiles the next layer skips
}

// SimulateAdaptiveDepthLayer is the host reference for Name_probe at
// layer: tokens with prevExited[t] == NumLayers ran the layer and, at a
// checkpoint, exit if cfg's criterion holds on hCurr[t] (and hPrev[t]).
// The survivors are compacted in token order.
func SimulateAdaptiveDepthLayer(hCurr, hPrev [][]float32, prevExited []int, layer int, cfg EarlyExitConfig) (LayerCompactionState, error) {
	cfg, err := cfg.norm()
	if err != nil {
		return LayerCompactionState{}, err
	}
	if len(hCurr) != cfg.Tokens || len(prevExited) != cfg.Tokens || (cfg.Criterion == ExitHiddenCosine && len(hPrev) != cfg.Tokens) {
		return LayerCompactionState{}, fmt.Errorf("early exit %s: %d rows, %d previous rows and %d exit layers for %d tokens",
			cfg.Name, len(hCurr), len(hPrev), len(prevExited), cfg.Tokens)
	}
	if layer < 0 || layer >= cfg.NumLayers {
		return LayerCompactionState{}, fmt.Errorf("early exit %s: layer %d outside 0..%d", cfg.Name, layer, cfg.NumLayers-1)
	}
	st := LayerCompactionState{Layer: layer, ExitedAtLayer: make([]int, cfg.Tokens)}
	for t := range cfg.Tokens {
		e := prevExited[t]
		if e == cfg.NumLayers && cfg.Checkpoint(layer) {
			var prev []float32
			if cfg.Criterion == ExitHiddenCosine {
				prev = hPrev[t]
			}
			if len(hCurr[t]) != cfg.Dim || (prev != nil && len(prev) != cfg.Dim) {
				return LayerCompactionState{}, fmt.Errorf("early exit %s: token %d row has %d floats, want %d", cfg.Name, t, len(hCurr[t]), cfg.Dim)
			}
			if _, exit := cfg.Probe(hCurr[t], prev); exit {
				e = layer
			}
		}
		st.ExitedAtLayer[t] = e
		if e == cfg.NumLayers && layer+1 < cfg.NumLayers {
			st.ActiveTokenIdx = append(st.ActiveTokenIdx, int32(t))
		}
	}
	for i := 0; i < len(st.ActiveTokenIdx); i += 4 {
		q := [4]int32{-1, -1, -1, -1}
		copy(q[:], st.ActiveTokenIdx[i:])
		st.Gather4Quads = append(st.Gather4Quads, q)
	}
	st.ActiveTiles = (len(st.ActiveTokenIdx) + cfg.TileM - 1) / cfg.TileM
	st.CancelledTiles = cfg.StaticTiles() - st.ActiveTiles
	return st, nil
}

func (c EarlyExitConfig) probeParams() []string {
	return []string{
		"const float* __restrict__ h",
		"const float* __restrict__ hprev",
		"float* __restrict__ kv",
		"int32_t* __restrict__ exit_layer",
		"int32_t* __restrict__ active_idx",
		"int32_t* __restrict__ ctrl",
		"cudaGraphConditionalHandle cond_handle",
	}
}

func (c EarlyExitConfig) tileParams() []string {
	return []string{
		"const int32_t* __restrict__ active_idx",
		"const int32_t* __restrict__ ctrl",
		"int32_t* __restrict__ tiles_run",
		"void* ctx",
	}
}

// CUDA returns Name_probe and Name_tiles (without [Prelude]). Buffers:
// h and hprev [Tokens][Dim]; kv [NumLayers][Tokens][Dim]; exit_layer
// [Tokens], NumLayers while active; active_idx [StaticTiles*TileM];
// ctrl {next layer, active tokens, active tiles}, initially
// {0, Tokens, StaticTiles} with active_idx the identity padded with -1;
// tiles_run counts tiles run.
func (c EarlyExitConfig) CUDA() (string, error) {
	c, err := c.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(specDecodeCondPrelude)
	if c.CLC {
		b.WriteString(tcgen05.CLCHelpers())
	}
	fmt.Fprintf(&b, "\n// %s: on-device early exit (%v, threshold %v), %d tokens, %d layers, tile %d, propagate KV %v.\n",
		c.Name, c.Criterion, c.Threshold, c.Tokens, c.NumLayers, c.TileM, c.PropagateSkipKV)
	fmt.Fprintf(&b, "__device__ void %s_row(void* ctx, int layer, int32_t token);\n\n", c.Name)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s_probe(\n    %s) {\n", c.Name, strings.Join(c.probeParams(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int T = %d, D = %d, L = %d, TILE_M = %d, MIN_LAYER = %d, STEP = %d, CRIT = %d;\n",
		c.Tokens, c.Dim, c.NumLayers, c.TileM, c.MinLayer, c.CheckpointStep, c.Criterion)
	fmt.Fprintf(&b, "    constexpr bool PROPAGATE = %v;\n    constexpr float THRESH = %s;\n", c.PropagateSkipKV, floatLiteral(c.Threshold))
	b.WriteString(`    __shared__ int32_t warp_cnt[8], warp_off[8], chunk_total;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int layer = ctrl[0];
    const bool checkpoint = layer >= MIN_LAYER && (layer - MIN_LAYER) % STEP == 0 && layer < L - 1;
    int32_t running = 0;
    for (int base = 0; base < T; base += 256) {
        const int t = base + tid;
        bool keep = false;
        if (t < T) {
            int32_t e = exit_layer[t];
            const float* x = h + (size_t)t * D;
            if (e < L) {
                if (PROPAGATE) {
                    float* dst = kv + ((size_t)layer * T + t) * D;
                    for (int d = 0; d < D; ++d) dst[d] = x[d];
                }
            } else if (checkpoint) {
                bool exit = false;
                if (CRIT == 0) {
                    const float* y = hprev + (size_t)t * D;
                    float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += x[d] * y[d];
                        n1 += x[d] * x[d];
                        n2 += y[d] * y[d];
                    }
                    const float cs = (n1 > 0.0f && n2 > 0.0f) ? dot / (sqrtf(n1) * sqrtf(n2)) : 1.0f;
                    exit = cs >= THRESH;
                } else if (CRIT == 1) {
                    float m = x[0];
                    for (int d = 1; d < D; ++d) m = fmaxf(m, x[d]);
                    float z = 0.0f, ez = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        const float e = expf(x[d] - m);
                        z += e;
                        ez += e * (x[d] - m);
                    }
                    exit = logf(z) - ez / z <= THRESH;
                } else {
                    float a = -__uint_as_float(0x7f800000u), b2 = a;
                    for (int d = 0; d < D; ++d) {
                        if (x[d] > a) {
                            b2 = a;
                            a = x[d];
                        } else if (x[d] > b2) {
                            b2 = x[d];
                        }
                    }
                    exit = a - b2 >= THRESH;
                }
                if (exit) {
                    e = layer;
                    exit_layer[t] = layer;
                }
            }
            keep = e == L && layer + 1 < L;
        }
        // Stable compaction: rank within the warp by ballot, then by warp.
        const unsigned ballot = __ballot_sync(0xffffffffu, keep);
        if (lane == 0) warp_cnt[warp] = __popc(ballot);
        __syncthreads();
        if (tid == 0) {
            int32_t off = 0;
            for (int w = 0; w < 8; ++w) {
                warp_off[w] = off;
                off += warp_cnt[w];
            }
            chunk_total = off;
        }
        __syncthreads();
        if (keep) {
            active_idx[running + warp_off[warp] + __popc(ballot & ((1u << lane) - 1u))] = t;
        }
        running += chunk_total;
        __syncthreads();
    }
    const int32_t tiles = (running + TILE_M - 1) / TILE_M;
    for (int i = running + tid; i < tiles * TILE_M; i += 256) active_idx[i] = -1;
    if (tid == 0) {
        const int next = layer + 1;
        ctrl[0] = next;
        ctrl[1] = running;
        ctrl[2] = tiles;
        cudaGraphSetConditional(cond_handle, (next < L && (running > 0 || PROPAGATE)) ? 1u : 0u);
    }
}

`)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s_tiles(\n    %s) {\n", c.Name, strings.Join(c.tileParams(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int TILE_M = %d;\n", c.TileM)
	b.WriteString(`    const int layer = ctrl[0], active_tiles = ctrl[2];
    const int tid = threadIdx.x;
`)
	if c.CLC {
		b.WriteString(`    __shared__ uint4 clc_response;
    __shared__ uint64_t clc_mbar;
    __shared__ int clc_next_tile;
    if ((int)blockIdx.x >= active_tiles) return;
    if (tid == 0) eg_clc_mbar_init(&clc_mbar, 1);
    __syncthreads();
    uint32_t clc_phase = 0;
    for (int tile = (int)blockIdx.x; tile < active_tiles;) {
        if (tid == 0) eg_clc_try_cancel_async(&clc_response, &clc_mbar);
`)
	} else {
		b.WriteString(`    const int tile = (int)blockIdx.x;
    if (tile >= active_tiles) return;
    {
`)
	}
	fmt.Fprintf(&b, `        for (int r = tid; r < TILE_M; r += blockDim.x) {
            const int32_t token = active_idx[tile * TILE_M + r];
            if (token >= 0) %s_row(ctx, layer, token);
        }
        if (tid == 0) atomicAdd(tiles_run, 1);
`, c.Name)
	if c.CLC {
		b.WriteString(`        if (tid == 0) {
            eg_clc_wait(&clc_mbar, clc_phase);
            clc_phase ^= 1u;
            // A stolen tile past the active count means every later one
            // is dead too: stop claiming them.
            clc_next_tile = eg_clc_query_is_canceled(clc_response) ? (int)eg_clc_query_first_ctaid_x(clc_response) : active_tiles;
        }
        __syncthreads();
        tile = clc_next_tile;
        __syncthreads();
    }
}
`)
	} else {
		b.WriteString("    }\n}\n")
	}
	return b.String(), nil
}

// Launcher returns host functions Name_probe_launch and Name_tiles_launch
// taking the kernel parameters followed by a cudaStream_t. The tiles
// launcher launches the static grid of StaticTiles blocks.
func (c EarlyExitConfig) Launcher() (string, error) {
	c, err := c.norm()
	if err != nil {
		return "", err
	}
	grid := c.StaticTiles()
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_probe_launch(%s, cudaStream_t stream) {\n", c.Name, strings.Join(c.probeParams(), ", "))
	fmt.Fprintf(&b, "    EG_LAUNCH(%s_probe, dim3(1), dim3(256), 0, stream, %s);\n}\n", c.Name, strings.Join(paramNames(c.probeParams()), ", "))
	fmt.Fprintf(&b, "inline void %s_tiles_launch(%s, cudaStream_t stream) {\n", c.Name, strings.Join(c.tileParams(), ", "))
	fmt.Fprintf(&b, "    EG_LAUNCH(%s_tiles, dim3(%d), dim3(128), 0, stream, %s);\n}\n", c.Name, grid, strings.Join(paramNames(c.tileParams()), ", "))
	return b.String(), nil
}

// Source returns [Prelude], the kernels and their launchers. The caller
// appends the definition of Name_row.
func (c EarlyExitConfig) Source() (string, error) {
	src, err := c.CUDA()
	if err != nil {
		return "", err
	}
	l, err := c.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}
