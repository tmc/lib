package train

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// GRPOConfig configures the fused GRPO loss over an LM head:
// logits z = h·Wᵀ / Temperature, computed TileV vocabulary columns at a
// time and never materialized as an [M, V] tensor.
type GRPOConfig struct {
	VocabSize   int
	HiddenDim   int
	TileM       int     // tokens per tile (default 128)
	TileV       int     // vocabulary columns per tile (default 256)
	ClipEpsLow  float32 // ratio clip below 1 (e.g. 0.2)
	ClipEpsHigh float32 // ratio clip above 1 (e.g. 0.28, DAPO clip-higher)
	BetaKL      float32 // weight of the k3 KL estimator (e.g. 0.04)
	Temperature float32 // logit temperature (default 1)
}

func (c GRPOConfig) norm() (GRPOConfig, error) {
	if c.VocabSize <= 0 || c.HiddenDim <= 0 {
		return c, fmt.Errorf("grpo: VocabSize %d and HiddenDim %d must be > 0", c.VocabSize, c.HiddenDim)
	}
	if c.TileM == 0 {
		c.TileM = 128
	}
	if c.TileV == 0 {
		c.TileV = 256
	}
	if c.TileM < 0 || c.TileV < 0 {
		return c, fmt.Errorf("grpo: negative tile %dx%d", c.TileM, c.TileV)
	}
	if c.Temperature == 0 {
		c.Temperature = 1
	}
	if c.Temperature < 0 || c.ClipEpsLow < 0 || c.ClipEpsLow >= 1 || c.ClipEpsHigh < 0 || c.BetaKL < 0 {
		return c, fmt.Errorf("grpo: bad coefficients clip=(%v,%v) beta=%v temperature=%v", c.ClipEpsLow, c.ClipEpsHigh, c.BetaKL, c.Temperature)
	}
	return c, nil
}

// GRPOTokenMeta is the per-token metadata of a rollout: 16 bytes per token
// in place of a row of logits.
type GRPOTokenMeta struct {
	TargetTokenID int32
	OldLogProb    float32 // log π_old(y | x)
	RefLogProb    float32 // log π_ref(y | x)
	Advantage     float32 // group-normalized advantage
}

// GRPOTokenLoss returns the GRPO loss of one token and its derivative with
// respect to the token's log-probability logp:
//
//	r    = exp(logp − old)
//	pg   = −min(r·A, clip(r, 1−εlow, 1+εhigh)·A)
//	kl   = exp(ref − logp) − (ref − logp) − 1      (k3 estimator)
//	loss = pg + β·kl
//
// The PPO clip passes no gradient when the clipped term is selected.
func GRPOTokenLoss(logp float64, m GRPOTokenMeta, cfg GRPOConfig) (loss, dlogp float64) {
	a := float64(m.Advantage)
	r := math.Exp(logp - float64(m.OldLogProb))
	clipped := min(max(r, 1-float64(cfg.ClipEpsLow)), 1+float64(cfg.ClipEpsHigh))
	pg := -min(r*a, clipped*a)
	var dpg float64
	if r*a <= clipped*a {
		dpg = -r * a
	}
	d := float64(m.RefLogProb) - logp
	kl := math.Exp(d) - d - 1
	dkl := 1 - math.Exp(d)
	beta := float64(cfg.BetaKL)
	return pg + beta*kl, dpg + beta*dkl
}

// ComputeFusedGRPOChunk computes the mean GRPO loss over the tokens of h,
// and its gradients dH [M][D] and dW [V][D], streaming the vocabulary in
// TileV-column tiles in two passes:
//
//   - pass 1 keeps a running (max, sum-exp) per token and the target logit,
//     giving the log-sum-exp and the target log-probability;
//   - pass 2 recomputes each logit tile, turns it in place into
//     dZ = g·(1[v=y] − softmax) / Temperature with g = ∂loss/∂logp, and
//     accumulates dH += dZ·W and dW += dZᵀ·h.
//
// The only scratch is one TileM×TileV logit tile. Arithmetic is float64;
// results are rounded to float32. This is the host model of the tile
// schedule a tcgen05 kernel would run with the tile in TMEM.
func ComputeFusedGRPOChunk(h, w [][]float32, meta []GRPOTokenMeta, cfg GRPOConfig) (loss float32, dH, dW [][]float32, err error) {
	cfg, err = cfg.norm()
	if err != nil {
		return 0, nil, nil, err
	}
	M, V, D := len(h), cfg.VocabSize, cfg.HiddenDim
	if M == 0 || len(meta) != M {
		return 0, nil, nil, fmt.Errorf("grpo: %d tokens with %d metadata entries", M, len(meta))
	}
	if len(w) != V {
		return 0, nil, nil, fmt.Errorf("grpo: w has %d rows, want VocabSize %d", len(w), V)
	}
	for i, row := range h {
		if len(row) != D {
			return 0, nil, nil, fmt.Errorf("grpo: h[%d] has %d columns, want %d", i, len(row), D)
		}
	}
	for i, row := range w {
		if len(row) != D {
			return 0, nil, nil, fmt.Errorf("grpo: w[%d] has %d columns, want %d", i, len(row), D)
		}
	}
	for i, m := range meta {
		if m.TargetTokenID < 0 || int(m.TargetTokenID) >= V {
			return 0, nil, nil, fmt.Errorf("grpo: token %d target %d out of range", i, m.TargetTokenID)
		}
	}

	invT := 1 / float64(cfg.Temperature)
	tile := make([]float64, cfg.TileM*cfg.TileV)
	// logitTile fills tile with z[m0+i][v0+j] for the given tile bounds.
	logitTile := func(m0, mn, v0, vn int) {
		for i := range mn {
			hr := h[m0+i]
			for j := range vn {
				wr := w[v0+j]
				var s float64
				for k := range D {
					s += float64(hr[k]) * float64(wr[k])
				}
				tile[i*cfg.TileV+j] = s * invT
			}
		}
	}

	dH64 := make([][]float64, M)
	for i := range dH64 {
		dH64[i] = make([]float64, D)
	}
	dW64 := make([][]float64, V)
	for i := range dW64 {
		dW64[i] = make([]float64, D)
	}
	mx := make([]float64, cfg.TileM)
	se := make([]float64, cfg.TileM)
	zy := make([]float64, cfg.TileM)
	g := make([]float64, cfg.TileM)
	var total float64
	for m0 := 0; m0 < M; m0 += cfg.TileM {
		mn := min(cfg.TileM, M-m0)
		for i := range mn {
			mx[i], se[i] = math.Inf(-1), 0
		}
		// Pass 1: online log-sum-exp.
		for v0 := 0; v0 < V; v0 += cfg.TileV {
			vn := min(cfg.TileV, V-v0)
			logitTile(m0, mn, v0, vn)
			for i := range mn {
				row := tile[i*cfg.TileV : i*cfg.TileV+vn]
				tm := math.Inf(-1)
				for _, z := range row {
					tm = max(tm, z)
				}
				nm := max(mx[i], tm)
				s := se[i] * math.Exp(mx[i]-nm)
				for _, z := range row {
					s += math.Exp(z - nm)
				}
				mx[i], se[i] = nm, s
				if y := int(meta[m0+i].TargetTokenID); y >= v0 && y < v0+vn {
					zy[i] = row[y-v0]
				}
			}
		}
		for i := range mn {
			lse := mx[i] + math.Log(se[i])
			l, d := GRPOTokenLoss(zy[i]-lse, meta[m0+i], cfg)
			total += l
			g[i] = d / float64(M)
			mx[i] = lse
		}
		// Pass 2: gradient tiles.
		for v0 := 0; v0 < V; v0 += cfg.TileV {
			vn := min(cfg.TileV, V-v0)
			logitTile(m0, mn, v0, vn)
			for i := range mn {
				y := int(meta[m0+i].TargetTokenID)
				for j := range vn {
					p := math.Exp(tile[i*cfg.TileV+j] - mx[i])
					ind := 0.0
					if v0+j == y {
						ind = 1
					}
					tile[i*cfg.TileV+j] = g[i] * (ind - p) * invT
				}
			}
			for i := range mn {
				hr, dh := h[m0+i], dH64[m0+i]
				for j := range vn {
					dz := tile[i*cfg.TileV+j]
					if dz == 0 {
						continue
					}
					wr, dw := w[v0+j], dW64[v0+j]
					for k := range D {
						dh[k] += dz * float64(wr[k])
						dw[k] += dz * float64(hr[k])
					}
				}
			}
		}
	}
	return float32(total / float64(M)), round32(dH64), round32(dW64), nil
}

func round32(a [][]float64) [][]float32 {
	out := make([][]float32, len(a))
	for i, row := range a {
		out[i] = make([]float32, len(row))
		for j, v := range row {
			out[i][j] = float32(v)
		}
	}
	return out
}

// MuonConfig configures the quintic Newton–Schulz orthogonalization used by
// the Muon optimizer.
type MuonConfig struct {
	Steps  int        // default 5
	Coeffs [3]float32 // (a, b, c); default (3.4445, −4.7750, 2.0315)
}

// DefaultMuonCoeffs are the quintic coefficients from the Muon reference
// implementation.
var DefaultMuonCoeffs = [3]float32{3.4445, -4.7750, 2.0315}

func (c MuonConfig) norm() (MuonConfig, error) {
	if c.Steps == 0 {
		c.Steps = 5
	}
	if c.Steps < 0 {
		return c, fmt.Errorf("muon: Steps %d must be > 0", c.Steps)
	}
	if c.Coeffs == [3]float32{} {
		c.Coeffs = DefaultMuonCoeffs
	}
	return c, nil
}

// MuonOrthogonalize maps g [rows][cols] towards the nearest semi-orthogonal
// matrix without an SVD. It normalizes X₀ = G / (‖G‖_F + 1e-7) and runs
//
//	A = X·Xᵀ
//	B = b·A + c·A²
//	X = a·X + B·X
//
// Steps times, on the transpose when rows > cols so A is the smaller Gram
// matrix. The quintic pushes every singular value into roughly [0.7, 1.2];
// it does not converge to exactly 1. Arithmetic is float64.
func MuonOrthogonalize(g [][]float32, cfg MuonConfig) ([][]float32, error) {
	cfg, err := cfg.norm()
	if err != nil {
		return nil, err
	}
	rows := len(g)
	if rows == 0 || len(g[0]) == 0 {
		return nil, errors.New("muon: empty matrix")
	}
	cols := len(g[0])
	for i, r := range g {
		if len(r) != cols {
			return nil, fmt.Errorf("muon: row %d has %d columns, want %d", i, len(r), cols)
		}
	}
	tr := rows > cols
	m, n := rows, cols
	if tr {
		m, n = cols, rows
	}
	x := make([]float64, m*n)
	var fro float64
	for i := range rows {
		for j := range cols {
			v := float64(g[i][j])
			fro += v * v
			if tr {
				x[j*n+i] = v
			} else {
				x[i*n+j] = v
			}
		}
	}
	s := 1 / (math.Sqrt(fro) + 1e-7)
	for i := range x {
		x[i] *= s
	}
	a, b, c := float64(cfg.Coeffs[0]), float64(cfg.Coeffs[1]), float64(cfg.Coeffs[2])
	A := make([]float64, m*m)
	B := make([]float64, m*m)
	next := make([]float64, m*n)
	for range cfg.Steps {
		for i := range m {
			for j := i; j < m; j++ {
				var acc float64
				for k := range n {
					acc += x[i*n+k] * x[j*n+k]
				}
				A[i*m+j], A[j*m+i] = acc, acc
			}
		}
		for i := range m {
			for j := range m {
				var acc float64
				for k := range m {
					acc += A[i*m+k] * A[k*m+j]
				}
				B[i*m+j] = b*A[i*m+j] + c*acc
			}
		}
		for i := range m {
			for j := range n {
				acc := a * x[i*n+j]
				for k := range m {
					acc += B[i*m+k] * x[k*n+j]
				}
				next[i*n+j] = acc
			}
		}
		x, next = next, x
	}
	out := make([][]float32, rows)
	for i := range rows {
		out[i] = make([]float32, cols)
		for j := range cols {
			if tr {
				out[i][j] = float32(x[j*n+i])
			} else {
				out[i][j] = float32(x[i*n+j])
			}
		}
	}
	return out, nil
}

// FusedGRPOChunk emits a CUDA C++ kernel for [ComputeFusedGRPOChunk] in
// FP32. One CTA handles one token: it streams the vocabulary in TileV
// shared-memory tiles twice (online log-sum-exp, then gradient), writes the
// token's loss and its dH row, and atomically accumulates dW, which the
// caller zeroes first. The gradient is scaled by the scale argument
// (1/M for a mean over M tokens).
//
// The kernel is plain SIMT CUDA: it compiles for any sm_80+ target and runs
// under cudahostemu. It does not use TMEM, tcgen05.mma or tcgen05.ld.red;
// those would replace the shared-memory tile and the dot-product loops.
type FusedGRPOChunk struct {
	Name    string
	Config  GRPOConfig
	Threads int
}

func (k FusedGRPOChunk) norm() (FusedGRPOChunk, error) {
	if k.Name == "" {
		k.Name = "fused_grpo_chunk"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("fused grpo: bad name %q", k.Name)
	}
	var err error
	if k.Config, err = k.Config.norm(); err != nil {
		return k, fmt.Errorf("fused grpo %s: %w", k.Name, err)
	}
	if k.Config.TileV > 4096 {
		return k, fmt.Errorf("fused grpo %s: TileV %d exceeds 4096", k.Name, k.Config.TileV)
	}
	if k.Threads, err = normThreads(k.Threads); err != nil {
		return k, fmt.Errorf("fused grpo %s: %w", k.Name, err)
	}
	return k, nil
}

func (k FusedGRPOChunk) params() []string {
	return []string{
		"float* __restrict__ loss",
		"float* __restrict__ dH",
		"float* __restrict__ dW",
		"const float* __restrict__ H",
		"const float* __restrict__ W",
		"const int* __restrict__ target",
		"const float* __restrict__ old_logp",
		"const float* __restrict__ ref_logp",
		"const float* __restrict__ adv",
		"int rows",
		"float scale",
	}
}

// CUDA returns the kernel source without [Prelude].
func (k FusedGRPOChunk) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	c := k.Config
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: fused GRPO loss and LM-head backward, V=%d D=%d TileV=%d.\n", k.Name, c.VocabSize, c.HiddenDim, c.TileV)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n", k.Threads, k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int V = %d, D = %d, TILE_V = %d, THREADS = %d;\n", c.VocabSize, c.HiddenDim, c.TileV, k.Threads)
	fmt.Fprintf(&b, "    constexpr float INV_T = %s, EPS_LO = %s, EPS_HI = %s, BETA = %s;\n",
		floatLiteral(1/c.Temperature), floatLiteral(c.ClipEpsLow), floatLiteral(c.ClipEpsHigh), floatLiteral(c.BetaKL))
	b.WriteString(`    const int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float eg_red[32];
    __shared__ float s_z[TILE_V];
    __shared__ float s_zy;
    const float* h = H + (size_t)r * D;
    const int y = target[r];
    float run_max = -__uint_as_float(0x7f800000u), run_sum = 0.0f;
    // Pass 1: online log-sum-exp over vocabulary tiles.
    for (int v0 = 0; v0 < V; v0 += TILE_V) {
        const int vn = V - v0 < TILE_V ? V - v0 : TILE_V;
        float tmax = -__uint_as_float(0x7f800000u);
        for (int j = threadIdx.x; j < vn; j += THREADS) {
            const float* w = W + (size_t)(v0 + j) * D;
            float z = 0.0f;
            for (int k = 0; k < D; ++k) z += h[k] * w[k];
            z *= INV_T;
            s_z[j] = z;
            tmax = fmaxf(tmax, z);
            if (v0 + j == y) s_zy = z;
        }
        tmax = eg_block_combine<1>(eg_warp_max(tmax), eg_red);
        const float nmax = fmaxf(run_max, tmax);
        float tsum = 0.0f;
        for (int j = threadIdx.x; j < vn; j += THREADS) tsum += expf(s_z[j] - nmax);
        tsum = eg_block_combine<0>(eg_warp_sum(tsum), eg_red);
        run_sum = run_sum * expf(run_max - nmax) + tsum;
        run_max = nmax;
    }
    const float lse = run_max + logf(run_sum);
    // Per-token GRPO loss and dloss/dlogp.
    const float logp = s_zy - lse;
    const float a = adv[r];
    const float ratio = expf(logp - old_logp[r]);
    const float clipped = fminf(fmaxf(ratio, 1.0f - EPS_LO), 1.0f + EPS_HI);
    const float pg = -fminf(ratio * a, clipped * a);
    const float dpg = ratio * a <= clipped * a ? -ratio * a : 0.0f;
    const float d = ref_logp[r] - logp;
    const float kl = expf(d) - d - 1.0f;
    const float g = (dpg + BETA * (1.0f - expf(d))) * scale;
    if (threadIdx.x == 0) loss[r] = pg + BETA * kl;
    for (int k = threadIdx.x; k < D; k += THREADS) dH[(size_t)r * D + k] = 0.0f;
    // Pass 2: recompute each tile and turn it into dZ in place.
    for (int v0 = 0; v0 < V; v0 += TILE_V) {
        const int vn = V - v0 < TILE_V ? V - v0 : TILE_V;
        __syncthreads();
        for (int j = threadIdx.x; j < vn; j += THREADS) {
            const float* w = W + (size_t)(v0 + j) * D;
            float z = 0.0f;
            for (int k = 0; k < D; ++k) z += h[k] * w[k];
            const float p = expf(z * INV_T - lse);
            s_z[j] = g * ((v0 + j == y ? 1.0f : 0.0f) - p) * INV_T;
        }
        __syncthreads();
        for (int k = threadIdx.x; k < D; k += THREADS) {
            float acc = 0.0f;
            for (int j = 0; j < vn; ++j) {
                const float dz = s_z[j];
                acc += dz * W[(size_t)(v0 + j) * D + k];
                atomicAdd(&dW[(size_t)(v0 + j) * D + k], dz * h[k]);
            }
            dH[(size_t)r * D + k] += acc;
        }
    }
}
`)
	return b.String(), nil
}

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k FusedGRPOChunk) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	return launcher(k.Name, k.params(), func(b *strings.Builder) {
		fmt.Fprintf(b, "    const dim3 grid(rows > 0 ? rows : 0), block(%d);\n", k.Threads)
	}), nil
}

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k FusedGRPOChunk) Emit() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// MuonNewtonSchulz emits FP32 CUDA C++ kernels for [MuonOrthogonalize] on a
// Rows×Cols matrix with Rows <= Cols (callers transpose taller matrices).
// The launcher normalizes G into X and runs Steps iterations of three
// kernels (Gram A = X·Xᵀ, B = b·A + c·A², X = a·X + B·X), ping-ponging X with
// a scratch buffer, and leaves the result in X.
//
// The kernels are plain SIMT CUDA, one thread per output element, and run
// under cudahostemu. They model the data flow a tcgen05 kernel would keep
// in TMEM (A and B resident across sub-steps); they do not use tcgen05.
type MuonNewtonSchulz struct {
	Name       string
	Rows, Cols int
	Config     MuonConfig
}

func (k MuonNewtonSchulz) norm() (MuonNewtonSchulz, error) {
	if k.Name == "" {
		k.Name = "muon_newton_schulz"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("muon newton-schulz: bad name %q", k.Name)
	}
	if k.Rows <= 0 || k.Cols <= 0 || k.Rows > k.Cols {
		return k, fmt.Errorf("muon newton-schulz %s: need 0 < Rows %d <= Cols %d", k.Name, k.Rows, k.Cols)
	}
	var err error
	if k.Config, err = k.Config.norm(); err != nil {
		return k, fmt.Errorf("muon newton-schulz %s: %w", k.Name, err)
	}
	return k, nil
}

func (k MuonNewtonSchulz) params() []string {
	return []string{
		"float* __restrict__ X",
		"float* __restrict__ Xtmp",
		"float* __restrict__ A",
		"float* __restrict__ B",
		"const float* __restrict__ G",
	}
}

// CUDA returns the kernel sources without [Prelude].
func (k MuonNewtonSchulz) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	c := k.Config.Coeffs
	var b strings.Builder
	n := k.Name
	fmt.Fprintf(&b, "// %s: Muon quintic Newton-Schulz, %dx%d, %d steps, (a,b,c)=(%v,%v,%v).\n",
		n, k.Rows, k.Cols, k.Config.Steps, c[0], c[1], c[2])
	fmt.Fprintf(&b, "namespace %s_ns {\nconstexpr int R = %d, C = %d;\n", n, k.Rows, k.Cols)
	fmt.Fprintf(&b, "constexpr float CA = %s, CB = %s, CC = %s;\n}\n", floatLiteral(c[0]), floatLiteral(c[1]), floatLiteral(c[2]))
	ns := n + "_ns::"
	fmt.Fprintf(&b, `// %[1]s_normalize: X = G / (||G||_F + 1e-7), one CTA.
__global__ void __launch_bounds__(256) %[1]s_normalize(float* __restrict__ X, const float* __restrict__ G) {
    __shared__ float eg_red[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < %[2]sR * %[2]sC; i += 256) ss += G[i] * G[i];
    ss = eg_block_combine<0>(eg_warp_sum(ss), eg_red);
    const float s = 1.0f / (sqrtf(ss) + 1e-7f);
    for (int i = threadIdx.x; i < %[2]sR * %[2]sC; i += 256) X[i] = G[i] * s;
}

// %[1]s_gram: A = X * X^T (R x R).
__global__ void __launch_bounds__(128) %[1]s_gram(float* __restrict__ A, const float* __restrict__ X) {
    const int idx = (int)blockIdx.x * 128 + (int)threadIdx.x;
    if (idx >= %[2]sR * %[2]sR) return;
    const int i = idx / %[2]sR, j = idx %% %[2]sR;
    float acc = 0.0f;
    for (int k = 0; k < %[2]sC; ++k) acc += X[i * %[2]sC + k] * X[j * %[2]sC + k];
    A[idx] = acc;
}

// %[1]s_poly: B = b*A + c*A*A (R x R).
__global__ void __launch_bounds__(128) %[1]s_poly(float* __restrict__ B, const float* __restrict__ A) {
    const int idx = (int)blockIdx.x * 128 + (int)threadIdx.x;
    if (idx >= %[2]sR * %[2]sR) return;
    const int i = idx / %[2]sR, j = idx %% %[2]sR;
    float acc = 0.0f;
    for (int k = 0; k < %[2]sR; ++k) acc += A[i * %[2]sR + k] * A[k * %[2]sR + j];
    B[idx] = %[2]sCB * A[idx] + %[2]sCC * acc;
}

// %[1]s_update: Xn = a*X + B*X (R x C).
__global__ void __launch_bounds__(128) %[1]s_update(float* __restrict__ Xn, const float* __restrict__ X, const float* __restrict__ B) {
    const int idx = (int)blockIdx.x * 128 + (int)threadIdx.x;
    if (idx >= %[2]sR * %[2]sC) return;
    const int i = idx / %[2]sC, j = idx %% %[2]sC;
    float acc = %[2]sCA * X[idx];
    for (int k = 0; k < %[2]sR; ++k) acc += B[i * %[2]sR + k] * X[k * %[2]sC + j];
    Xn[idx] = acc;
}

// %[1]s_copy: dst = src (R x C).
__global__ void __launch_bounds__(128) %[1]s_copy(float* __restrict__ dst, const float* __restrict__ src) {
    const int idx = (int)blockIdx.x * 128 + (int)threadIdx.x;
    if (idx < %[2]sR * %[2]sC) dst[idx] = src[idx];
}
`, n, ns)
	return b.String(), nil
}

// Launcher returns void Name_launch(X, Xtmp, A, B, G, cudaStream_t stream).
func (k MuonNewtonSchulz) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	n := k.Name
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", n, strings.Join(k.params(), ", "))
	fmt.Fprintf(&b, "    const dim3 one(1), b256(256), b128(128);\n")
	fmt.Fprintf(&b, "    const dim3 gramGrid((%d + 127) / 128), updGrid((%d + 127) / 128);\n", k.Rows*k.Rows, k.Rows*k.Cols)
	fmt.Fprintf(&b, "    EG_LAUNCH(%s_normalize, one, b256, 0, stream, X, G);\n", n)
	fmt.Fprintf(&b, "    float* cur = X;\n    float* nxt = Xtmp;\n")
	fmt.Fprintf(&b, "    for (int s = 0; s < %d; ++s) {\n", k.Config.Steps)
	fmt.Fprintf(&b, "        EG_LAUNCH(%s_gram, gramGrid, b128, 0, stream, A, (const float*)cur);\n", n)
	fmt.Fprintf(&b, "        EG_LAUNCH(%s_poly, gramGrid, b128, 0, stream, B, (const float*)A);\n", n)
	fmt.Fprintf(&b, "        EG_LAUNCH(%s_update, updGrid, b128, 0, stream, nxt, (const float*)cur, (const float*)B);\n", n)
	b.WriteString("        float* t = cur; cur = nxt; nxt = t;\n    }\n")
	fmt.Fprintf(&b, "    if (cur != X) {\n        EG_LAUNCH(%s_copy, updGrid, b128, 0, stream, X, (const float*)cur);\n    }\n}\n", n)
	return b.String(), nil
}

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernels, and launcher).
func (k MuonNewtonSchulz) Emit() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}
