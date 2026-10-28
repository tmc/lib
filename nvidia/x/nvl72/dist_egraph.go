package nvl72

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// Distributed e-graph operators. Each takes and returns one per-rank
// matrix; the collectives define how ranks combine.
const (
	DistGemm          = "dist_gemm"              // (dist_gemm x w): per-rank x*w
	DistAllReduceSum  = "dist_allreduce_sum"     // every rank gets the sum over ranks
	DistReduceScatter = "dist_reducescatter_sum" // rank r gets row block r of the sum
	DistAllGather     = "dist_allgather"         // every rank gets all ranks' rows in rank order
	DistAdd           = "dist_add"               // per-rank elementwise sum
	DistRMSNorm       = "dist_rmsnorm"           // (dist_rmsnorm x g): row-wise RMSNorm scaled by g

	// FusedTileFabricRedGemm is (x w r) = add(allreduce(gemm(x,w)), r)
	// computed tile by tile, each tile reduced in the switch as soon as it
	// leaves the epilogue (fabric.try_red / cp.reduce.async.bulk.tensor to a
	// multimem address).
	FusedTileFabricRedGemm = "fused_tile_fabric_red_gemm"
	// FusedMultimemLdReduceRMSNorm is (p r g) = rmsnorm(add(allreduce(p), r), g)
	// with the reduction pulled by multimem.ld_reduce inside the norm kernel.
	FusedMultimemLdReduceRMSNorm = "fused_multimem_ld_reduce_rmsnorm"
	// PipelinedFabricGetGemm is (x w) = gemm(allgather(x), w) with each
	// peer's rows fetched while the previous block multiplies.
	PipelinedFabricGetGemm = "pipelined_fabric_get_gemm"
	// TiledMultimemScatterGemm is (x w) = reducescatter(gemm(x, w)) with
	// each row block reduced to its owner as it is produced.
	TiledMultimemScatterGemm = "tiled_multimem_scatter_gemm"
)

// DistRewriteKind names an algebraic collective-compute rewrite.
type DistRewriteKind string

const (
	RuleTilePushFabricRed      DistRewriteKind = "gemm_allreduce_to_tile_fabric_try_red_multimem"
	RulePullMultimemLdReduce   DistRewriteKind = "allreduce_rmsnorm_to_multimem_ld_reduce_fused"
	RuleAGMatmulToRingPrefetch DistRewriteKind = "allgather_matmul_to_pipelined_fabric_try_get_mma"
	RuleMatmulRSAssociative    DistRewriteKind = "matmul_reducescatter_to_tiled_multimem_scatter"
)

var distRuleForOp = map[string]DistRewriteKind{
	FusedTileFabricRedGemm:       RuleTilePushFabricRed,
	FusedMultimemLdReduceRMSNorm: RulePullMultimemLdReduce,
	PipelinedFabricGetGemm:       RuleAGMatmulToRingPrefetch,
	TiledMultimemScatterGemm:     RuleMatmulRSAssociative,
}

// AlgebraicCollectiveRules returns the four collective-compute rewrites.
// Each rewrites a collective applied to a linear op's output; none crosses
// a non-linear op. In particular the pull rule needs RMSNorm outside the
// all-reduce, and does not match RMSNorm inside it.
func AlgebraicCollectiveRules() []*egraph.Rewrite {
	return []*egraph.Rewrite{
		egraph.MustRewrite(string(RuleTilePushFabricRed),
			"(dist_add (dist_allreduce_sum (dist_gemm ?x ?w)) ?r)",
			"(fused_tile_fabric_red_gemm ?x ?w ?r)"),
		egraph.MustRewrite(string(RulePullMultimemLdReduce),
			"(dist_rmsnorm (dist_add (dist_allreduce_sum ?p) ?r) ?g)",
			"(fused_multimem_ld_reduce_rmsnorm ?p ?r ?g)"),
		egraph.MustRewrite(string(RuleAGMatmulToRingPrefetch),
			"(dist_gemm (dist_allgather ?x) ?w)",
			"(pipelined_fabric_get_gemm ?x ?w)"),
		egraph.MustRewrite(string(RuleMatmulRSAssociative),
			"(dist_reducescatter_sum (dist_gemm ?x ?w))",
			"(tiled_multimem_scatter_gemm ?x ?w)"),
	}
}

// DistFusionCostParams parameterizes the switch-fabric and tensor-core
// latency model. Every matrix in the graph is BatchTokens x HiddenDim BF16
// and every GEMM is BatchTokens x HiddenDim x HiddenDim.
type DistFusionCostParams struct {
	WorldSize          int     // ranks, e.g. 8, 72 or 144
	SwitchBWGBps       float64 // per-GPU unidirectional NVLink bandwidth (900 NVLink 5, 1800 NVLink 6)
	SwitchLatencyUs    float64 // NVSwitch traversal latency
	TensorCoreTFLOPS   float64 // per-GPU dense BF16 peak
	HBMGBps            float64 // per-GPU HBM bandwidth (default 8000)
	BatchTokens        int     // M
	HiddenDim          int     // N = K (default 7168)
	KernelLaunchUs     float64 // per kernel launch and drain (default 3)
	CollectiveLaunchUs float64 // NCCL-style collective setup and proxy (default 8)
	TileM, TileN       int     // output tile of the push epilogue (default 128 x 256)
	Target             string  // "sm_90a", "sm_100a" or "sm_107a" for the PTX listing (default sm_100a)
}

func (p DistFusionCostParams) norm() (DistFusionCostParams, error) {
	if p.HBMGBps == 0 {
		p.HBMGBps = NVL72HBM3eGBps
	}
	if p.HiddenDim == 0 {
		p.HiddenDim = 7168
	}
	if p.KernelLaunchUs == 0 {
		p.KernelLaunchUs = 3
	}
	if p.CollectiveLaunchUs == 0 {
		p.CollectiveLaunchUs = 8
	}
	if p.TileM == 0 {
		p.TileM = 128
	}
	if p.TileN == 0 {
		p.TileN = 256
	}
	if p.Target == "" {
		p.Target = "sm_100a"
	}
	switch {
	case p.WorldSize < 2:
		return p, fmt.Errorf("dist egraph: WorldSize %d must be >= 2", p.WorldSize)
	case p.SwitchBWGBps <= 0 || p.TensorCoreTFLOPS <= 0 || p.HBMGBps <= 0 || p.SwitchLatencyUs < 0:
		return p, fmt.Errorf("dist egraph: bandwidths and TFLOPS must be positive")
	case p.BatchTokens < 1 || p.HiddenDim < 1:
		return p, fmt.Errorf("dist egraph: BatchTokens %d and HiddenDim %d must be positive", p.BatchTokens, p.HiddenDim)
	case p.TileM < 1 || p.TileN < 1:
		return p, fmt.Errorf("dist egraph: tile %dx%d must be positive", p.TileM, p.TileN)
	}
	switch p.Target {
	case "sm_90a", "sm_100a", "sm_107a":
	default:
		return p, fmt.Errorf("dist egraph: target %q unsupported", p.Target)
	}
	return p, nil
}

// DistCostModel prices distributed e-nodes in microseconds. It implements
// [egraph.AdditiveCostModel]; leaves cost nothing.
type DistCostModel struct{ P DistFusionCostParams }

func (m DistCostModel) bytes() float64 { return float64(m.P.BatchTokens*m.P.HiddenDim) * 2 }

func (m DistCostModel) gemmUs() float64 {
	n := float64(m.P.HiddenDim)
	return 2 * float64(m.P.BatchTokens) * n * n / (m.P.TensorCoreTFLOPS * 1e6)
}

func (m DistCostModel) linkUs(b float64) float64 { return b / (m.P.SwitchBWGBps * 1e3) }
func (m DistCostModel) hbmUs(b float64) float64  { return b / (m.P.HBMGBps * 1e3) }

// tileTailUs is the communication of the last output tile, which cannot
// overlap compute.
func (m DistCostModel) tileTailUs() float64 {
	tiles := math.Ceil(float64(m.P.BatchTokens)/float64(m.P.TileM)) * math.Ceil(float64(m.P.HiddenDim)/float64(m.P.TileN))
	return m.linkUs(2*m.bytes()) / tiles
}

// NodeLocalCost returns the latency of n alone.
func (m DistCostModel) NodeLocalCost(n egraph.ENode) float64 {
	p, b := m.P, m.bytes()
	lat := p.SwitchLatencyUs
	switch n.Op {
	case DistGemm:
		return p.KernelLaunchUs + m.gemmUs()
	case DistAllReduceSum:
		// NVLS all-reduce: reduce-scatter and all-gather through the switch.
		return p.CollectiveLaunchUs + 2*lat + m.linkUs(2*b)
	case DistReduceScatter, DistAllGather:
		return p.CollectiveLaunchUs + lat + m.linkUs(b)
	case DistAdd:
		return p.KernelLaunchUs + m.hbmUs(3*b)
	case DistRMSNorm:
		return p.KernelLaunchUs + m.hbmUs(2*b)
	case FusedTileFabricRedGemm:
		// Tile reductions overlap later tiles' MMAs; the last tile and a
		// completion barrier remain exposed.
		return p.KernelLaunchUs + max(m.gemmUs(), m.linkUs(2*b)) + m.tileTailUs() + 2*lat
	case FusedMultimemLdReduceRMSNorm:
		// One pass: pull the reduced row, add residual, normalize.
		return p.KernelLaunchUs + lat + m.linkUs(b) + m.hbmUs(2*b)
	case PipelinedFabricGetGemm, TiledMultimemScatterGemm:
		return p.KernelLaunchUs + max(m.gemmUs(), m.linkUs(b)) + m.tileTailUs() + lat
	}
	return 0
}

// DistSchedule is the result of [ExtractOptimalDistSchedule].
type DistSchedule struct {
	Rule             DistRewriteKind // outermost fused rewrite used, or "" for none
	Expr             *egraph.Expr
	EstLatencyUs     float64
	UnfusedLatencyUs float64 // the unrewritten graph under the same model
	PTX              string  // PTX listing of the chosen fused op
}

// ExtractOptimalDistSchedule prices root as built, saturates g with
// [AlgebraicCollectiveRules], and extracts the cheapest equivalent
// schedule under [DistCostModel]. g is modified.
func ExtractOptimalDistSchedule(g *egraph.EGraph, root egraph.Id, params DistFusionCostParams) (DistSchedule, error) {
	params, err := params.norm()
	if err != nil {
		return DistSchedule{}, err
	}
	model := DistCostModel{P: params}
	extract := func() (*egraph.Expr, float64, error) {
		ext, err := egraph.NewAdditiveExtractorForRootChecked(g, root, model)
		if err != nil {
			return nil, 0, err
		}
		return ext.Extract(root)
	}
	_, unfused, err := extract()
	if err != nil {
		return DistSchedule{}, fmt.Errorf("dist egraph: extract unfused: %w", err)
	}
	egraph.Saturate(g, AlgebraicCollectiveRules(), 0)
	e, cost, err := extract()
	if err != nil {
		return DistSchedule{}, fmt.Errorf("dist egraph: extract: %w", err)
	}
	s := DistSchedule{Expr: e, EstLatencyUs: cost, UnfusedLatencyUs: unfused}
	var walk func(*egraph.Expr) bool
	walk = func(x *egraph.Expr) bool {
		if k, ok := distRuleForOp[x.Op]; ok {
			s.Rule = k
			return true
		}
		for _, c := range x.Children {
			if walk(c) {
				return true
			}
		}
		return false
	}
	walk(e)
	if s.PTX, err = DistRulePTX(s.Rule, params.Target); err != nil {
		return DistSchedule{}, err
	}
	return s, nil
}

// DistRulePTX returns the PTX instruction sequence of rule's fused op on
// target. The pull listing is the core of [MultimemRMSNormKernel]. The push
// listing uses fabric.try_red.async.tensor on sm_107a and, elsewhere,
// cp.reduce.async.bulk.tensor into a tensor map over a multimem address;
// that second form is modelled here and has not been assembled or run.
func DistRulePTX(rule DistRewriteKind, target string) (string, error) {
	switch rule {
	case "":
		return "", nil
	case RulePullMultimemLdReduce:
		return multimemPullPTX, nil
	case RuleTilePushFabricRed:
		red := "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%rd_tmap_mc, {%r_col, %r_row}], [%r_smem_tile];\n\tcp.async.bulk.commit_group;"
		if target == "sm_107a" {
			inst, err := RubinFabricAsyncTensorKernel{Op: FabricTensorTryRed, Multimem: true, Target: "sm_107a"}.PTXInstruction()
			if err != nil {
				return "", err
			}
			red = inst + " %r_status, [%rd_tmap_le_id, %rd_tmap_off, {%r_col, %r_row}], [%r_smem_tile], [%rd_mbar_le_id, %rd_mbar_off];"
		}
		return "// per output tile, after the epilogue has staged it in shared memory:\n" +
			"\tfence.proxy.async.shared::cta;\n\t" + red + "\n" +
			"\t// the next tile's tcgen05.mma issues here, overlapping the in-switch reduction\n", nil
	case RuleAGMatmulToRingPrefetch:
		return "\tcp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r_smem_next], [%rd_peer_rows], %r_bytes, [%r_mbar_next];\n" +
			"\tmbarrier.try_wait.parity.shared::cta.b64 %p_ready, [%r_mbar_cur], %r_phase;\n", nil
	case RuleMatmulRSAssociative:
		return "\tmultimem.red.relaxed.sys.global.add.v4.bf16x2 [%rd_owner_mc], {%r0, %r1, %r2, %r3};\n", nil
	}
	return "", fmt.Errorf("dist egraph: unknown rule %q", rule)
}

const multimemPullPTX = "\tmultimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%r0, %r1, %r2, %r3}, [%rd_mc];\n" +
	"\t// unpack 8 bf16, add residual, sum squares\n" +
	"\tshfl.sync.bfly.b32 %f_sq_o, %f_sq, 16, 31, -1;\n" +
	"\trsqrt.approx.ftz.f32 %f_inv, %f_ms;\n"

// DistMatrix is a dense row-major float32 matrix.
type DistMatrix struct {
	Rows, Cols int
	Data       []float32
}

func newDistMatrix(r, c int) DistMatrix { return DistMatrix{r, c, make([]float32, r*c)} }

func (a DistMatrix) at(i, j int) float32 { return a.Data[i*a.Cols+j] }

// DistEvalConfig sets the numeric semantics of [EvalDistExpr].
type DistEvalConfig struct {
	Ranks        int
	Eps          float32 // RMSNorm epsilon (default 1e-6)
	TileM, TileN int     // tile shape of fused push and scatter (default 2 x 4)
}

// EvalDistExpr evaluates e on cfg.Ranks ranks. Leaves name entries of env,
// one matrix per rank. Fused ops are evaluated by their tiled or pipelined
// algorithm, not by expanding to their definition, so comparing an
// expression with its rewrite checks the rewrite.
func EvalDistExpr(e *egraph.Expr, cfg DistEvalConfig, env map[string][]DistMatrix) ([]DistMatrix, error) {
	if cfg.Ranks < 1 {
		return nil, fmt.Errorf("dist eval: Ranks %d must be positive", cfg.Ranks)
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-6
	}
	if cfg.TileM == 0 {
		cfg.TileM = 2
	}
	if cfg.TileN == 0 {
		cfg.TileN = 4
	}
	ev := distEval{cfg: cfg, env: env}
	return ev.eval(e)
}

type distEval struct {
	cfg DistEvalConfig
	env map[string][]DistMatrix
}

func (d distEval) eval(e *egraph.Expr) ([]DistMatrix, error) {
	if len(e.Children) == 0 {
		v, ok := d.env[e.Op]
		if !ok {
			return nil, fmt.Errorf("dist eval: unbound leaf %q", e.Op)
		}
		if len(v) != d.cfg.Ranks {
			return nil, fmt.Errorf("dist eval: leaf %q has %d ranks, want %d", e.Op, len(v), d.cfg.Ranks)
		}
		return v, nil
	}
	args := make([][]DistMatrix, len(e.Children))
	for i, c := range e.Children {
		v, err := d.eval(c)
		if err != nil {
			return nil, err
		}
		args[i] = v
	}
	want := map[string]int{
		DistGemm: 2, DistAllReduceSum: 1, DistReduceScatter: 1, DistAllGather: 1, DistAdd: 2, DistRMSNorm: 2,
		FusedTileFabricRedGemm: 3, FusedMultimemLdReduceRMSNorm: 3, PipelinedFabricGetGemm: 2, TiledMultimemScatterGemm: 2,
	}
	n, ok := want[e.Op]
	if !ok {
		return nil, fmt.Errorf("dist eval: unknown op %q", e.Op)
	}
	if len(args) != n {
		return nil, fmt.Errorf("dist eval: %s takes %d operands, have %d", e.Op, n, len(args))
	}
	P := d.cfg.Ranks
	out := make([]DistMatrix, P)
	var err error
	switch e.Op {
	case DistGemm:
		for r := range P {
			if out[r], err = gemm(args[0][r], args[1][r]); err != nil {
				return nil, err
			}
		}
	case DistAllReduceSum:
		s, err := rankSum(args[0])
		if err != nil {
			return nil, err
		}
		for r := range P {
			out[r] = s
		}
	case DistReduceScatter:
		s, err := rankSum(args[0])
		if err != nil {
			return nil, err
		}
		for r := range P {
			if out[r], err = rowBlock(s, r, P); err != nil {
				return nil, err
			}
		}
	case DistAllGather:
		g := allGather(args[0])
		for r := range P {
			out[r] = g
		}
	case DistAdd:
		for r := range P {
			if out[r], err = addMat(args[0][r], args[1][r]); err != nil {
				return nil, err
			}
		}
	case DistRMSNorm:
		for r := range P {
			if out[r], err = rmsNorm(args[0][r], args[1][r], d.cfg.Eps); err != nil {
				return nil, err
			}
		}
	case FusedTileFabricRedGemm:
		return d.tilePush(args[0], args[1], args[2])
	case FusedMultimemLdReduceRMSNorm:
		return d.pullNorm(args[0], args[1], args[2])
	case PipelinedFabricGetGemm:
		return d.ringGemm(args[0], args[1])
	case TiledMultimemScatterGemm:
		return d.scatterGemm(args[0], args[1])
	}
	return out, nil
}

func gemm(a, b DistMatrix) (DistMatrix, error) {
	if a.Cols != b.Rows {
		return DistMatrix{}, fmt.Errorf("dist eval: gemm %dx%d by %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	c := newDistMatrix(a.Rows, b.Cols)
	for i := range a.Rows {
		for j := range b.Cols {
			c.Data[i*c.Cols+j] = dotRowCol(a, b, i, j)
		}
	}
	return c, nil
}

func dotRowCol(a, b DistMatrix, i, j int) float32 {
	var s float32
	for k := range a.Cols {
		s += float32(a.at(i, k) * b.at(k, j))
	}
	return s
}

func rankSum(xs []DistMatrix) (DistMatrix, error) {
	s := newDistMatrix(xs[0].Rows, xs[0].Cols)
	for r, x := range xs {
		if x.Rows != s.Rows || x.Cols != s.Cols {
			return DistMatrix{}, fmt.Errorf("dist eval: rank %d is %dx%d, rank 0 %dx%d", r, x.Rows, x.Cols, s.Rows, s.Cols)
		}
		for i, v := range x.Data {
			s.Data[i] += v
		}
	}
	return s, nil
}

func rowBlock(m DistMatrix, r, p int) (DistMatrix, error) {
	if m.Rows%p != 0 {
		return DistMatrix{}, fmt.Errorf("dist eval: %d rows do not split over %d ranks", m.Rows, p)
	}
	n := m.Rows / p
	return DistMatrix{n, m.Cols, m.Data[r*n*m.Cols : (r+1)*n*m.Cols]}, nil
}

func allGather(xs []DistMatrix) DistMatrix {
	g := DistMatrix{Cols: xs[0].Cols}
	for _, x := range xs {
		g.Rows += x.Rows
		g.Data = append(g.Data, x.Data...)
	}
	return g
}

func addMat(a, b DistMatrix) (DistMatrix, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return DistMatrix{}, fmt.Errorf("dist eval: add %dx%d and %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	c := newDistMatrix(a.Rows, a.Cols)
	for i := range c.Data {
		c.Data[i] = a.Data[i] + b.Data[i]
	}
	return c, nil
}

func rmsNorm(x, g DistMatrix, eps float32) (DistMatrix, error) {
	if g.Rows != 1 || g.Cols != x.Cols {
		return DistMatrix{}, fmt.Errorf("dist eval: rmsnorm weight %dx%d for %d columns", g.Rows, g.Cols, x.Cols)
	}
	y := newDistMatrix(x.Rows, x.Cols)
	for i := range x.Rows {
		normRow(y.Data[i*x.Cols:(i+1)*x.Cols], x.Data[i*x.Cols:(i+1)*x.Cols], g.Data, eps)
	}
	return y, nil
}

func normRow(dst, h, g []float32, eps float32) {
	var sq float64
	for _, v := range h {
		sq += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(sq/float64(len(h))+float64(eps)))
	for j, v := range h {
		dst[j] = v * inv * g[j]
	}
}

// tilePush computes each output tile on every rank, sums the tiles across
// ranks as the switch would, and adds the residual tile.
func (d distEval) tilePush(x, w, res []DistMatrix) ([]DistMatrix, error) {
	P := d.cfg.Ranks
	M, N := x[0].Rows, w[0].Cols
	for r := range P {
		if x[r].Cols != w[r].Rows || x[r].Rows != M || w[r].Cols != N || res[r].Rows != M || res[r].Cols != N {
			return nil, fmt.Errorf("dist eval: fused push shapes on rank %d", r)
		}
	}
	out := make([]DistMatrix, P)
	for r := range P {
		out[r] = newDistMatrix(M, N)
	}
	for i0 := 0; i0 < M; i0 += d.cfg.TileM {
		for j0 := 0; j0 < N; j0 += d.cfg.TileN {
			for i := i0; i < min(i0+d.cfg.TileM, M); i++ {
				for j := j0; j < min(j0+d.cfg.TileN, N); j++ {
					var s float32
					for r := range P {
						s += dotRowCol(x[r], w[r], i, j)
					}
					for r := range P {
						out[r].Data[i*N+j] = s + res[r].at(i, j)
					}
				}
			}
		}
	}
	return out, nil
}

// pullNorm pulls each reduced row, adds the residual and normalizes, one
// row at a time.
func (d distEval) pullNorm(p, res, g []DistMatrix) ([]DistMatrix, error) {
	P := d.cfg.Ranks
	M, N := p[0].Rows, p[0].Cols
	out := make([]DistMatrix, P)
	h := make([]float32, N)
	for r := range P {
		if res[r].Rows != M || res[r].Cols != N || g[r].Rows != 1 || g[r].Cols != N {
			return nil, fmt.Errorf("dist eval: fused pull shapes on rank %d", r)
		}
		out[r] = newDistMatrix(M, N)
		for i := range M {
			for j := range N {
				var s float32
				for q := range P {
					s += p[q].at(i, j)
				}
				h[j] = s + res[r].at(i, j)
			}
			normRow(out[r].Data[i*N:(i+1)*N], h, g[r].Data, d.cfg.Eps)
		}
	}
	return out, nil
}

// ringGemm multiplies the row blocks of every rank's x in ring order
// starting from the local block, as a pipelined fabric get would.
func (d distEval) ringGemm(x, w []DistMatrix) ([]DistMatrix, error) {
	P := d.cfg.Ranks
	rows := x[0].Rows
	out := make([]DistMatrix, P)
	for r := range P {
		out[r] = newDistMatrix(rows*P, w[r].Cols)
		for s := range P {
			src := (r + s) % P
			blk, err := gemm(x[src], w[r])
			if err != nil {
				return nil, err
			}
			copy(out[r].Data[src*rows*w[r].Cols:], blk.Data)
		}
	}
	return out, nil
}

// scatterGemm produces row block b on every rank and reduces it straight
// to its owner b.
func (d distEval) scatterGemm(x, w []DistMatrix) ([]DistMatrix, error) {
	P := d.cfg.Ranks
	M, N := x[0].Rows, w[0].Cols
	if M%P != 0 {
		return nil, fmt.Errorf("dist eval: %d rows do not split over %d ranks", M, P)
	}
	n := M / P
	out := make([]DistMatrix, P)
	for b := range P {
		out[b] = newDistMatrix(n, N)
		for i := range n {
			for j := range N {
				var s float32
				for r := range P {
					s += dotRowCol(x[r], w[r], b*n+i, j)
				}
				out[b].Data[i*N+j] = s
			}
		}
	}
	return out, nil
}

// PatternExpr converts a rewrite pattern to an expression whose leaves are
// the pattern variables, named without the leading '?', for evaluating a
// rule's two sides with [EvalDistExpr].
func PatternExpr(p *egraph.Pattern) *egraph.Expr {
	e := &egraph.Expr{Op: strings.TrimPrefix(p.Op, "?")}
	for _, c := range p.Children {
		e.Children = append(e.Children, PatternExpr(c))
	}
	return e
}

// MultimemRMSNormKernel emits the pulled form of RulePullMultimemLdReduce:
// one CTA per row; each thread loads 8 BF16 values per step with
// multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 (sm_90 and later),
// adds the BF16 residual, and the CTA applies RMSNorm with weight w,
// writing FP32. Under cudahostemu the reduction reads every rank's slice
// of the symmetric buffer and rounds the sum to BF16, as the switch does.
// The NVFP4 quantization the spec lists after the norm is not emitted.
type MultimemRMSNormKernel struct {
	Name    string
	Cols    int // multiple of 8
	Ranks   int
	Eps     float32
	Threads int // default 128
}

func (k MultimemRMSNormKernel) norm() (MultimemRMSNormKernel, error) {
	if k.Name == "" {
		k.Name = "nvl72_multimem_rmsnorm"
	}
	if k.Eps == 0 {
		k.Eps = 1e-6
	}
	if k.Threads == 0 {
		k.Threads = 128
	}
	switch {
	case !cIdent(k.Name):
		return k, fmt.Errorf("multimem rmsnorm: bad name %q", k.Name)
	case k.Cols < 8 || k.Cols%8 != 0:
		return k, fmt.Errorf("multimem rmsnorm: Cols %d must be a positive multiple of 8", k.Cols)
	case k.Ranks < 1:
		return k, fmt.Errorf("multimem rmsnorm: Ranks %d must be positive", k.Ranks)
	case k.Threads < 32 || k.Threads > 1024 || k.Threads%32 != 0:
		return k, fmt.Errorf("multimem rmsnorm: Threads %d must be a multiple of 32 in 32..1024", k.Threads)
	}
	return k, nil
}

// Grid returns one CTA per row.
func (k MultimemRMSNormKernel) Grid(rows int) (grid, block [3]int) {
	t := k.Threads
	if t == 0 {
		t = 128
	}
	return [3]int{rows, 1, 1}, [3]int{t, 1, 1}
}

// CUDA returns the kernel without the preludes. Parameters: rows, the
// symmetric BF16 partials sym_in, BF16 residual and weight, FP32 out.
func (k MultimemRMSNormKernel) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank multimem.ld_reduce.v4.bf16x2 + residual + RMSNorm over %d columns.\n", k.Name, k.Ranks, k.Cols)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(int rows, const uint16_t* __restrict__ sym_in,\n", k.Threads, k.Name)
	b.WriteString("    const uint16_t* __restrict__ residual, const uint16_t* __restrict__ weight, float* __restrict__ out) {\n")
	fmt.Fprintf(&b, "    constexpr int COLS = %d, VEC = 8, PER = (COLS / VEC + %d - 1) / %d;\n", k.Cols, k.Threads, k.Threads)
	fmt.Fprintf(&b, "    constexpr float EPS = %s;\n", floatLiteral(k.Eps))
	b.WriteString(`    const int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    __shared__ float eg_red[32];
    const size_t n = (size_t)rows * COLS;
    const uint16_t* lres = eg_nvlink_local_ptr(residual, n);
    const uint16_t* lw = eg_nvlink_local_ptr(weight, (size_t)COLS);
    float* lout = eg_nvlink_local_ptr(out, n);
    float h[PER][VEC];
    float sq = 0.0f;
    for (int j = 0; j < PER; ++j) {
        const int c0 = (tid + j * (int)blockDim.x) * VEC;
        if (c0 >= COLS) break;
        const size_t idx = (size_t)row * COLS + c0;
        uint32_t r4[4];
#if defined(EG_HOST_EMU) || defined(EG_EMU)
        for (int v = 0; v < 4; ++v) {
            const uint32_t lo = eg_emu_multimem_ld_reduce_add_bf16(sym_in, idx + 2 * v).b;
            const uint32_t hi = eg_emu_multimem_ld_reduce_add_bf16(sym_in, idx + 2 * v + 1).b;
            r4[v] = lo | (hi << 16);
        }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r4[0]), "=r"(r4[1]), "=r"(r4[2]), "=r"(r4[3]) : "l"(sym_in + idx) : "memory");
#else
        for (int v = 0; v < 4; ++v) r4[v] = (uint32_t)sym_in[idx + 2 * v] | ((uint32_t)sym_in[idx + 2 * v + 1] << 16);
#endif
        for (int v = 0; v < 4; ++v) {
            const float lo = eg_bf16_to_f32(r4[v]) + eg_bf16_to_f32(lres[idx + 2 * v]);
            const float hi = eg_bf16_hi_to_f32(r4[v]) + eg_bf16_to_f32(lres[idx + 2 * v + 1]);
            h[j][2 * v] = lo;
            h[j][2 * v + 1] = hi;
            sq += lo * lo;
            sq += hi * hi;
        }
    }
    sq = eg_warp_sum(sq);
    if (blockDim.x > 32) sq = eg_block_combine<0>(sq, eg_red);
    const float inv = rsqrtf(sq / (float)COLS + EPS);
    for (int j = 0; j < PER; ++j) {
        const int c0 = (tid + j * (int)blockDim.x) * VEC;
        if (c0 >= COLS) break;
        const size_t idx = (size_t)row * COLS + c0;
        for (int v = 0; v < VEC; ++v) lout[idx + v] = h[j][v] * inv * eg_bf16_to_f32(lw[c0 + v]);
    }
}
`)
	return b.String(), nil
}

// Source returns a complete translation unit with [Prelude],
// [CollectivePrelude] and the kernel.
func (k MultimemRMSNormKernel) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + CollectivePrelude + "\n" + src, nil
}
