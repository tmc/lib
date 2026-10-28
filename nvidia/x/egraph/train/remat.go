package train

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// ActivationPolicy says how a forward activation reaches the backward pass.
type ActivationPolicy uint8

const (
	// PolicyStoreHBM keeps the activation in GPU HBM from forward to backward.
	PolicyStoreHBM ActivationPolicy = iota
	// PolicyRematInTMEM discards the activation after forward and recomputes it
	// inline (in TMEM or registers) in the backward tile prologue.
	PolicyRematInTMEM
	// PolicyOffloadC2CAsync copies the activation to Grace/Vera LPDDR5X over
	// NVLink-C2C during forward and prefetches it back one layer ahead of
	// backward through a two-layer HBM staging ring.
	PolicyOffloadC2CAsync
)

func (p ActivationPolicy) String() string {
	switch p {
	case PolicyStoreHBM:
		return "store_hbm"
	case PolicyRematInTMEM:
		return "remat_tmem"
	case PolicyOffloadC2CAsync:
		return "offload_c2c"
	}
	return fmt.Sprintf("ActivationPolicy(%d)", uint8(p))
}

// ActivationNode is one forward activation of a single layer. Every layer of
// the model holds the same set of nodes.
type ActivationNode struct {
	Name         string
	OpKind       string  // e.g. "RMSNorm", "RoPE", "SwiGLU", "AttnLSE", "QKVProj"
	Bytes        int64   // size of the activation for one layer
	RecomputeUs  float64 // backward time to recompute it inline
	CanRematTMEM bool    // whether it can be recomputed without materializing it in HBM
}

// RematConstraints are the budgets for [SolveJointRematC2C].
type RematConstraints struct {
	MaxHBMBytes       int64   // HBM activation budget across all layers
	MaxC2CHostBytes   int64   // Grace/Vera LPDDR5X capacity for offloaded activations
	C2CBandwidthGBps  float64 // unidirectional C2C bandwidth; 0 forbids offload
	LayerFwdComputeUs float64 // forward compute per layer that hides offload (D2H)
	LayerBwdComputeUs float64 // backward compute per layer that hides prefetch (H2D)
	NumLayers         int
}

// RematPlan is the policy chosen for each activation and its cost.
//
// PeakHBMBytes counts stored activations in every layer plus a two-layer HBM
// staging ring for offloaded activations. RecomputeOverheadUs,
// UnexposedC2CUs and ExposedC2CStallUs are per layer; ExposedC2CStallUs adds
// the forward (D2H) and backward (H2D) stalls.
type RematPlan struct {
	Decisions            map[string]ActivationPolicy
	PeakHBMBytes         int64
	PeakC2CHostBytes     int64
	OffloadBytesPerLayer int64
	RecomputeOverheadUs  float64
	UnexposedC2CUs       float64
	ExposedC2CStallUs    float64
}

// RecomputeFraction returns the per-layer recompute time as a fraction of
// the per-layer forward plus backward compute in c.
func (p RematPlan) RecomputeFraction(c RematConstraints) float64 {
	total := c.LayerFwdComputeUs + c.LayerBwdComputeUs
	if total <= 0 {
		return 0
	}
	return p.RecomputeOverheadUs / total
}

// Errors returned by [SolveJointRematC2C].
var (
	ErrHBMBudgetExceeded     = errors.New("hbm activation budget exceeded")
	ErrC2CHostBudgetExceeded = errors.New("c2c host memory budget exceeded")
)

// maxRematNodes bounds the exact search: the solver visits at most 3^n
// assignments.
const maxRematNodes = 16

// c2cStagingLayers is the depth of the HBM ring that holds offloaded
// activations in flight: one layer being prefetched while the previous one
// is consumed.
const c2cStagingLayers = 2

// SolveJointRematC2C chooses a policy for each activation of a layer.
//
// Each activation is an e-class with up to three e-nodes (store, remat,
// offload); the solver extracts the assignment that minimizes per-layer
// recompute plus exposed C2C stall, subject to MaxHBMBytes and
// MaxC2CHostBytes. Ties prefer fewer offloaded bytes, then less HBM. The
// search is exact (branch and bound over all 3^n assignments) and accepts at
// most 16 nodes per layer.
//
// If no assignment fits, SolveJointRematC2C returns an error wrapping
// [ErrHBMBudgetExceeded] or [ErrC2CHostBudgetExceeded].
func SolveJointRematC2C(nodes []ActivationNode, c RematConstraints) (RematPlan, error) {
	if err := checkRemat(nodes, c); err != nil {
		return RematPlan{}, err
	}
	L := int64(c.NumLayers)
	// bytesPerUs is the C2C bandwidth in bytes per microsecond.
	bytesPerUs := c.C2CBandwidthGBps * 1e3

	// Visit large activations first so the HBM bound prunes early.
	order := make([]int, len(nodes))
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int {
		switch {
		case nodes[a].Bytes > nodes[b].Bytes:
			return -1
		case nodes[a].Bytes < nodes[b].Bytes:
			return 1
		}
		return 0
	})
	// minHBMRest[i] is the least HBM the nodes order[i:] can occupy.
	minHBMRest := make([]int64, len(order)+1)
	for i := len(order) - 1; i >= 0; i-- {
		n := nodes[order[i]]
		least := L * n.Bytes
		if n.CanRematTMEM {
			least = 0
		} else if bytesPerUs > 0 {
			least = min(least, c2cStagingLayers*n.Bytes)
		}
		minHBMRest[i] = minHBMRest[i+1] + least
	}

	var (
		cur      = make([]ActivationPolicy, len(nodes))
		best     []ActivationPolicy
		bestCost = math.Inf(1)
		bestOff  int64
		bestHBM  int64
		sawHBM   bool // some complete assignment fit HBM but not C2C host memory
	)
	cost := func(recompute float64, offload int64) float64 {
		if offload == 0 {
			return recompute
		}
		t := float64(offload) / bytesPerUs
		return recompute + max(0, t-c.LayerFwdComputeUs) + max(0, t-c.LayerBwdComputeUs)
	}
	var search func(i int, hbm, offload int64, recompute float64)
	search = func(i int, hbm, offload int64, recompute float64) {
		if hbm+minHBMRest[i] > c.MaxHBMBytes {
			return
		}
		// Recompute only grows and stall only grows with more offload, so
		// the current partial cost is a lower bound.
		if cost(recompute, offload) > bestCost {
			return
		}
		if i == len(order) {
			if L*offload > c.MaxC2CHostBytes {
				sawHBM = true
				return
			}
			k := cost(recompute, offload)
			if k < bestCost || k == bestCost && (offload < bestOff || offload == bestOff && hbm < bestHBM) {
				bestCost, bestOff, bestHBM = k, offload, hbm
				best = slices.Clone(cur)
			}
			return
		}
		j := order[i]
		n := nodes[j]
		cur[j] = PolicyStoreHBM
		search(i+1, hbm+L*n.Bytes, offload, recompute)
		if n.CanRematTMEM {
			cur[j] = PolicyRematInTMEM
			search(i+1, hbm, offload, recompute+n.RecomputeUs)
		}
		if bytesPerUs > 0 {
			cur[j] = PolicyOffloadC2CAsync
			// The staging ring grows with the per-layer offload, so charge
			// it here rather than at the leaf.
			search(i+1, hbm+c2cStagingLayers*n.Bytes, offload+n.Bytes, recompute)
		}
	}
	search(0, 0, 0, 0)
	if best == nil {
		if sawHBM {
			return RematPlan{}, fmt.Errorf("solve remat: %w: offload needs more than %d bytes", ErrC2CHostBudgetExceeded, c.MaxC2CHostBytes)
		}
		return RematPlan{}, fmt.Errorf("solve remat: %w: least HBM is %d bytes, budget %d", ErrHBMBudgetExceeded, minHBMRest[0], c.MaxHBMBytes)
	}
	return evalRemat(nodes, best, c), nil
}

func checkRemat(nodes []ActivationNode, c RematConstraints) error {
	if len(nodes) == 0 {
		return errors.New("solve remat: no activations")
	}
	if len(nodes) > maxRematNodes {
		return fmt.Errorf("solve remat: %d activations, at most %d per layer", len(nodes), maxRematNodes)
	}
	if c.NumLayers <= 0 {
		return fmt.Errorf("solve remat: NumLayers %d must be > 0", c.NumLayers)
	}
	if c.MaxHBMBytes < 0 || c.MaxC2CHostBytes < 0 {
		return errors.New("solve remat: negative memory budget")
	}
	if c.C2CBandwidthGBps < 0 || c.LayerFwdComputeUs < 0 || c.LayerBwdComputeUs < 0 {
		return errors.New("solve remat: negative bandwidth or compute window")
	}
	seen := make(map[string]bool, len(nodes))
	for _, n := range nodes {
		if n.Name == "" {
			return errors.New("solve remat: activation with empty name")
		}
		if seen[n.Name] {
			return fmt.Errorf("solve remat: duplicate activation %q", n.Name)
		}
		seen[n.Name] = true
		if n.Bytes <= 0 {
			return fmt.Errorf("solve remat: activation %q has %d bytes", n.Name, n.Bytes)
		}
		if n.RecomputeUs < 0 {
			return fmt.Errorf("solve remat: activation %q has negative recompute time", n.Name)
		}
	}
	return nil
}

// evalRemat computes the cost of a policy assignment.
func evalRemat(nodes []ActivationNode, pol []ActivationPolicy, c RematConstraints) RematPlan {
	L := int64(c.NumLayers)
	p := RematPlan{Decisions: make(map[string]ActivationPolicy, len(nodes))}
	var stored int64
	for i, n := range nodes {
		p.Decisions[n.Name] = pol[i]
		switch pol[i] {
		case PolicyStoreHBM:
			stored += n.Bytes
		case PolicyRematInTMEM:
			p.RecomputeOverheadUs += n.RecomputeUs
		case PolicyOffloadC2CAsync:
			p.OffloadBytesPerLayer += n.Bytes
		}
	}
	p.PeakHBMBytes = L*stored + c2cStagingLayers*p.OffloadBytesPerLayer
	p.PeakC2CHostBytes = L * p.OffloadBytesPerLayer
	if p.OffloadBytesPerLayer > 0 {
		t := float64(p.OffloadBytesPerLayer) / (c.C2CBandwidthGBps * 1e3)
		fwd := max(0, t-c.LayerFwdComputeUs)
		bwd := max(0, t-c.LayerBwdComputeUs)
		p.ExposedC2CStallUs = fwd + bwd
		p.UnexposedC2CUs = 2*t - p.ExposedC2CStallUs
	}
	return p
}

// BlockShape describes one transformer decoder layer for
// [TransformerBlockActivations].
type BlockShape struct {
	Tokens       int64   // tokens per microbatch (batch * sequence length)
	Hidden       int64   // model dimension
	FFN          int64   // SwiGLU intermediate dimension
	Heads        int64   // query heads
	KVHeads      int64   // key/value heads
	HeadDim      int64   // per-head dimension
	ElemBytes    int64   // activation element size (2 for BF16)
	HBMGBps      float64 // HBM bandwidth that bounds elementwise recompute
	TensorTFLOPS float64 // dense tensor-core throughput for GEMM recompute
}

// TransformerBlockActivations lists the forward activations a decoder layer
// saves for backward, with sizes and recompute estimates.
//
// Elementwise and normalization outputs (RMSNorm, RoPE, SwiGLU) can be
// recomputed inline from their saved inputs; their recompute time is the
// bytes they read and write at HBMGBps. GEMM outputs and attention outputs
// are marked as not rematerializable in TMEM, since recomputing them repeats
// a projection or the attention forward pass.
func TransformerBlockActivations(s BlockShape) ([]ActivationNode, error) {
	if s.Tokens <= 0 || s.Hidden <= 0 || s.FFN <= 0 || s.Heads <= 0 || s.KVHeads <= 0 || s.HeadDim <= 0 || s.ElemBytes <= 0 {
		return nil, fmt.Errorf("transformer activations: non-positive shape %+v", s)
	}
	if s.HBMGBps <= 0 {
		return nil, errors.New("transformer activations: HBMGBps must be > 0")
	}
	T, e := s.Tokens, s.ElemBytes
	q := T * s.Heads * s.HeadDim * e
	kv := T * s.KVHeads * s.HeadDim * e
	h := T * s.Hidden * e
	f := T * s.FFN * e
	// us returns the time to move n bytes at HBM bandwidth.
	us := func(n int64) float64 { return float64(n) / (s.HBMGBps * 1e3) }
	return []ActivationNode{
		{Name: "attn_in", OpKind: "Residual", Bytes: h},
		{Name: "attn_norm", OpKind: "RMSNorm", Bytes: h, RecomputeUs: us(2 * h), CanRematTMEM: true},
		{Name: "q_proj", OpKind: "QKVProj", Bytes: q},
		{Name: "k_proj", OpKind: "QKVProj", Bytes: kv},
		{Name: "v_proj", OpKind: "QKVProj", Bytes: kv},
		{Name: "qk_rope", OpKind: "RoPE", Bytes: q + kv, RecomputeUs: us(2 * (q + kv)), CanRematTMEM: true},
		{Name: "attn_out", OpKind: "AttnOut", Bytes: q},
		{Name: "attn_lse", OpKind: "AttnLSE", Bytes: T * s.Heads * 4},
		{Name: "mlp_in", OpKind: "Residual", Bytes: h},
		{Name: "mlp_norm", OpKind: "RMSNorm", Bytes: h, RecomputeUs: us(2 * h), CanRematTMEM: true},
		{Name: "gate_proj", OpKind: "LinearIn", Bytes: f},
		{Name: "up_proj", OpKind: "LinearIn", Bytes: f},
		{Name: "swiglu", OpKind: "SwiGLU", Bytes: f, RecomputeUs: us(3 * f), CanRematTMEM: true},
	}, nil
}

// C2CStreamOp is one step of the C2C offload and prefetch stream schedule
// produced by [RematPlan.PrefetchSchedule].
//
// Ops run in slice order on their Stream. An op starts only after every
// event in Wait has been recorded, and records Signal when it finishes.
type C2CStreamOp struct {
	Stream string // "compute" or "c2c"
	Kind   string // "fwd", "offload", "prefetch", "bwd"
	Layer  int
	Slot   int // staging ring slot for offload and prefetch; -1 otherwise
	Wait   []string
	Signal string
}

// PrefetchSchedule returns the double-buffered stream schedule for p over
// numLayers layers. Forward layer l signals its staging slot, the C2C stream
// copies it to host memory, and the slot is reused two layers later only
// after that copy finishes. In backward, layer l waits for its prefetch, and
// the prefetch of layer l-1 is issued as soon as the slot it reuses is free,
// so it overlaps backward layer l.
//
// If p offloads nothing, the schedule holds only compute ops.
func (p RematPlan) PrefetchSchedule(numLayers int) ([]C2CStreamOp, error) {
	if numLayers <= 0 {
		return nil, fmt.Errorf("prefetch schedule: numLayers %d must be > 0", numLayers)
	}
	offload := p.OffloadBytesPerLayer > 0
	var ops []C2CStreamOp
	ev := func(kind string, l int) string { return fmt.Sprintf("%s%d", kind, l) }
	for l := range numLayers {
		op := C2CStreamOp{Stream: "compute", Kind: "fwd", Layer: l, Slot: -1, Signal: ev("fwd", l)}
		if offload && l >= c2cStagingLayers {
			// Slot l%2 still holds layer l-2 until its offload completes.
			op.Wait = []string{ev("offload", l-c2cStagingLayers)}
		}
		ops = append(ops, op)
		if offload {
			ops = append(ops, C2CStreamOp{Stream: "c2c", Kind: "offload", Layer: l, Slot: l % c2cStagingLayers,
				Wait: []string{ev("fwd", l)}, Signal: ev("offload", l)})
		}
	}
	prefetch := func(l int) C2CStreamOp {
		op := C2CStreamOp{Stream: "c2c", Kind: "prefetch", Layer: l, Slot: l % c2cStagingLayers, Signal: ev("prefetch", l)}
		op.Wait = []string{ev("offload", l)}
		if l+c2cStagingLayers < numLayers {
			// The slot is free once backward of layer l+2 has consumed it.
			op.Wait = append(op.Wait, ev("bwd", l+c2cStagingLayers))
		}
		return op
	}
	if offload {
		ops = append(ops, prefetch(numLayers-1))
	}
	for l := numLayers - 1; l >= 0; l-- {
		if offload && l > 0 {
			ops = append(ops, prefetch(l-1))
		}
		op := C2CStreamOp{Stream: "compute", Kind: "bwd", Layer: l, Slot: -1, Signal: ev("bwd", l)}
		if offload {
			op.Wait = []string{ev("prefetch", l)}
		}
		if l+1 < numLayers {
			op.Wait = append(op.Wait, ev("bwd", l+1))
		}
		ops = append(ops, op)
	}
	return ops, nil
}

// VerifyPrefetchSchedule checks that ops execute safely under every
// interleaving of their streams. It builds the happens-before order (stream
// order plus event waits) and checks that it is acyclic, that each
// offload follows its forward layer, that a staging slot is overwritten only
// after its previous occupant has been offloaded (forward) or consumed by
// backward, and that each backward layer follows its own prefetch.
func VerifyPrefetchSchedule(ops []C2CStreamOp, numLayers int) error {
	idx := make(map[string]int, len(ops))
	for i, op := range ops {
		if op.Signal == "" {
			return fmt.Errorf("verify prefetch: op %d (%s layer %d) has no signal", i, op.Kind, op.Layer)
		}
		if _, dup := idx[op.Signal]; dup {
			return fmt.Errorf("verify prefetch: event %q signalled twice", op.Signal)
		}
		idx[op.Signal] = i
	}
	// pred[i] lists the ops that must finish before op i starts.
	pred := make([][]int, len(ops))
	last := make(map[string]int)
	for i, op := range ops {
		if j, ok := last[op.Stream]; ok {
			pred[i] = append(pred[i], j)
		}
		last[op.Stream] = i
		for _, w := range op.Wait {
			j, ok := idx[w]
			if !ok {
				return fmt.Errorf("verify prefetch: %s layer %d waits on unknown event %q", op.Kind, op.Layer, w)
			}
			pred[i] = append(pred[i], j)
		}
	}
	// before[i] is the set of ops that happen before op i.
	before := make([]map[int]bool, len(ops))
	state := make([]uint8, len(ops)) // 0 unvisited, 1 visiting, 2 done
	var visit func(i int) error
	visit = func(i int) error {
		switch state[i] {
		case 1:
			return fmt.Errorf("verify prefetch: deadlock through %s layer %d", ops[i].Kind, ops[i].Layer)
		case 2:
			return nil
		}
		state[i] = 1
		before[i] = make(map[int]bool)
		for _, j := range pred[i] {
			if err := visit(j); err != nil {
				return err
			}
			before[i][j] = true
			for k := range before[j] {
				before[i][k] = true
			}
		}
		state[i] = 2
		return nil
	}
	for i := range ops {
		if err := visit(i); err != nil {
			return err
		}
	}
	find := func(kind string, l int) (int, bool) {
		i, ok := idx[fmt.Sprintf("%s%d", kind, l)]
		return i, ok && ops[i].Kind == kind
	}
	hb := func(a string, la int, b string, lb int) error {
		i, ok := find(a, la)
		if !ok {
			return fmt.Errorf("verify prefetch: missing %s layer %d", a, la)
		}
		j, ok := find(b, lb)
		if !ok {
			return fmt.Errorf("verify prefetch: missing %s layer %d", b, lb)
		}
		if !before[j][i] {
			return fmt.Errorf("verify prefetch: %s layer %d is not ordered before %s layer %d", a, la, b, lb)
		}
		return nil
	}
	offload := false
	for _, op := range ops {
		if op.Kind == "offload" {
			offload = true
		}
	}
	for l := range numLayers {
		if _, ok := find("fwd", l); !ok {
			return fmt.Errorf("verify prefetch: missing fwd layer %d", l)
		}
		if l+1 < numLayers {
			if err := hb("fwd", l, "fwd", l+1); err != nil {
				return err
			}
			if err := hb("bwd", l+1, "bwd", l); err != nil {
				return err
			}
		}
		if err := hb("fwd", numLayers-1, "bwd", l); err != nil {
			return err
		}
		if !offload {
			continue
		}
		for _, e := range [][2]string{{"fwd", "offload"}, {"offload", "prefetch"}, {"prefetch", "bwd"}} {
			if err := hb(e[0], l, e[1], l); err != nil {
				return err
			}
		}
		if l+c2cStagingLayers < numLayers {
			// Slot reuse in forward and in backward.
			if err := hb("offload", l, "fwd", l+c2cStagingLayers); err != nil {
				return err
			}
			if err := hb("bwd", l+c2cStagingLayers, "prefetch", l); err != nil {
				return err
			}
		}
	}
	return nil
}
