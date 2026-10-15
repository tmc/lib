package egraph

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// AttentionCandidatePolicy opts into an absolute/relative output tolerance.
// The zero value requires identical FP32 bits, including the sign of zero.
// A nonzero policy permits numerical differences only for the bound inputs.
type AttentionCandidatePolicy struct {
	Absolute, Relative float64
}

// AttentionCandidateTile records the state after one provided online-row rule
// step. Output is the unnormalized FP64 accumulator, not the final FP32 result.
type AttentionCandidateTile struct {
	Start, End         int
	Maximum            float64
	Normalizer         float64
	UnnormalizedOutput []float64
}

// AttentionCandidateDerivation describes a row candidate derived from
// one checked graph. It does not assert an equality or authorize graph mutation.
type AttentionCandidateDerivation struct {
	Rule                       string
	ScoreID, ValueID, OutputID int
	NodeIDs                    []int
	TileSize                   int
	Policy                     AttentionCandidatePolicy
	Tiles                      []AttentionCandidateTile
	// Mask is nil for the dense constructor. Otherwise it records the
	// predicate proved by the retained score input, without changing it.
	Mask *AttentionRowMask
	// FullyMaskedPolicy is "canonical-nan-exact-fp32" for masked candidates.
	// NaN output bits must match the unchanged graph on this host.
	FullyMaskedPolicy string
}

// OnlineAttentionCandidate owns one checked, input-bound CPU candidate.
// Its zero value is not usable. All returned slices and maps are detached.
// It carries no compiler, device, held-out, timing or promotion qualification.
type OnlineAttentionCandidate struct {
	snapshot   *cudair.Snapshot
	inputs     map[string][]float32
	output     []float32
	reference  []float32
	derivation AttentionCandidateDerivation
	identity   string
}

// NewCheckedOnlineAttentionCandidate recognizes exactly this readonly FP32 graph:
//
//	e = exp(scores - max(scores, axis=0))
//	p = e / sum(e, axis=0)
//	out = matmul(reshape(p, [1,N]), values[N,D])
//
// Scores must be a contiguous rank-one input and values a contiguous rank-two
// input. N and D are in [1,64], tileSize in [1,N], and the graph has exactly
// these nine nodes and one output. Inputs and reference/candidate outputs must
// be finite. Effects other than the checked reshape alias are unsupported.
// Masks, score generation/scaling and additional observable outputs are refused.
//
// The original snapshot is evaluated independently by cudaref. The existing
// FP64 online-row rule is returned only when its output satisfies policy for
// these copied inputs. The FP32 graph and FP64 rule need not be bitwise equal;
// the default exact policy refuses differing outputs. This function never
// inserts a rewrite, unions e-classes or replaces the original graph.
func NewCheckedOnlineAttentionCandidate(snapshot *cudair.Snapshot, inputs map[string][]float32, tileSize int, policy AttentionCandidatePolicy) (*OnlineAttentionCandidate, error) {
	return newCheckedOnlineAttentionCandidate(snapshot, inputs, tileSize, policy, nil)
}

func newCheckedOnlineAttentionCandidate(snapshot *cudair.Snapshot, inputs map[string][]float32, tileSize int, policy AttentionCandidatePolicy, mask *AttentionRowMask) (*OnlineAttentionCandidate, error) {
	if policy.Absolute < 0 || policy.Relative < 0 || !finite64(policy.Absolute) || !finite64(policy.Relative) {
		return nil, errors.New("egraph: invalid attention candidate tolerance")
	}
	scores, values, output, ids, err := checkedAttentionRegion(snapshot)
	if err != nil {
		return nil, err
	}
	n, d := scores.Shape[0], values.Shape[1]
	var excluded []bool
	if mask != nil {
		if len(mask.excluded) != n || mask.kind == "" {
			return nil, errors.New("egraph: invalid attention candidate mask")
		}
		excluded = mask.excluded
	}
	if tileSize < 1 || tileSize > n {
		return nil, errors.New("egraph: invalid attention candidate tile size")
	}
	if len(inputs) != 2 {
		return nil, errors.New("egraph: attention candidate needs exactly two input bindings")
	}
	bound := make(map[string][]float32, 2)
	for _, node := range []*cudair.Node{scores, values} {
		data, ok := inputs[node.Name]
		if !ok || len(data) != node.NumElements() {
			return nil, fmt.Errorf("egraph: invalid attention candidate input %q", node.Name)
		}
		for i, value := range data {
			if node.ID == scores.ID && excluded != nil && excluded[i] {
				if !math.IsInf(float64(value), -1) {
					return nil, fmt.Errorf("egraph: excluded attention score %d must already be negative infinity", i)
				}
				continue
			}
			if !finite64(float64(value)) {
				return nil, fmt.Errorf("egraph: nonfinite attention candidate input %q", node.Name)
			}
		}
		bound[node.Name] = slices.Clone(data)
	}
	refs, err := cudaref.EvaluateSnapshot(snapshot, bound)
	if err != nil {
		return nil, fmt.Errorf("egraph: attention candidate reference: %w", err)
	}
	reference := refs[output.Name]
	fullyMasked := excluded != nil && !slices.Contains(excluded, false)
	state := NewOnlineAttentionState(d)
	derivation := AttentionCandidateDerivation{
		Rule: "provided-stable-online-row-v1", ScoreID: scores.ID, ValueID: values.ID,
		OutputID: output.ID, NodeIDs: ids, TileSize: tileSize, Policy: policy,
	}
	if mask != nil {
		owned := mask.clone()
		derivation.Mask = &owned
		derivation.Rule = "provided-input-masked-stable-online-row-v1"
		derivation.FullyMaskedPolicy = "canonical-nan-exact-fp32"
	}
	for start := 0; start < n; start += tileSize {
		end := min(start+tileSize, n)
		var tileMask []bool
		if excluded != nil {
			tileMask = excluded[start:end]
		}
		if err := state.StepTile(bound[scores.Name][start:end], bound[values.Name][start*d:end*d], tileMask, d); err != nil {
			return nil, fmt.Errorf("egraph: attention candidate tile: %w", err)
		}
		derivation.Tiles = append(derivation.Tiles, AttentionCandidateTile{
			Start: start, End: end, Maximum: state.M, Normalizer: state.D,
			UnnormalizedOutput: slices.Clone(state.O),
		})
	}
	candidate := state.Finalize()
	if fullyMasked {
		// This graph's all-excluded softmax has no finite denominator.
		// Emit our declared NaN independently of the reference. Admission
		// below requires bit agreement, including the NaN sign/payload.
		for i := range candidate {
			candidate[i] = math.Float32frombits(0x7fc00000)
		}
	}
	if len(reference) != d || len(candidate) != d {
		return nil, errors.New("egraph: attention candidate output extent mismatch")
	}
	for i, got := range candidate {
		want := reference[i]
		if fullyMasked {
			if math.Float32bits(got) != math.Float32bits(want) {
				return nil, fmt.Errorf("egraph: fully masked attention nan bits differ from source at %d", i)
			}
			continue
		}
		if !finite64(float64(got)) || !finite64(float64(want)) {
			return nil, errors.New("egraph: nonfinite attention candidate output")
		}
		if policy == (AttentionCandidatePolicy{}) {
			if math.Float32bits(got) != math.Float32bits(want) {
				return nil, fmt.Errorf("egraph: attention candidate differs from exact reference at %d", i)
			}
			continue
		}
		tolerance := policy.Absolute + policy.Relative*math.Abs(float64(want))
		if !finite64(tolerance) || math.Abs(float64(got)-float64(want)) > tolerance {
			return nil, fmt.Errorf("egraph: attention candidate exceeds tolerance at %d", i)
		}
	}
	c := &OnlineAttentionCandidate{snapshot: snapshot, inputs: bound, output: candidate,
		reference: slices.Clone(reference), derivation: derivation}
	c.identity = attentionCandidateIdentity(c, scores, values)
	return c, nil
}

func finite64(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func checkedAttentionRegion(snapshot *cudair.Snapshot) (scores, values, output *cudair.Node, ids []int, err error) {
	unsupported := errors.New("egraph: unsupported attention candidate region")
	if snapshot == nil {
		return nil, nil, nil, nil, unsupported
	}
	nodes, outputs, inputs := snapshot.Nodes(), snapshot.Outputs(), snapshot.Inputs()
	if len(nodes) != 9 || len(outputs) != 1 || len(inputs) != 2 {
		return nil, nil, nil, nil, unsupported
	}
	match := func(node *cudair.Node, kind cudair.NodeKind, op string, arity int) bool {
		return node != nil && node.Kind == kind && node.Op == op && len(node.Inputs) == arity
	}
	output = outputs[0]
	if !match(output, cudair.KindMatMul, "matmul", 2) {
		return nil, nil, nil, nil, unsupported
	}
	reshape, values := output.Inputs[0], output.Inputs[1]
	if !match(reshape, cudair.KindReshape, "reshape", 1) || !match(values, cudair.KindInput, "input", 0) || len(values.Shape) != 2 {
		return nil, nil, nil, nil, unsupported
	}
	probability := reshape.Inputs[0]
	if !match(probability, cudair.KindBinary, "div", 2) {
		return nil, nil, nil, nil, unsupported
	}
	exponential, denominator := probability.Inputs[0], probability.Inputs[1]
	if !match(exponential, cudair.KindUnary, "exp", 1) || !match(denominator, cudair.KindReduce, "sum", 1) || denominator.Inputs[0].ID != exponential.ID || !slices.Equal(denominator.Axes, []int{0}) {
		return nil, nil, nil, nil, unsupported
	}
	shifted := exponential.Inputs[0]
	if !match(shifted, cudair.KindBinary, "sub", 2) {
		return nil, nil, nil, nil, unsupported
	}
	scores, maximum := shifted.Inputs[0], shifted.Inputs[1]
	if !match(scores, cudair.KindInput, "input", 0) || len(scores.Shape) != 1 || !match(maximum, cudair.KindReduce, "max", 1) || maximum.Inputs[0].ID != scores.ID || !slices.Equal(maximum.Axes, []int{0}) {
		return nil, nil, nil, nil, unsupported
	}
	n, d := scores.Shape[0], values.Shape[1]
	if n < 1 || n > 64 || d < 1 || d > 64 || values.Shape[0] != n || scores.Name == values.Name || !slices.Equal(reshape.Shape, []int{1, n}) || !slices.Equal(output.Shape, []int{1, d}) {
		return nil, nil, nil, nil, unsupported
	}
	roles := []*cudair.Node{scores, values, maximum, shifted, exponential, denominator, probability, reshape, output}
	// The registry accepts more shapes than this candidate supports. In
	// particular, a full reduction has the canonical shape [1], not [N].
	shapes := [][]int{{n}, {n, d}, {1}, {n}, {n}, {1}, {n}, {1, n}, {1, d}}
	seen := make(map[int]bool, len(roles))
	for i, node := range roles {
		if seen[node.ID] || !slices.Equal(node.Shape, shapes[i]) {
			return nil, nil, nil, nil, unsupported
		}
		seen[node.ID] = true
	}
	for _, node := range nodes {
		if !seen[node.ID] || node.Type != cudair.Float32 || node.Layout != cudashape.LayoutRowMajor {
			return nil, nil, nil, nil, unsupported
		}
		strides, strideErr := cudashape.RowMajorStrides(node.Shape)
		if strideErr != nil || !slices.Equal(node.Strides, strides) {
			return nil, nil, nil, nil, unsupported
		}
		if node.ID == reshape.ID {
			if node.Effect != cudair.EffectInPlaceAlias || node.StorageAlias == nil || node.StorageAlias.ID != probability.ID {
				return nil, nil, nil, nil, unsupported
			}
		} else if node.Effect != cudair.EffectReadOnly || node.StorageAlias != nil {
			return nil, nil, nil, nil, unsupported
		}
		ids = append(ids, node.ID)
	}
	return scores, values, output, ids, nil
}

// Identity returns the graph/input/policy/tile binding digest. It is not a
// device qualification or evidence that a different workload is equivalent.
func (c *OnlineAttentionCandidate) Identity() string { return c.identity }

// Snapshot returns the original immutable snapshot, whose accessors clone nodes.
func (c *OnlineAttentionCandidate) Snapshot() *cudair.Snapshot { return c.snapshot }

// Output returns the checked CPU candidate output for the owned inputs.
func (c *OnlineAttentionCandidate) Output() []float32 { return slices.Clone(c.output) }

// Reference returns the unchanged graph's independently evaluated output.
func (c *OnlineAttentionCandidate) Reference() []float32 { return slices.Clone(c.reference) }

// Inputs returns detached copies of the candidate's exact input bindings.
func (c *OnlineAttentionCandidate) Inputs() map[string][]float32 {
	out := make(map[string][]float32, len(c.inputs))
	for name, data := range c.inputs {
		out[name] = slices.Clone(data)
	}
	return out
}

// Derivation returns detached semantic IDs and explicit finite tile states.
func (c *OnlineAttentionCandidate) Derivation() AttentionCandidateDerivation {
	out := c.derivation
	if out.Mask != nil {
		mask := out.Mask.clone()
		out.Mask = &mask
	}
	out.NodeIDs = slices.Clone(out.NodeIDs)
	out.Tiles = slices.Clone(out.Tiles)
	for i := range out.Tiles {
		out.Tiles[i].UnnormalizedOutput = slices.Clone(out.Tiles[i].UnnormalizedOutput)
	}
	return out
}

func attentionCandidateIdentity(c *OnlineAttentionCandidate, scores, values *cudair.Node) string {
	h := sha256.New()
	put := func(value uint64) { var word [8]byte; binary.LittleEndian.PutUint64(word[:], value); h.Write(word[:]) }
	text := func(s string) { put(uint64(len(s))); h.Write([]byte(s)) }
	text("checked-online-attention-candidate-v1")
	text(c.derivation.Rule)
	put(uint64(c.derivation.TileSize))
	put(math.Float64bits(c.derivation.Policy.Absolute))
	put(math.Float64bits(c.derivation.Policy.Relative))
	if mask := c.derivation.Mask; mask != nil {
		text(c.derivation.FullyMaskedPolicy)
		text(mask.kind)
		put(uint64(mask.query))
		put(uint64(mask.prefix))
		put(uint64(len(mask.parents)))
		for _, parent := range mask.parents {
			put(uint64(parent))
		}
		put(uint64(len(mask.excluded)))
		for _, excluded := range mask.excluded {
			if excluded {
				put(1)
			} else {
				put(0)
			}
		}
	}
	for _, node := range c.snapshot.Nodes() {
		put(uint64(node.ID))
		text(node.Name)
		text(node.Op)
		put(uint64(len(node.Shape)))
		for _, extent := range node.Shape {
			put(uint64(extent))
		}
		put(uint64(len(node.Inputs)))
		for _, input := range node.Inputs {
			put(uint64(input.ID))
		}
	}
	for _, node := range []*cudair.Node{scores, values} {
		text(node.Name)
		put(uint64(len(c.inputs[node.Name])))
		for _, value := range c.inputs[node.Name] {
			put(uint64(math.Float32bits(value)))
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}
