package egraph

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// GroupingPolicy bounds admission CPU multiply-add pairs across the reference
// and all requested candidates. Later Run calls repeat that candidate's work.
// MaxOperations must be positive. Zero tolerances require
// identical FP32 output bits, including signed zero; nonzero tolerances apply
// only to these bound inputs. Accumulation is FP64 in ascending channel order,
// with one FP32 conversion per output, matching the snapshot reference.
type GroupingPolicy struct {
	MaxOperations      int
	Absolute, Relative float64
}

// GroupingRange is a half-open interval of independent token rows.
type GroupingRange struct{ Start, End int }

// GroupingDerivation binds a supplied scheduling rule to original semantic IDs.
// Rows cover each token exactly once. It asserts no semantic e-class union.
type GroupingDerivation struct {
	Rule                        string
	InputID, WeightID, OutputID int
	Width                       int
	Rows                        []GroupingRange
	Policy                      GroupingPolicy
}

// GroupingCandidate owns a checked, input-bound CPU scheduling alternative for
// one dense matmul. Its zero value is unusable. It carries no compiler, device,
// performance or whole-lane qualification. Accessors return detached data.
type GroupingCandidate struct {
	snapshot     *cudair.Snapshot
	inputs       map[string][]float32
	xName, wName string
	m, k, n      int
	reference    []float32
	derivation   GroupingDerivation
	identity     string
}

// NewCheckedGroupingCandidates admits exactly two distinct readonly contiguous
// FP32 matrix inputs and their sole matmul output, with each dimension in [1,64].
// That graph proves row independence: each output row reads only its matching
// input row and shared immutable weights; there are no alias or update nodes.
// Other graphs, including causal/stateful regions, are unsupported.
//
// widths contains 1 to 8 distinct group sizes in [1,token count]. Each candidate
// executes a weight-major loop within each group, including a final partial
// group, and is compared with independent cudaref evaluation of the unchanged
// snapshot. No graph is synthesized, mutated or unioned, and no width is chosen
// by an unmeasured performance objective. The original matmul remains available.
func NewCheckedGroupingCandidates(snapshot *cudair.Snapshot, inputs map[string][]float32, widths []int, policy GroupingPolicy) ([]*GroupingCandidate, error) {
	if policy.MaxOperations <= 0 || policy.Absolute < 0 || policy.Relative < 0 ||
		!finite64(policy.Absolute) || !finite64(policy.Relative) {
		return nil, errors.New("egraph: invalid grouping policy")
	}
	x, w, out, err := checkedGroupingRegion(snapshot)
	if err != nil {
		return nil, err
	}
	m, k, n := x.Shape[0], x.Shape[1], w.Shape[1]
	if len(widths) < 1 || len(widths) > 8 {
		return nil, errors.New("egraph: grouping needs 1 to 8 widths")
	}
	seen := make(map[int]bool, len(widths))
	for _, width := range widths {
		if width < 1 || width > m || seen[width] {
			return nil, errors.New("egraph: invalid or repeated grouping width")
		}
		seen[width] = true
	}
	// Region extents and candidate count were bounded before multiplication.
	if (len(widths)+1)*m*k*n > policy.MaxOperations {
		return nil, errors.New("egraph: grouping operation budget exceeded")
	}
	if len(inputs) != 2 {
		return nil, errors.New("egraph: grouping needs exactly two input bindings")
	}
	bound := make(map[string][]float32, 2)
	for _, node := range []*cudair.Node{x, w} {
		data, ok := inputs[node.Name]
		if !ok || len(data) != node.NumElements() {
			return nil, fmt.Errorf("egraph: invalid grouping input %q", node.Name)
		}
		for _, value := range data {
			if !finite64(float64(value)) {
				return nil, fmt.Errorf("egraph: nonfinite grouping input %q", node.Name)
			}
		}
		bound[node.Name] = slices.Clone(data)
	}
	refs, err := cudaref.EvaluateSnapshot(snapshot, bound)
	if err != nil {
		return nil, fmt.Errorf("egraph: grouping reference: %w", err)
	}
	reference := refs[out.Name]
	candidates := make([]*GroupingCandidate, 0, len(widths))
	for _, width := range widths {
		c := &GroupingCandidate{snapshot: snapshot, inputs: bound, xName: x.Name, wName: w.Name,
			m: m, k: k, n: n, reference: slices.Clone(reference),
			derivation: GroupingDerivation{Rule: "provided-independent-dense-rows-v1", InputID: x.ID,
				WeightID: w.ID, OutputID: out.ID, Width: width, Policy: policy}}
		for start := 0; start < m; start += width {
			c.derivation.Rows = append(c.derivation.Rows, GroupingRange{start, min(start+width, m)})
		}
		if _, err := c.Run(); err != nil {
			return nil, err
		}
		// All fields below are deterministic and finite after admission. JSON
		// includes input names, values, dimensions, semantic IDs and full policy.
		encoded, err := json.Marshal(struct {
			X, W, Output string
			M, K, N      int
			Inputs       map[string][]float32
			Derivation   GroupingDerivation
		}{x.Name, w.Name, out.Name, m, k, n, bound, c.derivation})
		if err != nil {
			return nil, fmt.Errorf("egraph: grouping identity: %w", err)
		}
		digest := sha256.Sum256(encoded)
		c.identity = hex.EncodeToString(digest[:])
		candidates = append(candidates, c)
	}
	return candidates, nil
}

func checkedGroupingRegion(snapshot *cudair.Snapshot) (x, w, out *cudair.Node, err error) {
	unsupported := errors.New("egraph: unsupported grouping region")
	if snapshot == nil {
		return nil, nil, nil, unsupported
	}
	nodes, outputs, inputs := snapshot.Nodes(), snapshot.Outputs(), snapshot.Inputs()
	if len(nodes) != 3 || len(inputs) != 2 || len(outputs) != 1 {
		return nil, nil, nil, unsupported
	}
	out = outputs[0]
	if out.Kind != cudair.KindMatMul || out.Op != "matmul" || len(out.Inputs) != 2 {
		return nil, nil, nil, unsupported
	}
	x, w = out.Inputs[0], out.Inputs[1]
	if x.ID == w.ID || x.Name == w.Name || x.Kind != cudair.KindInput || w.Kind != cudair.KindInput ||
		x.Op != "input" || w.Op != "input" || len(x.Inputs) != 0 || len(w.Inputs) != 0 ||
		len(x.Shape) != 2 || len(w.Shape) != 2 || x.Shape[1] != w.Shape[0] ||
		!slices.Equal(out.Shape, []int{x.Shape[0], w.Shape[1]}) {
		return nil, nil, nil, unsupported
	}
	for _, node := range nodes {
		if node.Type != cudair.Float32 || node.Effect != cudair.EffectReadOnly ||
			node.StorageAlias != nil || node.Layout != cudashape.LayoutRowMajor {
			return nil, nil, nil, unsupported
		}
		for _, extent := range node.Shape {
			if extent < 1 || extent > 64 {
				return nil, nil, nil, unsupported
			}
		}
		strides, strideErr := cudashape.RowMajorStrides(node.Shape)
		if strideErr != nil || !slices.Equal(node.Strides, strides) {
			return nil, nil, nil, unsupported
		}
	}
	return x, w, out, nil
}

// Run executes the retained grouped CPU schedule on owned inputs and checks
// every result against the unchanged graph reference. It returns a new slice.
func (c *GroupingCandidate) Run() ([]float32, error) {
	if c == nil || c.snapshot == nil {
		return nil, errors.New("egraph: uninitialized grouping candidate")
	}
	got, err := executeGrouping(c.inputs[c.xName], c.inputs[c.wName], c.m, c.k, c.n, c.derivation.Width, c.derivation.Rows)
	if err != nil {
		return nil, err
	}
	if err := checkGroupingOutput(got, c.reference, c.derivation.Policy); err != nil {
		return nil, err
	}
	return got, nil
}

func executeGrouping(x, w []float32, m, k, n, width int, rows []GroupingRange) ([]float32, error) {
	if m < 1 || m > 64 || k < 1 || k > 64 || n < 1 || n > 64 || width < 1 || width > m ||
		len(x) != m*k || len(w) != k*n || len(rows) != (m+width-1)/width {
		return nil, errors.New("egraph: invalid grouping execution extent")
	}
	for i, row := range rows {
		if row.Start != i*width || row.End != min(row.Start+width, m) {
			return nil, errors.New("egraph: invalid grouping row coverage")
		}
	}
	acc := make([]float64, m*n)
	for _, group := range rows {
		for column := 0; column < n; column++ {
			for channel := 0; channel < k; channel++ {
				weight := float64(w[channel*n+column])
				for row := group.Start; row < group.End; row++ {
					product := float64(x[row*k+channel]) * weight
					acc[row*n+column] += product
				}
			}
		}
	}
	out := make([]float32, len(acc))
	for i, value := range acc {
		out[i] = float32(value)
	}
	return out, nil
}

func checkGroupingOutput(got, reference []float32, policy GroupingPolicy) error {
	if len(got) != len(reference) {
		return errors.New("egraph: grouping output extent mismatch")
	}
	for i, value := range got {
		want := reference[i]
		if !finite64(float64(value)) || !finite64(float64(want)) {
			return errors.New("egraph: nonfinite grouping output")
		}
		if policy.Absolute == 0 && policy.Relative == 0 {
			if math.Float32bits(value) != math.Float32bits(want) {
				return fmt.Errorf("egraph: grouping differs from exact reference at %d", i)
			}
		} else {
			tolerance := policy.Absolute + policy.Relative*math.Abs(float64(want))
			if !finite64(tolerance) || math.Abs(float64(value)-float64(want)) > tolerance {
				return fmt.Errorf("egraph: grouping exceeds tolerance at %d", i)
			}
		}
	}
	return nil
}

// Identity returns the input, semantic-role, dimensions, policy and schedule digest.
func (c *GroupingCandidate) Identity() string { return c.identity }

// Snapshot returns the original immutable matmul snapshot.
func (c *GroupingCandidate) Snapshot() *cudair.Snapshot { return c.snapshot }

// Reference returns detached independently evaluated original matmul values.
func (c *GroupingCandidate) Reference() []float32 { return slices.Clone(c.reference) }

// Inputs returns detached input bindings.
func (c *GroupingCandidate) Inputs() map[string][]float32 {
	return map[string][]float32{c.xName: slices.Clone(c.inputs[c.xName]), c.wName: slices.Clone(c.inputs[c.wName])}
}

// Derivation returns detached original IDs, width, policy and row intervals.
func (c *GroupingCandidate) Derivation() GroupingDerivation {
	out := c.derivation
	out.Rows = slices.Clone(out.Rows)
	return out
}
