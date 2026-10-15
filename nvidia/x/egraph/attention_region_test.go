package egraph

import (
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func attentionRegionGraph(n, d int) (*cudair.Graph, map[string]*cudair.Node) {
	g := cudair.NewGraph()
	scores := g.AddInput("scores", []int{n}, cudair.Float32)
	values := g.AddInput("values", []int{n, d}, cudair.Float32)
	maximum := g.AddReduce("max", scores, 0)
	shifted := g.AddBinary("sub", scores, maximum)
	exponential := g.AddUnary("exp", shifted)
	denominator := g.AddReduce("sum", exponential, 0)
	probability := g.AddBinary("div", exponential, denominator)
	reshape := g.AddReshape(probability, []int{1, n})
	out := g.AddMatMul(reshape, values)
	_ = g.SetOutputs(out)
	return g, map[string]*cudair.Node{"scores": scores, "values": values, "max": maximum,
		"sub": shifted, "exp": exponential, "sum": denominator, "div": probability, "reshape": reshape, "out": out}
}

func attentionRegionSnapshot(t *testing.T, g *cudair.Graph) *cudair.Snapshot {
	t.Helper()
	snapshot, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func TestCheckedOnlineAttentionCandidateShapes(t *testing.T) {
	setShape := func(node *cudair.Node, shape []int) {
		node.Shape = slices.Clone(shape)
		var err error
		node.Strides, err = cudashape.RowMajorStrides(shape)
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, test := range []struct {
		name          string
		shapes        map[string][]int
		snapshotValid bool
	}{
		{"scores", map[string][]int{"scores": {2, 1}}, false},
		{"values", map[string][]int{"values": {1, 2}}, false},
		{"maximum", map[string][]int{"max": {2}}, false},
		{"maximum_rank_zero", map[string][]int{"max": {}}, false},
		{"shifted", map[string][]int{"sub": {1, 2}}, false},
		{"exponential", map[string][]int{"exp": {1, 2}}, false},
		{"denominator", map[string][]int{"sum": {2}}, false},
		{"denominator_rank_zero", map[string][]int{"sum": {}}, false},
		{"probabilities", map[string][]int{"div": {1, 2}}, false},
		{"reshape", map[string][]int{"reshape": {2, 1}}, false},
		{"output", map[string][]int{"out": {2, 1}}, false},
		// The current semantic authority checks unary shapes as well.
		{"exponential_and_probabilities", map[string][]int{"exp": {1, 2}, "div": {1, 2}}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			g, roles := attentionRegionGraph(2, 2)
			for role, shape := range test.shapes {
				setShape(roles[role], shape)
			}
			snapshot, err := g.Snapshot()
			if (err == nil) != test.snapshotValid {
				t.Fatalf("snapshot admission = %v, want %v: %v", err == nil, test.snapshotValid, err)
			}
			if err != nil {
				// The current graph authority refuses malformed shapes before
				// an immutable snapshot can reach the candidate constructor.
				if !errors.Is(err, cudair.ErrShapeMismatch) {
					t.Fatalf("unexpected shape refusal: %v", err)
				}
				return
			}
			inputs := map[string][]float32{"scores": {0, 0}, "values": {3, 5, -1, -3}}
			if _, err := NewCheckedOnlineAttentionCandidate(snapshot, inputs, 1, AttentionCandidatePolicy{}); err == nil {
				t.Fatal("malformed matched shape admitted")
			}
		})
	}
	// AddReduce's canonical full-reduction scalar form remains [1].
	g, roles := attentionRegionGraph(2, 2)
	for _, role := range []string{"max", "sum"} {
		if !slices.Equal(roles[role].Shape, []int{1}) {
			t.Fatalf("%s canonical reduction shape = %v", role, roles[role].Shape)
		}
	}
	if _, err := NewCheckedOnlineAttentionCandidate(attentionRegionSnapshot(t, g), map[string][]float32{"scores": {0, 0}, "values": {3, 5, -1, -3}}, 1, AttentionCandidatePolicy{}); err != nil {
		t.Fatal(err)
	}
}

func TestCheckedOnlineAttentionCandidateRoleMetadata(t *testing.T) {
	for _, role := range []string{"scores", "values", "max", "sub", "exp", "sum", "div", "reshape", "out"} {
		t.Run(role, func(t *testing.T) {
			g, roles := attentionRegionGraph(2, 1)
			roles[role].Layout = cudashape.LayoutColumnMajor
			snapshot, err := g.Snapshot()
			if err != nil {
				if !strings.Contains(err.Error(), "unsupported layout column-major") {
					t.Fatalf("unexpected layout refusal: %v", err)
				}
				return
			}
			if _, err := NewCheckedOnlineAttentionCandidate(snapshot, map[string][]float32{"scores": {0, 0}, "values": {3, -1}}, 1, AttentionCandidatePolicy{}); err == nil {
				t.Fatal("unsupported matched layout admitted")
			}
		})
	}
}

// Retain both independent reproducers with their original exact inputs. The
// current semantic validator may refuse malformed reductions before snapshot
// construction. Retain that explicit refusal without bypassing its authority.
func TestCheckedOnlineAttentionCandidateIndependentReductionShapeRefusal(t *testing.T) {
	for _, role := range []string{"max", "sum"} {
		t.Run(role, func(t *testing.T) {
			g, roles := attentionRegionGraph(2, 1)
			roles[role].Shape = []int{2}
			roles[role].Strides = []int{1}
			snapshot, err := g.Snapshot()
			if err != nil {
				if !errors.Is(err, cudair.ErrShapeMismatch) {
					t.Fatalf("unexpected reduction refusal: %v", err)
				}
				return
			}
			candidate, err := NewCheckedOnlineAttentionCandidate(snapshot, map[string][]float32{"scores": {0, 0}, "values": {3, -1}}, 1, AttentionCandidatePolicy{})
			if err == nil {
				t.Fatalf("malformed reduction admitted: output=%v reference=%v", candidate.Output(), candidate.Reference())
			}
		})
	}
}

func TestCheckedOnlineAttentionCandidateExactAndOptIn(t *testing.T) {
	g, _ := attentionRegionGraph(2, 1)
	snapshot := attentionRegionSnapshot(t, g)
	inputs := map[string][]float32{"scores": {0, 0}, "values": {3, -1}}
	candidate, err := NewCheckedOnlineAttentionCandidate(snapshot, inputs, 1, AttentionCandidatePolicy{})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(candidate.Output(), []float32{1}) || !reflect.DeepEqual(candidate.Output(), candidate.Reference()) {
		t.Fatalf("candidate %v, reference %v", candidate.Output(), candidate.Reference())
	}
	inputs["scores"][1] = -0.2
	if _, err := NewCheckedOnlineAttentionCandidate(snapshot, inputs, 1, AttentionCandidatePolicy{}); err == nil {
		t.Fatal("FP32 intermediate rounding difference passed exact policy")
	}
	loose, err := NewCheckedOnlineAttentionCandidate(snapshot, inputs, 1, AttentionCandidatePolicy{Absolute: 2e-7})
	if err != nil {
		t.Fatal(err)
	}
	refs, err := cudaref.EvaluateSnapshot(snapshot, inputs)
	if err != nil {
		t.Fatal(err)
	}
	want := refs[snapshot.Outputs()[0].Name]
	if !reflect.DeepEqual(loose.Reference(), want) || reflect.DeepEqual(loose.Output(), want) {
		t.Fatalf("candidate %v, unchanged reference %v, oracle %v", loose.Output(), loose.Reference(), want)
	}
	for _, policy := range []AttentionCandidatePolicy{{Absolute: 1e-10}, {Relative: 1e-10}, {Absolute: -1}, {Relative: -1}, {Absolute: math.NaN()}, {Relative: math.Inf(1)}} {
		if _, err := NewCheckedOnlineAttentionCandidate(snapshot, inputs, 1, policy); err == nil {
			t.Fatalf("policy %+v admitted", policy)
		}
	}
	if _, err := NewCheckedOnlineAttentionCandidate(snapshot, inputs, 1, AttentionCandidatePolicy{Relative: 2e-7}); err != nil {
		t.Fatalf("explicit relative policy: %v", err)
	}
}

func TestCheckedOnlineAttentionCandidatePartitionsAndTrace(t *testing.T) {
	g, roles := attentionRegionGraph(5, 2)
	snapshot := attentionRegionSnapshot(t, g)
	inputs := map[string][]float32{"scores": {-3, -2, 1, 2, 0}, "values": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
	for _, width := range []int{1, 2, 3, 5} {
		c, err := NewCheckedOnlineAttentionCandidate(snapshot, inputs, width, AttentionCandidatePolicy{Absolute: 1e-5})
		if err != nil {
			t.Fatal(err)
		}
		d := c.Derivation()
		if d.Rule != "provided-stable-online-row-v1" || d.ScoreID != roles["scores"].ID || d.ValueID != roles["values"].ID || d.OutputID != roles["out"].ID || len(d.NodeIDs) != 9 || d.TileSize != width || len(d.Tiles) != (5+width-1)/width {
			t.Fatalf("trace %+v", d)
		}
		for i, tile := range d.Tiles {
			if tile.Start != i*width || tile.End != min((i+1)*width, 5) || tile.Normalizer <= 0 || len(tile.UnnormalizedOutput) != 2 {
				t.Fatalf("tile %+v", tile)
			}
		}
		last := d.Tiles[len(d.Tiles)-1]
		if last.Maximum != 2 {
			t.Fatalf("final maximum %v", last.Maximum)
		}
		for i, value := range c.Output() {
			if value != float32(last.UnnormalizedOutput[i]/last.Normalizer) {
				t.Fatal("output does not consume trace normalizer")
			}
		}
	}
}

func TestCheckedOnlineAttentionCandidateRefusals(t *testing.T) {
	cases := []struct {
		name  string
		alter func(*cudair.Graph, map[string]*cudair.Node)
	}{
		{"numerator is denominator", func(g *cudair.Graph, r map[string]*cudair.Node) { r["div"].Inputs[0] = r["sum"] }},
		{"different denominator input", func(g *cudair.Graph, r map[string]*cudair.Node) { r["sum"].Inputs[0] = r["sub"] }},
		{"reversed subtraction", func(g *cudair.Graph, r map[string]*cudair.Node) {
			r["sub"].Inputs[0], r["sub"].Inputs[1] = r["sub"].Inputs[1], r["sub"].Inputs[0]
		}},
		{"different reduction", func(g *cudair.Graph, r map[string]*cudair.Node) { r["sum"].Op = "max" }},
		{"extra output", func(g *cudair.Graph, r map[string]*cudair.Node) { _ = g.SetOutputs(r["out"], r["scores"]) }},
		{"strided input", func(g *cudair.Graph, r map[string]*cudair.Node) { r["scores"].Strides = []int{2} }},
		{"other layout", func(g *cudair.Graph, r map[string]*cudair.Node) { r["scores"].Layout = cudashape.LayoutColumnMajor }},
		{"wrong axis", func(g *cudair.Graph, r map[string]*cudair.Node) { r["sum"].Axes = []int{1} }},
		{"mutation effect", func(g *cudair.Graph, r map[string]*cudair.Node) { r["scores"].Effect = cudair.EffectMutating }},
		{"extra scale", func(g *cudair.Graph, r map[string]*cudair.Node) {
			scale := g.AddConstant("scale", []int{1}, cudair.Float32, []float32{0.5})
			r["sub"].Inputs[0] = g.AddBinary("mul", r["scores"], scale)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, roles := attentionRegionGraph(2, 2)
			tc.alter(g, roles)
			snapshot, err := g.Snapshot()
			if err != nil {
				return
			} // The authoritative graph may reject earlier.
			if _, err := NewCheckedOnlineAttentionCandidate(snapshot, map[string][]float32{"scores": {0, 0}, "values": {1, 2, 3, 4}}, 1, AttentionCandidatePolicy{Absolute: 1}); err == nil {
				t.Fatal("unsupported graph admitted")
			}
		})
	}
	for _, dims := range [][2]int{{0, 1}, {65, 1}, {1, 0}, {1, 65}} {
		g, _ := attentionRegionGraph(dims[0], dims[1])
		snapshot, err := g.Snapshot()
		if err != nil {
			continue
		}
		if _, err := NewCheckedOnlineAttentionCandidate(snapshot, nil, 1, AttentionCandidatePolicy{}); err == nil {
			t.Fatalf("dimensions %v admitted", dims)
		}
	}
	g, _ := attentionRegionGraph(2, 1)
	snapshot := attentionRegionSnapshot(t, g)
	if _, err := NewCheckedOnlineAttentionCandidate(nil, nil, 1, AttentionCandidatePolicy{}); err == nil {
		t.Fatal("nil snapshot admitted")
	}
	for _, width := range []int{-1, 0, 3} {
		if _, err := NewCheckedOnlineAttentionCandidate(snapshot, map[string][]float32{"scores": {0, 0}, "values": {1, 1}}, width, AttentionCandidatePolicy{}); err == nil {
			t.Fatalf("tile %d admitted", width)
		}
	}
	for _, inputs := range []map[string][]float32{nil, {"scores": {0, 0}}, {"scores": {0}, "values": {1, 1}}, {"scores": {0, 0}, "values": {1, 1}, "mask": {0}}, {"scores": {float32(math.NaN()), 0}, "values": {1, 1}}, {"scores": {0, 0}, "values": {float32(math.Inf(1)), 1}}} {
		if _, err := NewCheckedOnlineAttentionCandidate(snapshot, inputs, 1, AttentionCandidatePolicy{}); err == nil {
			t.Fatalf("bindings %v admitted", inputs)
		}
	}
}

func TestCheckedOnlineAttentionCandidateOwnershipAndIdentity(t *testing.T) {
	g, roles := attentionRegionGraph(2, 1)
	snapshot := attentionRegionSnapshot(t, g)
	inputs := map[string][]float32{"scores": {0, 0}, "values": {3, -1}}
	a, err := NewCheckedOnlineAttentionCandidate(snapshot, inputs, 1, AttentionCandidatePolicy{})
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewCheckedOnlineAttentionCandidate(snapshot, inputs, 1, AttentionCandidatePolicy{})
	if err != nil {
		t.Fatal(err)
	}
	if a.Identity() != b.Identity() {
		t.Fatal("same bindings changed identity")
	}
	identity, baseline, graph := a.Identity(), a.Output(), a.Snapshot().Nodes()
	a.Output()[0] = 900
	a.Reference()[0] = 900
	a.Inputs()["values"][0] = 900
	d := a.Derivation()
	d.NodeIDs[0] = -1
	d.Tiles[0].UnnormalizedOutput[0] = 900
	d.Tiles[0].Start = -1
	a.Snapshot().Outputs()[0].Name = "mutated"
	inputs["values"][0] = 900
	roles["scores"].Name = "original graph changed"
	if a.Identity() != identity || !reflect.DeepEqual(a.Output(), baseline) || !reflect.DeepEqual(b.Output(), baseline) || a.Inputs()["values"][0] != 3 || !reflect.DeepEqual(a.Snapshot().Nodes(), graph) {
		t.Fatal("caller mutation changed a retained candidate or snapshot")
	}
	for _, change := range []struct {
		tile   int
		policy AttentionCandidatePolicy
		inputs map[string][]float32
	}{
		{2, AttentionCandidatePolicy{}, map[string][]float32{"scores": {0, 0}, "values": {3, -1}}},
		{1, AttentionCandidatePolicy{Absolute: 1e-5}, map[string][]float32{"scores": {0, 0}, "values": {3, -1}}},
		{1, AttentionCandidatePolicy{}, map[string][]float32{"scores": {0, 0}, "values": {5, -1}}},
	} {
		c, err := NewCheckedOnlineAttentionCandidate(snapshot, change.inputs, change.tile, change.policy)
		if err != nil {
			t.Fatal(err)
		}
		if c.Identity() == identity {
			t.Fatal("changed tile, policy or inputs reused identity")
		}
	}
}
