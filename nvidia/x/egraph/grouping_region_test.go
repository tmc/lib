package egraph

import (
	"errors"
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func groupingGraph(m, k, n int) (*cudair.Graph, *cudair.Node, *cudair.Node, *cudair.Node) {
	g := cudair.NewGraph()
	x := g.AddInput("tokens", []int{m, k}, cudair.Float32)
	w := g.AddInput("weights", []int{k, n}, cudair.Float32)
	out := g.AddMatMul(x, w)
	_ = g.SetOutputs(out)
	return g, x, w, out
}

func groupingSnapshot(t testing.TB, g *cudair.Graph) *cudair.Snapshot {
	t.Helper()
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func groupingInputs(m, k, n int) map[string][]float32 {
	x, w := make([]float32, m*k), make([]float32, k*n)
	for row := 0; row < m; row++ {
		for channel := 0; channel < k; channel++ {
			x[row*k+channel] = float32((row+1)*(channel+2)) / 8
		}
	}
	for channel := 0; channel < k; channel++ {
		for column := 0; column < n; column++ {
			w[channel*n+column] = float32((channel+1)*(column+1)-column*3) / 4
		}
	}
	return map[string][]float32{"tokens": x, "weights": w}
}

// Deliberately token-major, unlike the production weight-major group loop.
// No candidate/partition/evaluator helpers participate in this reference.
func groupingTokenOracle(x, w []float32, m, k, n int) []float32 {
	out := make([]float32, m*n)
	for token := 0; token < m; token++ {
		for column := 0; column < n; column++ {
			var sum float64
			for channel := 0; channel < k; channel++ {
				sum += float64(x[token*k+channel]) * float64(w[channel*n+column])
			}
			out[token*n+column] = float32(sum)
		}
	}
	return out
}

func TestCheckedGroupingCandidateReferenceAndReplay(t *testing.T) {
	for _, size := range [][3]int{{5, 3, 2}, {7, 2, 3}, {64, 64, 64}} {
		m, k, n := size[0], size[1], size[2]
		g, x, w, out := groupingGraph(m, k, n)
		snapshot := groupingSnapshot(t, g)
		inputs := groupingInputs(m, k, n)
		widths := []int{2, 3}
		policy := GroupingPolicy{MaxOperations: 3 * m * k * n}
		candidates, err := NewCheckedGroupingCandidates(snapshot, inputs, widths, policy)
		if err != nil || len(candidates) != len(widths) {
			t.Fatalf("candidates=%d err=%v", len(candidates), err)
		}
		want := groupingTokenOracle(inputs[x.Name], inputs[w.Name], m, k, n)
		for i, c := range candidates {
			got, err := c.Run()
			if err != nil || !slices.Equal(got, want) || !slices.Equal(c.Reference(), want) {
				t.Fatalf("width %d: got=%v want=%v err=%v", widths[i], got, want, err)
			}
			d := c.Derivation()
			if d.Rule != "provided-independent-dense-rows-v1" || d.InputID != x.ID || d.WeightID != w.ID ||
				d.OutputID != out.ID || d.Width != widths[i] || d.Policy != policy {
				t.Fatalf("unbound derivation: %+v", d)
			}
			coverage := make([]int, m)
			for _, group := range d.Rows {
				if group.End-group.Start > d.Width || group.Start < 0 || group.End > m || group.End <= group.Start {
					t.Fatal("invalid group", group)
				}
				for row := group.Start; row < group.End; row++ {
					coverage[row]++
				}
			}
			for row, count := range coverage {
				if count != 1 {
					t.Fatalf("row %d covered %d times", row, count)
				}
			}
			if m%d.Width != 0 && d.Rows[len(d.Rows)-1].End-d.Rows[len(d.Rows)-1].Start != m%d.Width {
				t.Fatal("tail lost")
			}
			replayed, err := NewCheckedGroupingCandidates(c.Snapshot(), c.Inputs(), []int{d.Width}, d.Policy)
			if err != nil || replayed[0].Identity() != c.Identity() || !reflect.DeepEqual(replayed[0].Derivation(), d) {
				t.Fatal("candidate replay changed binding", err)
			}
			if c.Snapshot() != snapshot || snapshot.Outputs()[0].ID != out.ID || len(snapshot.Nodes()) != 3 || snapshot.Outputs()[0].Op != "matmul" {
				t.Fatal("original matmul was replaced")
			}
		}
		if candidates[0].Identity() == candidates[1].Identity() {
			t.Fatal("width missing from identity")
		}
	}
}

func TestCheckedGroupingCandidateOwnership(t *testing.T) {
	g, x, _, _ := groupingGraph(5, 3, 2)
	inputs := groupingInputs(5, 3, 2)
	policy := GroupingPolicy{MaxOperations: 90}
	cs, err := NewCheckedGroupingCandidates(groupingSnapshot(t, g), inputs, []int{2, 3}, policy)
	if err != nil {
		t.Fatal(err)
	}
	c := cs[0]
	want, _ := c.Run()
	identity := c.Identity()
	inputs["tokens"][0] = 999
	inputs["weights"] = nil
	x.Name = "changed"
	c.Inputs()["tokens"][0] = 999
	c.Reference()[0] = 999
	c.Derivation().Rows[0].End = 999
	c.Snapshot().Nodes()[0].Shape[0] = 999
	wantCopy := slices.Clone(want)
	want[0] = 999
	got, err := c.Run()
	if err != nil || !slices.Equal(got, wantCopy) || c.Identity() != identity {
		t.Fatal("candidate borrowed caller/accessor storage", err)
	}
	other, err := cs[1].Run()
	if err != nil || !slices.Equal(other, wantCopy) {
		t.Fatal("candidate shared externally writable state", err)
	}
}

func TestCheckedGroupingCandidateAdmission(t *testing.T) {
	for _, tc := range []struct {
		name   string
		widths []int
		policy GroupingPolicy
	}{
		{"empty_widths", nil, GroupingPolicy{MaxOperations: 90}},
		{"too_many_widths", []int{1, 2, 3, 4, 5, 6, 7, 8, 9}, GroupingPolicy{MaxOperations: 10000}},
		{"zero_width", []int{0}, GroupingPolicy{MaxOperations: 90}},
		{"negative_width", []int{-1}, GroupingPolicy{MaxOperations: 90}},
		{"oversize_width", []int{6}, GroupingPolicy{MaxOperations: 90}},
		{"repeated_width", []int{2, 2}, GroupingPolicy{MaxOperations: 90}},
		{"missing_budget", []int{2}, GroupingPolicy{}},
		{"short_budget", []int{2, 3}, GroupingPolicy{MaxOperations: 89}},
		{"negative_tolerance", []int{2}, GroupingPolicy{MaxOperations: 90, Absolute: -1}},
		{"nan_tolerance", []int{2}, GroupingPolicy{MaxOperations: 90, Relative: math.NaN()}},
		{"infinite_tolerance", []int{2}, GroupingPolicy{MaxOperations: 90, Absolute: math.Inf(1)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g, _, _, _ := groupingGraph(5, 3, 2)
			cs, err := NewCheckedGroupingCandidates(groupingSnapshot(t, g), groupingInputs(5, 3, 2), tc.widths, tc.policy)
			if err == nil || cs != nil {
				t.Fatal("invalid request admitted")
			}
		})
	}
	for _, mutate := range []func(map[string][]float32){
		func(in map[string][]float32) { delete(in, "tokens") },
		func(in map[string][]float32) { in["tokens"] = nil },
		func(in map[string][]float32) { in["extra"] = []float32{1} },
		func(in map[string][]float32) { in["tokens"][0] = float32(math.NaN()) },
		func(in map[string][]float32) { in["weights"][0] = float32(math.Inf(1)) },
		func(in map[string][]float32) { in["tokens"][0], in["weights"][0] = math.MaxFloat32, math.MaxFloat32 },
	} {
		g, _, _, _ := groupingGraph(5, 3, 2)
		inputs := groupingInputs(5, 3, 2)
		mutate(inputs)
		if cs, err := NewCheckedGroupingCandidates(groupingSnapshot(t, g), inputs, []int{2}, GroupingPolicy{MaxOperations: 60}); err == nil || cs != nil {
			t.Fatal("invalid input/result admitted")
		}
	}
	if cs, err := NewCheckedGroupingCandidates(nil, nil, []int{1}, GroupingPolicy{MaxOperations: 1}); err == nil || cs != nil {
		t.Fatal("missing dependence graph admitted")
	}
	if _, err := new(GroupingCandidate).Run(); err == nil {
		t.Fatal("zero candidate ran")
	}
}

func TestCheckedGroupingCandidateDependence(t *testing.T) {
	// Snapshot is the authoritative first admission boundary: actual explicit
	// writes and reordered writes are rejected there, before grouping can run.
	for _, reorder := range []bool{false, true} {
		g, x, _, out := groupingGraph(5, 3, 2)
		before := g.AddUnary("neg", x)
		after := g.AddUnary("relu", before)
		before.Effect, before.StorageAlias = cudair.EffectMutating, x
		after.Effect, after.StorageAlias = cudair.EffectMutating, x
		if reorder {
			after.Inputs[0] = x
			before.Inputs[0] = after
		}
		_ = g.SetOutputs(out, after)
		if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrUnsupportedEffect) {
			t.Fatalf("reorder=%v: KV update graph admitted: %v", reorder, err)
		}
		if out.Op != "matmul" || before.StorageAlias != x || after.StorageAlias != x {
			t.Fatal("refusal erased observable writes")
		}
	}
	// A valid snapshot with an actual input alias must still be refused by
	// grouping; the candidate cannot interpret aliasing as independent rows.
	g := cudair.NewGraph()
	x := g.AddInput("tokens", []int{5, 3}, cudair.Float32)
	w := g.AddInput("weights", []int{3, 2}, cudair.Float32)
	alias := g.AddReshape(x, []int{5, 3})
	out := g.AddMatMul(alias, w)
	_ = g.SetOutputs(out)
	s := groupingSnapshot(t, g)
	if cs, err := NewCheckedGroupingCandidates(s, groupingInputs(5, 3, 2), []int{2}, GroupingPolicy{MaxOperations: 60}); err == nil || cs != nil {
		t.Fatal("actual alias boundary admitted")
	}
	if len(s.Nodes()) != 4 || s.NodeByID(alias.ID).StorageAlias.ID != x.ID {
		t.Fatal("alias refusal changed snapshot")
	}
	// Pure arithmetic before matmul is also outside the proved region: no
	// fabricated unrelated three-node graph may stand in for this input.
	g, x, w, _ = groupingGraph(5, 3, 2)
	out = g.AddMatMul(g.AddUnary("neg", x), w)
	_ = g.SetOutputs(out)
	if _, err := NewCheckedGroupingCandidates(groupingSnapshot(t, g), groupingInputs(5, 3, 2), []int{2}, GroupingPolicy{MaxOperations: 60}); err == nil {
		t.Fatal("unproved graph admitted")
	}
}

func TestCheckedGroupingCandidateShapeAndLayout(t *testing.T) {
	for _, edit := range []func(x, w, out *cudair.Node){
		func(x, w, out *cudair.Node) {
			x.Type, w.Type, out.Type = cudair.Float16, cudair.Float16, cudair.Float16
		},
		func(x, w, out *cudair.Node) { x.Layout = cudashape.LayoutColumnMajor; x.Strides = []int{1, 5} },
		func(x, w, out *cudair.Node) { w.Shape[0] = 4 },
		func(x, w, out *cudair.Node) { x.Strides = []int{4, 1} },
	} {
		g, x, w, out := groupingGraph(5, 3, 2)
		edit(x, w, out)
		s, err := g.Snapshot()
		if err != nil { // Refusal at the authoritative snapshot boundary counts.
			continue
		}
		if cs, err := NewCheckedGroupingCandidates(s, groupingInputs(5, 3, 2), []int{2}, GroupingPolicy{MaxOperations: 60}); err == nil || cs != nil {
			t.Fatal("unsupported type/layout/shape admitted")
		}
	}
	g, _, _, _ := groupingGraph(65, 1, 1)
	if _, err := NewCheckedGroupingCandidates(groupingSnapshot(t, g), groupingInputs(65, 1, 1), []int{2}, GroupingPolicy{MaxOperations: 130}); err == nil {
		t.Fatal("unbounded extent admitted")
	}
}

func TestCheckedGroupingCandidateCausalControls(t *testing.T) {
	m, k, n := 5, 3, 2
	inputs := groupingInputs(m, k, n)
	x, w := inputs["tokens"], inputs["weights"]
	want := groupingTokenOracle(x, w, m, k, n)
	rows := []GroupingRange{{0, 2}, {2, 4}, {4, 5}}
	for _, bad := range [][]GroupingRange{rows[:2], {{0, 2}, {0, 2}, {4, 5}}, {{0, 2}, {2, 4}, {4, 6}}} {
		if _, err := executeGrouping(x, w, m, k, n, 2, bad); err == nil {
			t.Fatal("missing/duplicate/invalid tail interval admitted", bad)
		}
	}
	missingTail := slices.Clone(want)
	clear(missingTail[4*n:])
	duplicateRow := slices.Clone(want)
	copy(duplicateRow[4*n:], duplicateRow[:n])
	wrongIndex := make([]float32, len(want))
	for row := 0; row < m; row++ {
		for column := 0; column < n; column++ {
			var sum float64
			for channel := 0; channel < k; channel++ {
				sum += float64(x[row*k+channel]) * float64(w[column*k+channel])
			}
			wrongIndex[row*n+column] = float32(sum)
		}
	}
	for _, mutant := range [][]float32{missingTail, duplicateRow, wrongIndex} {
		if slices.Equal(mutant, want) || checkGroupingOutput(mutant, want, GroupingPolicy{}) == nil {
			t.Fatal("value control did not reject grouping mutant")
		}
	}
	// Check the gate in the actual consumer, beyond calling its comparison helper.
	g, _, _, _ := groupingGraph(m, k, n)
	cs, err := NewCheckedGroupingCandidates(groupingSnapshot(t, g), inputs, []int{2}, GroupingPolicy{MaxOperations: 60})
	if err != nil {
		t.Fatal(err)
	}
	cs[0].inputs["tokens"][4*k] = -999
	if _, err := cs[0].Run(); err == nil {
		t.Fatal("actual consumer accepted changed row computation")
	}
}

func FuzzCheckedGroupingCandidate(f *testing.F) {
	f.Add(uint8(5), uint8(3), uint8(2), uint8(2), []byte{1, 4, 9, 15})
	f.Add(uint8(7), uint8(2), uint8(3), uint8(3), []byte{255, 0, 128})
	f.Fuzz(func(t *testing.T, mb, kb, nb, wb uint8, data []byte) {
		m, k, n := 1+int(mb%8), 1+int(kb%8), 1+int(nb%8)
		width := 1 + int(wb)%m
		g, _, _, _ := groupingGraph(m, k, n)
		inputs := groupingInputs(m, k, n)
		if len(data) > 0 {
			for i := range inputs["tokens"] {
				inputs["tokens"][i] = float32(int(data[i%len(data)])-128) / 16
			}
		}
		cs, err := NewCheckedGroupingCandidates(groupingSnapshot(t, g), inputs, []int{width}, GroupingPolicy{MaxOperations: 2 * m * k * n})
		if err != nil {
			t.Fatal(err)
		}
		got, err := cs[0].Run()
		want := groupingTokenOracle(inputs["tokens"], inputs["weights"], m, k, n)
		if err != nil || !slices.Equal(got, want) {
			t.Fatalf("shape=%dx%dx%d width=%d err=%v got=%v want=%v", m, k, n, width, err, got, want)
		}
	})
}
