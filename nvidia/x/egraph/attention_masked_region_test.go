package egraph

import (
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

func maskedRegionSnapshot(t *testing.T, n, d int) *cudair.Snapshot {
	t.Helper()
	g := cudair.NewGraph()
	scores := g.AddInput("scores", []int{n}, cudair.Float32)
	values := g.AddInput("values", []int{n, d}, cudair.Float32)
	e := g.AddUnary("exp", g.AddBinary("sub", scores, g.AddReduce("max", scores, 0)))
	p := g.AddBinary("div", e, g.AddReduce("sum", e, 0))
	out := g.AddMatMul(g.AddReshape(p, []int{1, n}), values)
	if err := g.SetOutputs(out); err != nil {
		t.Fatal(err)
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func TestAttentionRowMaskPredicates(t *testing.T) {
	for _, test := range []struct {
		name string
		make func() (AttentionRowMask, error)
		want []bool
	}{
		{"causal", func() (AttentionRowMask, error) { return NewCausalAttentionRowMask(2, 6) }, []bool{false, false, false, true, true, true}},
		{"prefix", func() (AttentionRowMask, error) { return NewPrefixAttentionRowMask(2, 6) }, []bool{false, false, true, true, true, true}},
		{"empty_prefix", func() (AttentionRowMask, error) { return NewPrefixAttentionRowMask(0, 3) }, []bool{true, true, true}},
		{"full_prefix", func() (AttentionRowMask, error) { return NewPrefixAttentionRowMask(3, 3) }, []bool{false, false, false}},
		{"tree_prefix", func() (AttentionRowMask, error) { return NewTreeAttentionRowMask([]int{-1, 0, 0, 1, 2}, 3, 1) }, []bool{false, false, false, true, false, true}},
		{"forest", func() (AttentionRowMask, error) { return NewTreeAttentionRowMask([]int{-1, -1, 1}, 2, 0) }, []bool{true, false, false}},
		{"tree_root", func() (AttentionRowMask, error) { return NewTreeAttentionRowMask([]int{-1, 0}, 0, 0) }, []bool{false, true}},
	} {
		t.Run(test.name, func(t *testing.T) {
			mask, err := test.make()
			if err != nil || !slices.Equal(mask.Dense(), test.want) {
				t.Fatalf("predicate = %v, %v; want %v", mask.Dense(), err, test.want)
			}
			mask.Dense()[0] = !test.want[0]
			if !slices.Equal(mask.Dense(), test.want) {
				t.Fatal("dense accessor aliases predicate")
			}
		})
	}
	for _, args := range [][2]int{{-1, 4}, {4, 4}, {0, 0}, {0, 65}} {
		if _, err := NewCausalAttentionRowMask(args[0], args[1]); err == nil {
			t.Fatalf("causal %v admitted", args)
		}
	}
	for _, args := range [][2]int{{-1, 4}, {5, 4}, {0, 0}, {0, 65}} {
		if _, err := NewPrefixAttentionRowMask(args[0], args[1]); err == nil {
			t.Fatalf("prefix %v admitted", args)
		}
	}
	for _, test := range []struct {
		parents       []int
		query, prefix int
	}{
		{nil, 0, 0}, {[]int{-1}, -1, 0}, {[]int{-1}, 1, 0},
		{[]int{-1}, 0, -1}, {[]int{-1}, 0, 64},
		{[]int{-2}, 0, 0}, {[]int{0}, 0, 0},
		{[]int{1, 0}, 1, 0}, {[]int{-1, 2, 1}, 0, 0},
	} {
		if _, err := NewTreeAttentionRowMask(test.parents, test.query, test.prefix); err == nil {
			t.Fatalf("invalid forest %+v admitted", test)
		}
	}
}

func TestCheckedMaskedOnlineAttentionCandidateReferenceAndReplay(t *testing.T) {
	causal, _ := NewCausalAttentionRowMask(2, 6)
	prefix, _ := NewPrefixAttentionRowMask(4, 6)
	tree, _ := NewTreeAttentionRowMask([]int{-1, 0, 0, 1, 2}, 3, 1)
	for _, mask := range []AttentionRowMask{causal, prefix, tree} {
		t.Run(mask.Kind(), func(t *testing.T) {
			snapshot := maskedRegionSnapshot(t, 6, 2)
			scores := []float32{-9, -3, 0, 3, 5, 8}
			for key, excluded := range mask.Dense() {
				if excluded {
					scores[key] = float32(math.Inf(-1))
				}
			}
			inputs := map[string][]float32{"scores": scores, "values": {1, -2, 3, 4, -5, 6, 7, -8, 9, 10, -11, 12}}
			before := slices.Clone(scores)
			refs, err := cudaref.EvaluateSnapshot(snapshot, inputs)
			if err != nil {
				t.Fatal(err)
			}
			want := refs[snapshot.Outputs()[0].Name]
			dense, err := MaterializedAttentionReference(scores, inputs["values"], mask.Dense(), 6, 2)
			if err != nil {
				t.Fatal(err)
			}
			for tile := 1; tile <= 6; tile++ {
				candidate, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, inputs, mask, tile, AttentionCandidatePolicy{Absolute: 2e-6})
				if err != nil {
					t.Fatalf("tile %d: %v", tile, err)
				}
				if !slices.Equal(candidate.Reference(), want) || !slices.Equal(inputs["scores"], before) {
					t.Fatal("reference or unchanged input mismatch")
				}
				for i, got := range candidate.Output() {
					if math.Abs(float64(got-dense[i])) > 2e-6 {
						t.Fatalf("dense mask oracle: %v != %v", candidate.Output(), dense)
					}
				}
				trace := candidate.Derivation()
				if trace.Mask == nil || !slices.Equal(trace.Mask.Dense(), mask.Dense()) || trace.Rule != "provided-input-masked-stable-online-row-v1" || trace.FullyMaskedPolicy != "canonical-nan-exact-fp32" {
					t.Fatalf("trace = %+v", trace)
				}
				state := NewOnlineAttentionState(2)
				for _, step := range trace.Tiles {
					excluded := mask.Dense()
					if err := state.StepTile(scores[step.Start:step.End], inputs["values"][step.Start*2:step.End*2], excluded[step.Start:step.End], 2); err != nil {
						t.Fatal(err)
					}
					if step.Maximum != state.M || step.Normalizer != state.D || !slices.Equal(step.UnnormalizedOutput, state.O) {
						t.Fatalf("tile state replay mismatch: %+v", step)
					}
				}
				if !slices.Equal(state.Finalize(), candidate.Output()) {
					t.Fatal("trace normalizer does not produce output")
				}
				replay, err := NewCheckedMaskedOnlineAttentionCandidate(candidate.Snapshot(), candidate.Inputs(), *trace.Mask, trace.TileSize, trace.Policy)
				if err != nil || replay.Identity() != candidate.Identity() || !slices.Equal(replay.Output(), candidate.Output()) {
					t.Fatalf("bound derivation replay: %v", err)
				}
			}
		})
	}
}

func TestCheckedMaskedOnlineAttentionCandidateRefusals(t *testing.T) {
	snapshot := maskedRegionSnapshot(t, 3, 1)
	mask, _ := NewCausalAttentionRowMask(1, 3)
	valid := map[string][]float32{"scores": {0, 0, float32(math.Inf(-1))}, "values": {3, -1, 999}}
	if _, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, valid, mask, 1, AttentionCandidatePolicy{}); err != nil {
		t.Fatal(err)
	}
	if _, err := NewCheckedOnlineAttentionCandidate(snapshot, valid, 1, AttentionCandidatePolicy{}); err == nil {
		t.Fatal("dense constructor admitted nonfinite input")
	}
	for _, test := range []struct {
		name           string
		scores, values []float32
	}{
		{"unmasked_source", []float32{0, 0, 100}, []float32{3, -1, 999}},
		{"finite_sentinel", []float32{0, 0, -math.MaxFloat32}, []float32{3, -1, 999}},
		{"included_minus_inf", []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1))}, []float32{3, -1, 999}},
		{"included_nan", []float32{float32(math.NaN()), 0, float32(math.Inf(-1))}, []float32{3, -1, 999}},
		{"excluded_nan", []float32{0, 0, float32(math.NaN())}, []float32{3, -1, 999}},
		{"excluded_plus_inf", []float32{0, 0, float32(math.Inf(1))}, []float32{3, -1, 999}},
		{"excluded_value_nan", valid["scores"], []float32{3, -1, float32(math.NaN())}},
		{"excluded_value_inf", valid["scores"], []float32{3, -1, float32(math.Inf(1))}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, map[string][]float32{"scores": test.scores, "values": test.values}, mask, 1, AttentionCandidatePolicy{Absolute: 1e9}); err == nil {
				t.Fatal("invalid source binding admitted")
			}
		})
	}
	wrong, _ := NewCausalAttentionRowMask(0, 3)
	short, _ := NewPrefixAttentionRowMask(1, 2)
	for _, bad := range []AttentionRowMask{{}, wrong, short} {
		if _, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, valid, bad, 1, AttentionCandidatePolicy{Absolute: 1e9}); err == nil {
			t.Fatal("wrong predicate admitted")
		}
	}
	// Mutation control: treating the unmasked graph as masked changes output.
	unmasked := map[string][]float32{"scores": {0, 0, 0}, "values": {3, -1, 999}}
	ref, err := cudaref.EvaluateSnapshot(snapshot, unmasked)
	if err != nil {
		t.Fatal(err)
	}
	masked, err := OnlineAttentionRow(unmasked["scores"], unmasked["values"], mask.Dense(), 3, 1, 1)
	if err != nil || masked[0] == ref[snapshot.Outputs()[0].Name][0] {
		t.Fatalf("wrong-mask control insensitive: %v", err)
	}
}

func TestCheckedMaskedOnlineAttentionCandidateFullyMasked(t *testing.T) {
	for _, n := range []int{1, 2, 5} {
		for _, d := range []int{1, 3} {
			for _, value := range []float32{-3, math.Float32frombits(0x80000000), 0, 7} {
				snapshot := maskedRegionSnapshot(t, n, d)
				mask, _ := NewPrefixAttentionRowMask(0, n)
				scores, values := make([]float32, n), make([]float32, n*d)
				for i := range scores {
					scores[i] = float32(math.Inf(-1))
				}
				for i := range values {
					values[i] = value
				}
				inputs := map[string][]float32{"scores": scores, "values": values}
				refs, err := cudaref.EvaluateSnapshot(snapshot, inputs)
				if err != nil {
					t.Fatal(err)
				}
				want := refs[snapshot.Outputs()[0].Name]
				if len(want) != d {
					t.Fatalf("source output length = %d, want %d", len(want), d)
				}
				canonical := true
				for _, value := range want {
					if math.Float32bits(value) != 0x7fc00000 {
						canonical = false
					}
				}
				for tile := 1; tile <= n; tile++ {
					for _, policy := range []AttentionCandidatePolicy{{}, {Absolute: 1e9}} {
						candidate, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, inputs, mask, tile, policy)
						if !canonical {
							if err == nil || !strings.Contains(err.Error(), "nan bits differ from source") {
								t.Fatalf("noncanonical source admitted: policy=%+v source=%v err=%v", policy, want, err)
							}
							continue
						}
						if err != nil {
							t.Fatalf("n=%d d=%d value=%v tile=%d: %v; source bits=%08x", n, d, value, tile, err, math.Float32bits(want[0]))
						}
						if len(candidate.Output()) != d {
							t.Fatalf("candidate output length = %d, want %d", len(candidate.Output()), d)
						}
						for i, got := range candidate.Output() {
							if math.Float32bits(got) != 0x7fc00000 || math.Float32bits(got) != math.Float32bits(want[i]) {
								t.Fatalf("nan policy: got %08x want %08x", math.Float32bits(got), math.Float32bits(want[i]))
							}
							// The generic zero finalizer must not substitute for graph semantics.
							if math.Float32bits(NewOnlineAttentionState(d).Finalize()[i]) == math.Float32bits(want[i]) {
								t.Fatal("zero-policy mutation survived")
							}
						}
						for _, step := range candidate.Derivation().Tiles {
							if step.Normalizer != 0 || !slices.Equal(step.UnnormalizedOutput, make([]float64, d)) {
								t.Fatal("fully excluded tile changed state")
							}
						}
					}
				}
			}
		}
	}
}

func TestCheckedMaskedOnlineAttentionCandidateGraphAuthority(t *testing.T) {
	for _, test := range []struct {
		name  string
		alter func(*cudair.Graph, map[string]*cudair.Node)
	}{
		{"mutation", func(_ *cudair.Graph, roles map[string]*cudair.Node) { roles["values"].Effect = cudair.EffectMutating }},
		{"extra_output", func(g *cudair.Graph, roles map[string]*cudair.Node) { _ = g.SetOutputs(roles["out"], roles["scores"]) }},
		{"wrong_denominator", func(_ *cudair.Graph, roles map[string]*cudair.Node) { roles["sum"].Inputs[0] = roles["sub"] }},
		{"reversed_subtraction", func(_ *cudair.Graph, roles map[string]*cudair.Node) {
			roles["sub"].Inputs[0], roles["sub"].Inputs[1] = roles["sub"].Inputs[1], roles["sub"].Inputs[0]
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			g, roles := attentionRegionGraph(3, 1)
			test.alter(g, roles)
			snapshot, err := g.Snapshot()
			if err != nil {
				return // Authoritative snapshot admission rejected the graph.
			}
			mask, _ := NewPrefixAttentionRowMask(2, 3)
			inputs := map[string][]float32{"scores": {0, 0, float32(math.Inf(-1))}, "values": {3, -1, 999}}
			if _, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, inputs, mask, 1, AttentionCandidatePolicy{Absolute: 1e9}); err == nil {
				t.Fatal("unsupported source graph admitted")
			}
		})
	}
}

func FuzzAttentionRowMaskPredicates(f *testing.F) {
	f.Add([]byte{1, 0, 0, 1}, uint8(3), uint8(1))
	f.Add([]byte{0}, uint8(0), uint8(0))
	f.Add([]byte{255, 255, 255}, uint8(1), uint8(2))
	f.Fuzz(func(t *testing.T, data []byte, queryByte, prefixByte uint8) {
		if len(data) == 0 || len(data) > 64 {
			return
		}
		prefix := int(prefixByte) % (65 - len(data))
		query := int(queryByte) % len(data)
		parents := make([]int, len(data))
		for node, value := range data {
			parents[node] = int(value)%(node+1) - 1
		}
		mask, err := NewTreeAttentionRowMask(parents, query, prefix)
		if err != nil {
			t.Fatal(err)
		}
		dense := mask.Dense()
		for key, excluded := range dense {
			allowed := key < prefix
			for node := query; node >= 0; node = parents[node] {
				allowed = allowed || key == prefix+node
			}
			if excluded == allowed {
				t.Fatalf("incorrect ancestry predicate: parents=%v query=%d prefix=%d key=%d", parents, query, prefix, key)
			}
		}
		// Corrupt a parent on any branch. Entire-forest validation must
		// reject it even if the selected query cannot reach that branch.
		bad := slices.Clone(parents)
		bad[len(bad)-1] = len(bad) - 1
		if _, err := NewTreeAttentionRowMask(bad, query, prefix); err == nil {
			t.Fatal("self-parent admitted")
		}
	})
}

func TestCheckedMaskedOnlineAttentionCandidateNumericalControls(t *testing.T) {
	snapshot := maskedRegionSnapshot(t, 4, 1)
	mask, _ := NewPrefixAttentionRowMask(3, 4)
	inputs := map[string][]float32{"scores": {-10, 0, 1, float32(math.Inf(-1))}, "values": {20, -3, 7, 999}}
	candidate, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, inputs, mask, 1, AttentionCandidatePolicy{Absolute: 1e-6})
	if err != nil {
		t.Fatal(err)
	}
	for _, mutant := range []func([]float32, []float32, []bool, int, int, int) ([]float32, error){OnlineAttentionRowMutantInverseRescale, OnlineAttentionRowMutantOmitNormalizer} {
		got, err := mutant(inputs["scores"], inputs["values"], mask.Dense(), 4, 1, 1)
		if err != nil || math.Abs(float64(got[0]-candidate.Reference()[0])) < 1e-3 {
			t.Fatalf("recurrence mutation survived: %v, %v", got, err)
		}
	}
	// The graph rounds its intermediate probabilities to FP32. Opt-in
	// tolerance does not establish equality for another set of inputs.
	inputs["scores"] = []float32{0, -0.2, float32(math.Inf(-1)), float32(math.Inf(-1))}
	mask, _ = NewPrefixAttentionRowMask(2, 4)
	inputs["values"] = []float32{3, -1, 7, 999}
	if _, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, inputs, mask, 1, AttentionCandidatePolicy{}); err == nil || !strings.Contains(err.Error(), "exact reference") {
		t.Fatalf("rounding control passed exact policy: %v", err)
	}
	if _, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, inputs, mask, 1, AttentionCandidatePolicy{Absolute: 2e-7}); err != nil {
		t.Fatal(err)
	}
}

func TestCheckedMaskedOnlineAttentionCandidateOwnershipAndIdentity(t *testing.T) {
	parents := []int{-1, 0, 0, 1}
	mask, _ := NewTreeAttentionRowMask(parents, 3, 1)
	snapshot := maskedRegionSnapshot(t, 5, 1)
	inputs := map[string][]float32{"scores": {0, 0, 0, float32(math.Inf(-1)), 0}, "values": {1, 1, 1, 999, 1}}
	candidate, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, inputs, mask, 2, AttentionCandidatePolicy{})
	if err != nil {
		t.Fatal(err)
	}
	identity := candidate.Identity()
	parents[1] = -1
	_, _, owned := mask.Definition()
	owned[0] = 99
	trace := candidate.Derivation()
	*trace.Mask = AttentionRowMask{}
	if candidate.Identity() != identity || !slices.Equal(candidate.Derivation().Mask.Dense(), mask.Dense()) {
		t.Fatal("caller changed retained predicate")
	}
	// A change outside the selected ancestor chain keeps the same dense mask,
	// but must remain identifiable in the retained predicate definition.
	other, _ := NewTreeAttentionRowMask([]int{-1, 0, -1, 1}, 3, 1)
	if !slices.Equal(other.Dense(), mask.Dense()) {
		t.Fatal("control changed active predicate")
	}
	b, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, inputs, other, 2, AttentionCandidatePolicy{})
	if err != nil || b.Identity() == identity {
		t.Fatalf("tree definition omitted from identity: %v", err)
	}
	causal, _ := NewCausalAttentionRowMask(1, 3)
	prefix, _ := NewPrefixAttentionRowMask(2, 3)
	s := maskedRegionSnapshot(t, 3, 1)
	in := map[string][]float32{"scores": {0, 0, float32(math.Inf(-1))}, "values": {3, -1, 999}}
	a, _ := NewCheckedMaskedOnlineAttentionCandidate(s, in, causal, 1, AttentionCandidatePolicy{})
	b, _ = NewCheckedMaskedOnlineAttentionCandidate(s, in, prefix, 1, AttentionCandidatePolicy{})
	if a.Identity() == b.Identity() {
		t.Fatal("mask kind omitted from identity")
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 10 {
				candidate.Derivation().Mask.Dense()[0] = true
				candidate.Inputs()["values"][0] = 123
				if candidate.Identity() != identity || !reflect.DeepEqual(candidate.Output(), []float32{1}) {
					t.Error("concurrent access changed retained candidate")
				}
			}
		})
	}
	wg.Wait()
}
