package egraph

import (
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func unionContract(t *testing.T) TypedMetadata {
	t.Helper()
	shape, err := cudashape.New(8, 16)
	if err != nil {
		t.Fatal(err)
	}
	return TypedMetadata{Shape: shape, DType: cudair.Float32, Layout: cudashape.LayoutRowMajor, Effect: cudair.EffectReadOnly, Alignment: 16, NumericalPolicy: "exact", LoopInvariant: true}
}

func TestTypedUnionMissingMetadata(t *testing.T) {
	for _, bits := range []int{0, 1, 2} {
		t.Run(string(rune('0'+bits)), func(t *testing.T) {
			g := New()
			a := NewLegalAdapter(g)
			left, right := g.Add(NewLeaf("left")), g.Add(NewLeaf("right"))
			meta := unionContract(t)
			if bits&1 != 0 {
				if err := a.SetMetadata(left, meta); err != nil {
					t.Fatal(err)
				}
			}
			if bits&2 != 0 {
				if err := a.SetMetadata(right, meta); err != nil {
					t.Fatal(err)
				}
			}
			classes, nodes := g.Size()
			tuples := g.Facts().TupleCount()
			_, merged, err := a.LegalUnion(left, right)
			if !errors.Is(err, ErrMissingTypedMetadata) || merged || g.Find(left) == g.Find(right) {
				t.Fatal("unknown class inherited typed proof", err, merged)
			}
			c, n := g.Size()
			if c != classes || n != nodes || g.Facts().TupleCount() != tuples {
				t.Fatal("refused union changed graph/facts")
			}
			for id, want := range map[Id]bool{left: bits&1 != 0, right: bits&2 != 0} {
				_, known, err := a.GetMetadataChecked(id)
				if err != nil || known != want {
					t.Fatal("refusal changed metadata availability", known, err)
				}
			}
		})
	}
}

func TestTypedUnionProofFacts(t *testing.T) {
	tests := []struct {
		name   string
		change func(*TypedMetadata)
		want   error
	}{
		{"unknown_alignment", func(m *TypedMetadata) { m.Alignment = 0 }, ErrUnalignedVectorAccess},
		{"unknown_numerical_policy", func(m *TypedMetadata) { m.NumericalPolicy = "" }, ErrIncompatibleNumericalPolicy},
		{"different_policy", func(m *TypedMetadata) { m.NumericalPolicy = "reassociated" }, ErrIncompatibleNumericalPolicy},
		{"different_alignment", func(m *TypedMetadata) { m.Alignment = 32 }, ErrUnalignedVectorAccess},
	}
	for _, tt := range tests {
		for _, reverse := range []bool{false, true} {
			t.Run(tt.name+map[bool]string{false: "", true: "_reverse"}[reverse], func(t *testing.T) {
				g := New()
				a := NewLegalAdapter(g)
				left, right := g.Add(NewLeaf("left")), g.Add(NewLeaf("right"))
				x, y := unionContract(t), unionContract(t)
				tt.change(&y)
				if reverse {
					x, y = y, x
				}
				if err := a.SetMetadata(left, x); err != nil {
					t.Fatal(err)
				}
				if err := a.SetMetadata(right, y); err != nil {
					t.Fatal(err)
				}
				_, merged, err := a.LegalUnion(left, right)
				if !errors.Is(err, tt.want) || merged || g.Find(left) == g.Find(right) {
					t.Fatal("unproved contract was merged", err, merged)
				}
				for id, want := range map[Id]TypedMetadata{left: x, right: y} {
					got, ok, err := a.GetMetadataChecked(id)
					if err != nil || !ok || !reflect.DeepEqual(got, want) {
						t.Fatal("refused union upgraded a proof fact", got, err)
					}
				}
			})
		}
	}
}

func TestTypedUnionEffectIdentityRequired(t *testing.T) {
	for _, effect := range []cudair.StorageEffect{cudair.EffectMutating, cudair.EffectInPlaceAlias} {
		g := New()
		a := NewLegalAdapter(g)
		left, right := g.Add(NewLeaf("kv_write_before")), g.Add(NewLeaf("kv_write_after"))
		meta := unionContract(t)
		meta.Effect = effect
		if err := a.SetMetadata(left, meta); err != nil {
			t.Fatal(err)
		}
		if err := a.SetMetadata(right, meta); err != nil {
			t.Fatal(err)
		}
		if _, merged, err := a.LegalUnion(left, right); !errors.Is(err, ErrEffectBoundaryViolation) || merged || g.Find(left) == g.Find(right) {
			t.Fatal("coarse matching effects erased storage/order identity", err, merged)
		}
	}
}

func TestTypedUnionCompatibleAndNoOp(t *testing.T) {
	for _, unknown := range []bool{false, true} {
		g := New()
		a := NewLegalAdapter(g)
		left, right := g.Add(NewLeaf("left")), g.Add(NewLeaf("right"))
		x, y := unionContract(t), unionContract(t)
		if unknown {
			x.Alignment, y.Alignment = 0, 0
			x.NumericalPolicy, y.NumericalPolicy = "", ""
		}
		y.LoopInvariant = false
		if err := a.SetMetadata(left, x); err != nil {
			t.Fatal(err)
		}
		if err := a.SetMetadata(right, y); err != nil {
			t.Fatal(err)
		}
		root, merged, err := a.LegalUnion(left, right)
		if err != nil || !merged || g.Find(left) != g.Find(right) {
			t.Fatal("compatible readonly union refused", err)
		}
		got, ok, err := a.GetMetadataChecked(root)
		if err != nil || !ok || got.LoopInvariant || got.Alignment != x.Alignment || got.NumericalPolicy != x.NumericalPolicy {
			t.Fatal("merged contract changed proof facts", got, err)
		}
		if _, merged, err := a.LegalUnion(left, right); err != nil || merged {
			t.Fatal("same-class no-op refused", err)
		}
	}
	var nilAdapter *LegalAdapter
	if _, _, err := nilAdapter.LegalUnion(0, 0); err == nil {
		t.Fatal("nil adapter accepted")
	}
	g := New()
	a := NewLegalAdapter(g)
	id := g.Add(NewLeaf("untyped"))
	if _, merged, err := a.LegalUnion(id, id); err != nil || merged {
		t.Fatal("untyped no-op refused", err)
	}
}

func vectorContractState(t *testing.T, rootAlignment, sourceAlignment, columns int, layout cudashape.Layout) (*SearchState, ActionCandidate) {
	t.Helper()
	g := New()
	tokens, indices, width := g.Add(NewLeaf("tokens")), g.Add(NewLeaf("indices")), g.Add(NewLeaf(8))
	root := g.Add(NewNode("moe_gather", tokens, indices, width))
	state := NewSearchState(g, map[string]Id{"root": root})
	meta := unionContract(t)
	var err error
	meta.Alignment = rootAlignment
	meta.Shape, err = cudashape.New(8, columns)
	if err != nil {
		t.Fatal(err)
	}
	state.SetMetadata(root, meta)
	meta.Alignment = sourceAlignment
	meta.Layout = layout
	meta.Shape, err = cudashape.New(8, columns)
	if err != nil {
		t.Fatal(err)
	}
	state.SetMetadata(tokens, meta)
	meta.DType = cudair.Int32
	meta.Layout = cudashape.LayoutRowMajor
	meta.Shape, err = cudashape.New(8)
	if err != nil {
		t.Fatal(err)
	}
	state.SetMetadata(indices, meta)
	candidates := EnumerateActions(state.Graph, MandatoryRewriteRules())
	for _, candidate := range candidates {
		if candidate.Rule.Name == "vectorize_moe_gather_w16" {
			return state, candidate
		}
	}
	t.Fatal("vector rule missing")
	return nil, ActionCandidate{}
}

func TestVectorGatherSourceAlignment(t *testing.T) {
	tests := []struct {
		name                  string
		root, source, columns int
		layout                cudashape.Layout
		want                  error
	}{
		{"aligned", 16, 16, 16, cudashape.LayoutRowMajor, nil},
		{"stronger_alignment", 32, 32, 16, cudashape.LayoutRowMajor, nil},
		{"nonmultiple_result_alignment", 17, 16, 16, cudashape.LayoutRowMajor, ErrUnalignedVectorAccess},
		{"unknown_source_alignment", 16, 0, 16, cudashape.LayoutRowMajor, ErrUnalignedVectorAccess},
		{"undersized_source_alignment", 16, 8, 16, cudashape.LayoutRowMajor, ErrUnalignedVectorAccess},
		{"nonmultiple_source_alignment", 16, 17, 16, cudashape.LayoutRowMajor, ErrUnalignedVectorAccess},
		{"unaligned_row_stride", 16, 16, 3, cudashape.LayoutRowMajor, ErrUnalignedVectorAccess},
		{"column_major_source", 16, 16, 16, cudashape.LayoutColumnMajor, ErrInvalidRewriteProof},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state, candidate := vectorContractState(t, tt.root, tt.source, tt.columns, tt.layout)
			classes, nodes := state.Graph.Size()
			next, err := state.ApplyAction(candidate)
			if tt.want == nil {
				if err != nil || next == nil || !next.Merged {
					t.Fatal("aligned candidate refused", err)
				}
				return
			}
			if !errors.Is(err, tt.want) || next != nil {
				t.Fatal("unaligned vector candidate admitted", err)
			}
			c, n := state.Graph.Size()
			if c != classes || n != nodes || state.Depth != 0 || len(state.Derivation) != 0 {
				t.Fatal("refusal changed source state")
			}
		})
	}
}

func TestVectorGatherRoleContracts(t *testing.T) {
	for _, test := range []struct {
		name   string
		role   string
		change func(*TypedMetadata)
		want   error
	}{
		{"source_dtype", "?tokens", func(m *TypedMetadata) { m.DType = cudair.Int32 }, ErrIncompatibleDType},
		{"index_rank", "?indices", func(m *TypedMetadata) { m.Shape, _ = cudashape.New(8, 1) }, ErrInvalidRewriteProof},
		{"index_count", "?indices", func(m *TypedMetadata) { m.Shape, _ = cudashape.New(9) }, ErrInvalidRewriteProof},
		{"index_layout", "?indices", func(m *TypedMetadata) { m.Layout = cudashape.LayoutColumnMajor }, ErrInvalidRewriteProof},
	} {
		t.Run(test.name, func(t *testing.T) {
			state, candidate := vectorContractState(t, 16, 16, 16, cudashape.LayoutRowMajor)
			id := candidate.Match.Subst[test.role]
			meta := state.Metadata[state.Graph.Find(id)]
			test.change(&meta)
			state.SetMetadata(id, meta)
			if next, err := state.ApplyAction(candidate); !errors.Is(err, test.want) || next != nil {
				t.Fatal("incompatible gather operand role admitted", err)
			}
		})
	}
}
