package egraph

import "testing"

func TestLeafSignaturesDisambiguatedByResultSort(t *testing.T) {
	g := New()
	for _, sort := range []string{"float32", "float16"} {
		if err := g.RegisterOperator("input", OperatorSignature{Result: sort}); err != nil {
			t.Fatal(err)
		}
	}
	for _, sort := range []string{"float32", "float16"} {
		if _, err := g.AddChecked(NewSortedNode(sort, "input")); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := g.AddChecked(NewSortedNode("int32", "input")); err == nil {
		t.Fatal("accepted unregistered result sort")
	}
}

func TestSignatureWildcardAmbiguityStillRejected(t *testing.T) {
	g := New()
	if err := g.RegisterOperator("op", OperatorSignature{Result: "float32", Children: []string{""}}); err != nil {
		t.Fatal(err)
	}
	if err := g.RegisterOperator("op", OperatorSignature{Result: "float32", Children: []string{"float32"}}); err == nil {
		t.Fatal("accepted same-result wildcard ambiguity")
	}
}
