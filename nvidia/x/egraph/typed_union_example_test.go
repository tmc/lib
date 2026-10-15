package egraph_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleLegalAdapter_LegalUnion() {
	g := egraph.New()
	a := egraph.NewLegalAdapter(g)
	// A frontend supplies equivalent forms; this adapter checks their contracts,
	// rather than proving equality from the leaf names.
	left, right := g.Add(egraph.NewLeaf("left")), g.Add(egraph.NewLeaf("right"))
	shape, err := cudashape.New(4)
	if err != nil {
		fmt.Println(err)
		return
	}
	meta := egraph.TypedMetadata{Shape: shape, DType: cudair.Float32, Layout: cudashape.LayoutRowMajor, Effect: cudair.EffectReadOnly, NumericalPolicy: "exact"}
	for _, id := range []egraph.Id{left, right} {
		if err := a.SetMetadata(id, meta); err != nil {
			fmt.Println(err)
			return
		}
	}
	_, merged, err := a.LegalUnion(left, right)
	fmt.Println(merged, err)
	// Output: true <nil>
}
