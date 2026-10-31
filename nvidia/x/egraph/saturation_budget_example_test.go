package egraph_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// ExampleSaturationOptions_ruleMatches caps one rule's matches and reports
// which rule stopped saturation.
func ExampleSaturationOptions_ruleMatches() {
	g := egraph.New()
	for _, leaf := range []string{"a", "b", "c"} {
		g.Add(egraph.NewNode("f", g.Add(egraph.NewLeaf(leaf))))
	}
	rules := []*egraph.Rewrite{egraph.MustRewrite("fuse", "(f ?x)", "(g ?x)")}
	res, err := egraph.SaturateWithReport(g, rules, egraph.SaturationOptions{
		RuleMatches: map[string]int{"fuse": 2},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("stopped on %v at rule %q\n", res.StopReason, res.StopRule)
	for _, st := range res.Rules {
		fmt.Printf("%s: found %d, applied %d, capped %v\n", st.Name, st.Found, st.Applied, st.Capped)
	}
	// Output:
	// stopped on rule match limit at rule "fuse"
	// fuse: found 3, applied 2, capped true
}
