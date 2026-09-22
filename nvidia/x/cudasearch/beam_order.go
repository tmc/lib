package cudasearch

import (
	"cmp"
	"slices"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// Compare fields rather than display strings, which may contain delimiters.
// Identical paths retain enumeration order through the stable frontier sort.
func compareDerivations(a, b []egraph.Action) int {
	return slices.CompareFunc(a, b, func(a, b egraph.Action) int {
		for _, pair := range [][2]string{{a.RuleName, b.RuleName}, {a.RuleID, b.RuleID}, {a.Source, b.Source}, {a.RHSNode, b.RHSNode}} {
			if n := cmp.Compare(pair[0], pair[1]); n != 0 {
				return n
			}
		}
		if n := cmp.Compare(a.ClassID, b.ClassID); n != 0 {
			return n
		}
		if n := cmp.Compare(a.StepIndex, b.StepIndex); n != 0 {
			return n
		}
		keys := func(subst egraph.Subst) []string {
			out := make([]string, 0, len(subst))
			for key := range subst {
				out = append(out, key)
			}
			slices.Sort(out)
			return out
		}
		ak, bk := keys(a.Subst), keys(b.Subst)
		for i := 0; i < min(len(ak), len(bk)); i++ {
			if n := cmp.Compare(ak[i], bk[i]); n != 0 {
				return n
			}
			if n := cmp.Compare(a.Subst[ak[i]], b.Subst[bk[i]]); n != 0 {
				return n
			}
		}
		return cmp.Compare(len(ak), len(bk))
	})
}
