package egraphcore

import (
	"errors"
	"math"
	"reflect"
)

// ErrRewriteWorkBudget reports that applying prepared rewrite matches
// exhausted its work budget.
var ErrRewriteWorkBudget = errors.New("egraphcore: rewrite work budget exceeded")

// ErrRewriteNodeLimit reports that an application admission check rejected a
// match because its conservative node-growth bound exceeds the graph limit.
var ErrRewriteNodeLimit = errors.New("egraphcore: rewrite node limit exceeded")

// RewriteApplicationReport summarizes one ordered batch of prepared term
// rewrites.
type RewriteApplicationReport struct {
	Applied int
	Merges  int
	Rebuilt bool
}

// ApplyMatches instantiates each prepared match in order, unions the result
// with its matched class, and optionally rebuilds once after the batch. Match
// search, validation, transactional staging, and backend side effects remain
// the caller's responsibility. A nil budget is unlimited; otherwise the
// budget charges one unit per match, estimated work for instantiate and admit,
// and one for a possible rebuild. instantiationWork must be deterministic,
// side-effect free, and return a conservative non-negative work estimate for
// those callbacks. The estimate callback itself and graph callback work are
// not included. admit, when non-nil, checks each match immediately before its
// RHS is instantiated. All charges are reserved before the first mutation. On
// error, the report describes mutations completed before the error; callers
// that require rollback must apply the batch to a staged graph and publish it
// only on success.
func ApplyMatches[ID ~uint32, Node, Match any](
	graph Graph[ID, Node],
	matches []Match,
	class func(Match) ID,
	instantiate func(Match) (ID, error),
	instantiationWork func(Match) int,
	admit func(Match) error,
	budget *WorkBudget,
	rebuild bool,
) (RewriteApplicationReport, error) {
	if graphIsNil(graph) || class == nil || instantiate == nil || instantiationWork == nil {
		return RewriteApplicationReport{}, ErrInvalidRewriteApplication
	}
	var report RewriteApplicationReport
	spend := func(units int) bool { return budget == nil || budget.TrySpend(units) }
	// Reserve application and possible rebuild work before the first mutation,
	// so budget exhaustion cannot leave a graph half-applied.
	reservation := len(matches)
	for _, match := range matches {
		work := instantiationWork(match)
		if work < 0 {
			return report, ErrInvalidRewriteApplication
		}
		if work > math.MaxInt-reservation {
			return report, ErrRewriteWorkBudget
		}
		reservation += work
	}
	if rebuild && len(matches) > 0 {
		if reservation == math.MaxInt {
			return report, ErrRewriteWorkBudget
		}
		reservation++
	}
	if !spend(reservation) {
		return report, ErrRewriteWorkBudget
	}
	for _, match := range matches {
		if admit != nil {
			if err := admit(match); err != nil {
				return report, err
			}
		}
		rhs, err := instantiate(match)
		if err != nil {
			return report, err
		}
		if _, merged, err := graph.UnionChecked(class(match), rhs); err != nil {
			return report, err
		} else if merged {
			report.Merges++
		}
		report.Applied++
	}
	if rebuild && report.Merges > 0 {
		if err := graph.RebuildChecked(); err != nil {
			return report, err
		}
		report.Rebuilt = true
	}
	return report, nil
}

func graphIsNil[ID ~uint32, Node any](graph Graph[ID, Node]) bool {
	if graph == nil {
		return true
	}
	value := reflect.ValueOf(graph)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice, reflect.UnsafePointer:
		return value.IsNil()
	default:
		return false
	}
}

// ErrInvalidRewriteApplication reports an invalid graph or missing callback.
var ErrInvalidRewriteApplication = errors.New("egraphcore: invalid rewrite application")
