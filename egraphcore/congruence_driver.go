package egraphcore

import "errors"

// ErrInvalidCongruenceDriver reports a missing callback for a congruence driver.
var ErrInvalidCongruenceDriver = errors.New("egraphcore: invalid congruence driver")

// ErrCongruencePassLimit reports that the driver exhausted its pass bound
// before the policy callbacks reached a stable result.
var ErrCongruencePassLimit = errors.New("egraphcore: congruence pass limit exceeded")

// CongruenceReport summarizes the scopes and owner unions processed by a
// congruence driver.
type CongruenceReport struct {
	Passes int
	Scopes int
	Merges int
}

// DriveCongruence runs policy-provided congruence scopes until they are stable.
// next returns scope batches in order, using pass and index to restart backend
// enumeration on each outer pass. publish receives stable owner groups only;
// it may request another outer pass for backend work such as function-table
// canonicalization. If applying a scope changes graph owners, its plan is not
// published and the driver restarts enumeration immediately. Thus callbacks
// must provide canonical owners and children for the requested pass. maxPasses
// bounds outer passes, including the final pass that observes stability.
func DriveCongruence[ID ~uint32, Node any, Key comparable, Scope any](
	graph Graph[ID, Node],
	maxPasses int,
	next func(pass, index int) (scope Scope, owned []OwnedNode[ID, Node], ok bool, err error),
	key func(Node) Key,
	equal func(Node, Node) bool,
	mergeEquivalent func(Node, Node) Node,
	publish func(Scope, map[ID][]Node) (repeat bool, err error),
) (CongruenceReport, error) {
	if graphIsNil(graph) || maxPasses <= 0 || next == nil || key == nil || equal == nil || publish == nil {
		return CongruenceReport{}, ErrInvalidCongruenceDriver
	}
	var report CongruenceReport
	for pass := 0; pass < maxPasses; pass++ {
		report.Passes++
		restart := false
		repeat := false
		for index := 0; ; index++ {
			scope, owned, ok, err := next(pass, index)
			if err != nil {
				return report, err
			}
			if !ok {
				break
			}
			report.Scopes++
			plan, merged, err := ApplyCongruencePass(graph, owned, key, equal, mergeEquivalent)
			report.Merges += merged
			if err != nil {
				return report, err
			}
			if merged != 0 {
				restart = true
				break
			}
			requestRepeat, err := publish(scope, plan.ByOwner)
			if err != nil {
				return report, err
			}
			if requestRepeat {
				repeat = true
				break
			}
		}
		if !restart && !repeat {
			return report, nil
		}
	}
	return report, ErrCongruencePassLimit
}
