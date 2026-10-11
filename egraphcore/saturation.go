package egraphcore

import (
	"errors"
	"fmt"
	"math"
)

// SaturationLimits bounds a fixed-point run. Non-positive limits are disabled;
// work is bounded by the supplied budget.
type SaturationLimits struct {
	MaxIterations int
	MaxNodes      int
}

// SaturationStopReason describes why RunFixedPoint stopped.
type SaturationStopReason uint8

const (
	SaturationFixedPoint SaturationStopReason = iota + 1
	SaturationIterationLimit
	SaturationWorkLimit
	SaturationNodeLimit
	SaturationScheduleDone
	SaturationTupleLimit
)


var (
	// ErrInvalidSaturationInput reports a nil budget or pass callback.
	ErrInvalidSaturationInput = errors.New("egraphcore: invalid saturation input")
	// ErrIncompleteSaturationPass reports a pass that returned incomplete before
	// its work budget was exhausted.
	ErrIncompleteSaturationPass = errors.New("egraphcore: incomplete pass without exhausted work budget")
	// ErrSaturationCounterOverflow reports an unrepresentable changes or
	// iteration count.
	ErrSaturationCounterOverflow = errors.New("egraphcore: saturation counter overflow")
)

// SaturationResult summarizes a fixed-point run. Iterations counts pass
// invocations, Changes sums completed pass changes, and Work is the budget's
// recorded work at return.
type SaturationResult struct {
	Iterations int
	Changes    int
	Work       int
	Nodes      int
	StopReason SaturationStopReason
}

// SaturationPass performs one complete pass over a rule group. It returns the
// number of changes, current node count, whether the pass completed, and any
// error. If the pass is incomplete or exceeds MaxNodes, adapters must not
// commit its effects. An incomplete result is classified as a node-limit stop
// when nodes exceeds MaxNodes, otherwise as a work-limit stop when budget is
// exhausted.
type SaturationPass func(budget *WorkBudget) (changes, nodes int, complete bool, err error)

// RunFixedPoint runs pass until it completes without changes or reaches a
// configured limit. A completed pass may consume the final work unit: if it
// reports no changes, the fixed point is known. Errors from pass are returned
// with the partial result.
func RunFixedPoint(limits SaturationLimits, budget *WorkBudget, initialNodes int, pass SaturationPass) (SaturationResult, error) {
	if budget == nil || pass == nil || initialNodes < 0 {
		return SaturationResult{}, ErrInvalidSaturationInput
	}
	result := SaturationResult{Nodes: initialNodes}
	if limits.MaxNodes > 0 && initialNodes > limits.MaxNodes {
		result.StopReason = SaturationNodeLimit
		result.Work = budget.Used()
		return result, nil
	}
	for {
		if limits.MaxIterations > 0 && result.Iterations >= limits.MaxIterations {
			result.StopReason = SaturationIterationLimit
			result.Work = budget.Used()
			return result, nil
		}
		if budget.Exhausted() {
			result.StopReason = SaturationWorkLimit
			result.Work = budget.Used()
			return result, nil
		}
		if result.Iterations == math.MaxInt {
			result.StopReason = SaturationIterationLimit
			result.Work = budget.Used()
			return result, nil
		}
		result.Iterations++
		changes, nodes, complete, err := pass(budget)
		result.Work = budget.Used()
		if err != nil {
			return result, err
		}
		if changes < 0 || nodes < 0 {
			return result, fmt.Errorf("%w: negative changes or node count (%d, %d)", ErrInvalidSaturationInput, changes, nodes)
		}
		if limits.MaxNodes > 0 && nodes > limits.MaxNodes {
			result.StopReason = SaturationNodeLimit
			return result, nil
		}
		result.Nodes = nodes
		if !complete {
			if changes != 0 {
				return result, fmt.Errorf("%w: incomplete pass reported %d changes", ErrInvalidSaturationInput, changes)
			}
			if budget.Exhausted() {
				result.StopReason = SaturationWorkLimit
				return result, nil
			}
			return result, ErrIncompleteSaturationPass
		}
		if changes > math.MaxInt-result.Changes {
			return result, ErrSaturationCounterOverflow
		}
		result.Changes += changes
		if changes == 0 {
			result.StopReason = SaturationFixedPoint
			return result, nil
		}
		if limits.MaxIterations > 0 && result.Iterations >= limits.MaxIterations {
			result.StopReason = SaturationIterationLimit
			return result, nil
		}
		if budget.Exhausted() {
			result.StopReason = SaturationWorkLimit
			return result, nil
		}
	}
}
