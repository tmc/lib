// Package cudaautograd implements reverse-mode automatic differentiation over
// authoritative cudair graphs, gradient checks, and offline training slices.
//
// # Differentiation
//
// Reverse-mode automatic differentiation is performed by [Differentiate]. Given
// a forward [cudair.Graph] or [cudair.Snapshot], a loss output node, and a set of
// target nodes (typically inputs or weights), [Differentiate] constructs an explicit
// backward [cudair.Graph] without mutating the original forward graph.
//
// Shared subexpressions accumulate incoming adjoint gradients via addition.
// Broadcast operations (such as bias additions) reduce gradients along the
// broadcast axes. Activations required during the backward pass (such as inputs
// to nonlinearities or matmul operands) are preserved and tracked in [BackwardResult].
//
// # Non-Differentiable and Mutating Operations
//
// Operations that are non-differentiable (such as integer operations or discrete
// reductions), or graphs containing state-mutating storage effects, are rejected
// with explicit errors ([ErrUnsupportedDerivative], [ErrUnsupportedMutation]).
//
// # Offline Training
//
// Offline parameter updates can be executed via [StepSGDMomentum] or evaluated
// end-to-end on CPU or native execution plans.
package cudaautograd
