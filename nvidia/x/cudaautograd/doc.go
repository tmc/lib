// Package cudaautograd implements reverse-mode automatic differentiation over
// authoritative cudair graphs, gradient checks, and offline training slices.
//
// # Differentiation
//
// Reverse-mode automatic differentiation is performed by [Differentiate]. Given
// a forward [cudair.Snapshot], a loss output node, and a set of target nodes
// (typically inputs or weights), [Differentiate] constructs an explicit
// backward [cudair.Graph] without mutating the forward graph.
//
// Shared subexpressions accumulate incoming adjoint gradients via addition.
// cudair binary operations broadcast only a one-element operand, so the
// gradient of such an operand is summed over every result element; general
// broadcasts such as a length-N bias added to an M-by-N matrix are not
// representable.
// Activations required during the backward pass (such as inputs to
// nonlinearities or matmul operands) are listed in [BackwardResult].
//
// # Saved Activations
//
// [Forward] evaluates the forward graph on the CPU and returns [Activations],
// a record of the buffers the backward pass reads. [Backward] consumes it once.
// It refuses activations recorded for a different [BackwardResult] or graph
// ([ErrActivationsMismatch]) and buffers that changed length or contents after
// [Forward], or a record already consumed ([ErrSavedActivationStale]).
// A [Backward] call that finds the buffers changed consumes the record;
// one refused for nil arguments or a different result does not.
// The record covers host buffers only: keeping device buffers live and
// unwritten until a native backward invocation completes is not yet checked.
//
// # Data Types
//
// Only [cudair.Float32] is differentiated. The loss, every target, and every
// node that a gradient flows through must be float32; seeds, saved
// activations, and gradients are float32 and accumulate in float32.
// Integer types have no derivative, and [cudair.Float16] and
// [cudair.BFloat16] have no defined accumulation type yet, so [Differentiate]
// refuses them with [ErrUnsupportedDerivative].
//
// # Non-Differentiable and Mutating Operations
//
// Operations that are non-differentiable (such as reduce max or slice), or
// graphs containing state-mutating storage effects, are rejected with
// explicit errors ([ErrUnsupportedDerivative], [ErrUnsupportedMutation]).
//
// # Offline Training
//
// Offline parameter updates can be executed via [StepSGDMomentum] or evaluated
// end-to-end on CPU or native execution plans.
package cudaautograd
