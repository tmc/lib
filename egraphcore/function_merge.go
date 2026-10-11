package egraphcore

import "errors"

// FunctionMergePolicy selects how two results for one function key combine.
type FunctionMergePolicy uint8

const (
	FunctionMergeKeepOld FunctionMergePolicy = iota + 1
	FunctionMergeUseNew
	FunctionMergeAssertEqual
	FunctionMergeConstant
	FunctionMergeMinInt64
	FunctionMergeMaxInt64
	FunctionMergeUnionClass
	FunctionMergeCustom
)

// FunctionMergeCallbacks supplies graph-dependent merge operations.
type FunctionMergeCallbacks[ID ~uint32] struct {
	Canonicalize func(ID) ID
	Union        func(ID, ID) (ID, error)
	Custom       func(old, incoming Value[ID]) (Value[ID], error)
}

var (
	// ErrInvalidFunctionMerge reports an invalid policy, value kind, or missing callback.
	ErrInvalidFunctionMerge = errors.New("egraphcore: invalid function merge")
	// ErrConflictingFunctionResults reports unequal results under assert-equal policy.
	ErrConflictingFunctionResults = errors.New("egraphcore: conflicting function results")
)

// MergeFunctionValue combines two results using a backend-neutral policy.
// Backends validate column sorts and canonicalize inputs before calling it.
// Union and Custom delegate operations that depend on graph state or a
// backend-defined expression language.
func MergeFunctionValue[ID ~uint32](
	policy FunctionMergePolicy,
	old, incoming, constant Value[ID],
	callbacks FunctionMergeCallbacks[ID],
) (Value[ID], error) {
	switch policy {
	case FunctionMergeKeepOld:
		return old, nil
	case FunctionMergeUseNew:
		return incoming, nil
	case FunctionMergeAssertEqual:
		if old != incoming {
			return Value[ID]{}, ErrConflictingFunctionResults
		}
		return old, nil
	case FunctionMergeConstant:
		return constant, nil
	case FunctionMergeMinInt64, FunctionMergeMaxInt64:
		if old.Kind != Int64Value || incoming.Kind != Int64Value {
			return Value[ID]{}, ErrInvalidFunctionMerge
		}
		if policy == FunctionMergeMinInt64 && old.Int64 <= incoming.Int64 ||
			policy == FunctionMergeMaxInt64 && old.Int64 >= incoming.Int64 {
			return old, nil
		}
		return incoming, nil
	case FunctionMergeUnionClass:
		if old.Kind != ClassValue || incoming.Kind != ClassValue || callbacks.Union == nil {
			return Value[ID]{}, ErrInvalidFunctionMerge
		}
		left, right := old.Class, incoming.Class
		if callbacks.Canonicalize != nil {
			left = callbacks.Canonicalize(left)
			right = callbacks.Canonicalize(right)
		}
		if left == right {
			return Class(left), nil
		}
		id, err := callbacks.Union(left, right)
		if err != nil {
			return Value[ID]{}, err
		}
		if callbacks.Canonicalize != nil {
			id = callbacks.Canonicalize(id)
		}
		return Class(id), nil
	case FunctionMergeCustom:
		if callbacks.Custom == nil {
			return Value[ID]{}, ErrInvalidFunctionMerge
		}
		return callbacks.Custom(old, incoming)
	default:
		return Value[ID]{}, ErrInvalidFunctionMerge
	}
}
