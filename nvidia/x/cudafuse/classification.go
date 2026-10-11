package cudafuse

import "strings"

// OpClassification categorizes an operator's role in the fusion engine.
type OpClassification int

const (
	// OpUnknown denotes an unrecognized operator.
	OpUnknown OpClassification = iota

	// OpFusibleElementwise denotes an operation that can be chained without
	// intermediate memory allocation or host dispatch.
	OpFusibleElementwise

	// OpMaterializingBoundary denotes an operation (such as reduction, scan, or host transfer)
	// that requires flushing pending fused kernels.
	OpMaterializingBoundary

	// OpConditionallyFusedCompaction denotes a size-altering or predicate-masking operation
	// (such as where, filter, or keep) that forms a compaction boundary.
	OpConditionallyFusedCompaction
)

// ClassifyOp returns the classification of an operation name.
func ClassifyOp(op string) OpClassification {
	switch strings.ToLower(strings.TrimSpace(op)) {
	case "add", "mul", "sub", "div", "relu", "silu", "exp", "sqrt", "neg", "reshape", "transpose":
		return OpFusibleElementwise
	case "reduce", "sum", "scan", "sort", "to_host", "gather", "cycle", "replicate":
		return OpMaterializingBoundary
	case "where", "filter", "keep":
		return OpConditionallyFusedCompaction
	default:
		return OpUnknown
	}
}

// IsFusibleElementwise reports whether op can be lazily fused into an elementwise mega-kernel.
func IsFusibleElementwise(op string) bool {
	return ClassifyOp(op) == OpFusibleElementwise
}

// IsMaterializingBoundary reports whether op forces kernel materialization and execution.
func IsMaterializingBoundary(op string) bool {
	return ClassifyOp(op) == OpMaterializingBoundary
}

// IsCompactionBoundary reports whether op alters output shape or requires predicate compaction.
func IsCompactionBoundary(op string) bool {
	return ClassifyOp(op) == OpConditionallyFusedCompaction
}
