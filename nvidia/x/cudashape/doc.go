// Package cudashape provides checked multi-dimensional tensor shapes,
// overflow-checked byte size calculation, stride and layout analysis, and
// bounded symbolic dimension binding with equality and divisibility constraints.
//
// # Extent and Overflow Validation
//
// Concrete shapes are validated upon creation using [New]. Element counts
// and byte sizes are computed using overflow-checked arithmetic against
// host and device addressable memory limits ([MaxByteLimit]).
//
// # Symbolic Shapes and Constraints
//
// Dimensions may be symbolic ([NamedDim]) or constant ([ConstDim]). An [Environment]
// holds concrete bindings and constraints ([EqualSymbols], [EqualConst], [DivisibleBy],
// [RangeBound]). Bounded symbolic binding via [SymbolicShape.Bind] returns explicit
// errors on unbound symbols, contradictory bindings, or constraint violations.
// [ParseDimTokens] parses the dimension labels used in einsum equations.
//
// # Layouts and Strides
//
// Tensor layouts ([LayoutRowMajor], [LayoutColumnMajor], [LayoutStrided]) and their
// corresponding strides are computed and validated with overflow safety.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
//
// # Launch Grid Inference
//
// Launch grid extents ([Grid3D]) are computed from tensor and tile shapes via [InferGrid],
// applying overflow-checked ceiling division across ranks 1 to 3.
package cudashape
