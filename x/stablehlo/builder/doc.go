// Package builder constructs StableHLO functions as MLIR text.
//
// Operations check ranks, axes and known dimensions before appending text.
// A dimension of -1 denotes an unknown size; the native compiler must validate
// runtime-dependent constraints. Constant and RMSNorm require static shapes.
// Direct edits to FuncBuilder.Ops bypass these checks.
package builder
