// Package tileir provides abstract syntax tree (AST) representations,
// type models, textual MLIR parsing, and binary bytecode decoding
// for CUDA Tile IR (.tilebc / .tileirbc).
//
// Bytecode encoding has separate paths for retained function bodies and
// AST construction; see the Bytecode Encoding section for their limits.
//
// Tile IR represents computations at the tile and tensor-core abstraction level,
// modeling multidimensional tiles, asynchronous TMA memory transfers, token-ordered
// synchronization, and scaled matrix multiplications.
//
// # Ordered Reduction Lowering
//
// [LowerSemanticRegion] lowers direct-input rank-one/two FP32 sum and max
// reductions to vectors of independent outputs. Sum visits positions in order
// using FP64 accumulators and one final FP32 cast. Max retains the first equal
// value, including signed zero, and propagates NaNs. Empty sum is positive zero;
// empty max is negative infinity. The result includes the masked launch grid.
// [GenerateSemanticTileCPP] retains the separate scalar reference route.
// Source emission and host models do not qualify vendor compilation or devices.
//
// # Bytecode Decoding
//
// Binary Tile IR bytecode starts with the 8-byte magic "\x7fTileIR\x00" and
// contains modular sections for string tables, types, constant pools, global
// definitions, debug location records, and function bodies.
// The following illustrative fragment uses an io.Reader named reader:
//
//	mod, err := tileir.DecodeBytecode(reader)
//	if err != nil {
//		log.Fatal(err)
//	}
//	for _, fn := range mod.Functions {
//		fmt.Printf("kernel %s with %d operations\n", fn.Name, len(fn.Operations))
//	}
//
// # Textual Parsing
//
// Textual Tile IR is parsed into high-level SSA AST structures.
// This illustrative fragment assumes tileIRText defines matrix_mul:
//
//	mod, err := tileir.ParseString(tileIRText)
//	if err != nil {
//		log.Fatal(err)
//	}
//	fn := mod.FindFunction("matrix_mul")
//	for _, op := range fn.Operations {
//		fmt.Printf("%s = %s\n", strings.Join(op.Results, ", "), op.OpCode)
//	}
//
// # Bytecode Encoding
//
// [EncodeBytecode] serializes a [BytecodeModule]. Decoded functions retain
// their raw body bytes, which this path writes back without reconstructing
// operations. Unsupported module metadata can prevent re-encoding.
//
// [Module.EncodeBytecode] constructs bytecode through [Module.ToBytecode].
// It supports a bounded straight-line subset of numeric, token, vector and
// pointer-memory operations, encoding operands, result types and supported
// attributes. Constructs outside this subset, including nested regions, globals,
// debug locations and unknown attributes, are refused. [ErrUnsupportedASTEncoding] identifies
// unsupported AST encodings. The method returns no bytes on failure.
//
// Raw-body preservation and host round trips do not establish acceptance by
// NVIDIA's assembler or compiler, or successful device execution. Qualify
// those separately with the actual tools and targets.
//
// See https://go.dev/doc/comment for formatting references.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package tileir
