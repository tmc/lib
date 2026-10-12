// Package tileir provides abstract syntax tree (AST) representations,
// type models, textual MLIR parsing, and binary bytecode decoding
// for CUDA Tile IR (.tilebc / .tileirbc).
//
// Decoding reads NVIDIA's files. There are two encoding paths and only one
// of them does: see the Bytecode Encoding section before relying on either.
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
// Textual Tile IR is parsed into high-level SSA AST structures:
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
// [EncodeBytecode] serializes a [BytecodeModule] and is faithful: it
// reproduces each vendor golden byte for byte, because a decoded function
// keeps its body bytes and the serializer writes them back unchanged. Use it
// to rewrite bytecode that NVIDIA produced.
//
// [Module.EncodeBytecode] builds a body from an AST instead, and that path is
// the broken one. It writes the header, string, type and signature
// sections, but a function body it writes is not Tile IR: it holds one opcode
// varint per operation, with a result type for the constant opcode and nothing
// for any other. Operands, result types and attributes are absent. The real
// assembler rejects such a body at the first operation that needs a result
// type, which is nearly every one:
//
//	error at offset 2: failed to get result type 0 for MakeTokenOp
//	input does not correspond to Tile IR bytecode
//
// [DecodeBytecode] reads that body back, because it reads exactly what the
// encoder writes, and it does not reject an opcode it does not know. A round
// trip through this package therefore proves nothing about the format: both
// round-trip tests stay green when the encoder is made to write every opcode
// as opcode+1. The golden re-encode test cannot see this path at all, since it
// starts from vendor bytes and never builds a body. x/tileiras's
// TestAssembleNative is the only gate that can, and it is expected to fail
// until the body encoding is written against NVIDIA's specification rather
// than against this package's own reader.
//
// See https://go.dev/doc/comment for formatting references.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package tileir
