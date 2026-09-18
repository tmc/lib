// Package tileir provides abstract syntax tree (AST) representations,
// type models, textual MLIR parsing, and binary bytecode decoding
// for CUDA Tile IR (.tilebc / .tileirbc).
//
// Decoding reads NVIDIA's files. Encoding does not produce them; see
// [Module.EncodeBytecode] before relying on its output.
//
// Tile IR represents computations at the tile and tensor-core abstraction level,
// modeling multidimensional tiles, asynchronous TMA memory transfers, token-ordered
// synchronization, and scaled matrix multiplications.
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
// [Module.EncodeBytecode] writes the header, string, type and signature
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
// encoder writes. A round trip through this package therefore proves nothing
// about the format; x/tileiras's TestAssembleNative is the gate that does, and
// it is expected to fail until the body encoding is written against NVIDIA's
// specification rather than against this package's own reader.
//
// See https://go.dev/doc/comment for formatting references.
package tileir
