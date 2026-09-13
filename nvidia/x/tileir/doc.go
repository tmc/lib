// Package tileir provides abstract syntax tree (AST) representations,
// type models, textual MLIR parsing, and binary bytecode decoding
// for CUDA Tile IR (.tilebc / .tileirbc).
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
// See https://go.dev/doc/comment for formatting references.
package tileir
