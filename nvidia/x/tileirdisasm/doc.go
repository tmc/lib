// Package tileirdisasm disassembles binary CUDA Tile IR bytecode (.tilebc)
// and parses textual Tile IR into an SSA abstract syntax tree (AST).
//
// Tile IR represents computations at the tile and tensor-core abstraction level,
// modeling matrix views, TMA transfers, token-ordered synchronization, and
// scaled matrix multiplications.
//
// The parsed AST correlates Tile IR operations and locations with virtual PTX
// and hardware SASS instructions for multi-tier profiling in cmd/cudaprof.
//
// # Usage
//
// Parse a Tile IR text module:
//
//	mod, err := tileirdisasm.ParseString(tileIRText)
//	if err != nil {
//		log.Fatal(err)
//	}
//	fn := mod.FindFunction("matrix")
//	for _, op := range fn.Operations {
//		fmt.Printf("%s = %s\n", strings.Join(op.Results, ", "), op.OpCode)
//	}
package tileirdisasm
