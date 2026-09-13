package tileirdisasm

import (
	"github.com/tmc/lib/nvidia/x/tileir"
)

// Re-export core Tile IR AST types from the canonical x/tileir package.
type (
	// Location represents a source code coordinate attached to an operation or function.
	Location = tileir.Location

	// Module represents a parsed Tile IR translation unit.
	Module = tileir.Module

	// Function represents a kernel entry point or device function in Tile IR.
	Function = tileir.Function

	// Param represents a function parameter.
	Param = tileir.Param

	// Block represents a basic block within a function.
	Block = tileir.Block

	// Operation represents an individual SSA statement in Tile IR.
	Operation = tileir.Operation
)
