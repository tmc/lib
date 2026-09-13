package tileirdisasm

import (
	"fmt"
	"strings"
)

// Location represents a source code coordinate attached to an operation or function.
type Location struct {
	File   string // resolved source file path
	Line   int    // 1-based source line
	Column int    // 1-based source column
	Scope  string // subprogram or lexical block name
}

func (l Location) String() string {
	if l.File == "" && l.Line == 0 {
		return ""
	}
	if l.Column > 0 {
		return fmt.Sprintf("%s:%d:%d", l.File, l.Line, l.Column)
	}
	return fmt.Sprintf("%s:%d", l.File, l.Line)
}

// Module represents a parsed Tile IR translation unit.
type Module struct {
	Producer  string
	Functions []*Function
	Comments  []string
	Locations map[string]Location
}

// FindFunction returns the function with the given symbol name, or nil if not found.
func (m *Module) FindFunction(name string) *Function {
	if m == nil {
		return nil
	}
	name = strings.TrimPrefix(name, "@")
	for _, fn := range m.Functions {
		if fn.Name == name {
			return fn
		}
	}
	return nil
}

// Function represents a kernel entry point or device function in Tile IR.
type Function struct {
	Name       string
	Entry      bool // true for entry kernels
	Params     []*Param
	Returns    []string
	Blocks     []*Block
	Operations []*Operation
	Loc        Location
	StartLine  int
	EndLine    int
}

// OperationAtLine returns the operation located on the given line number, or nil.
func (f *Function) OperationAtLine(line int) *Operation {
	if f == nil {
		return nil
	}
	for _, op := range f.Operations {
		if op.Line == line {
			return op
		}
	}
	return nil
}

// Param represents a function parameter.
type Param struct {
	Name string // e.g. "%arg0"
	Type string // e.g. "tile<ptr<f32>>"
}

// Block represents a basic block within a function.
type Block struct {
	Name       string
	Operations []*Operation
}

// Operation represents an individual SSA statement in Tile IR.
type Operation struct {
	Line       int        // 1-based line number in Tile IR text
	Results    []string   // SSA result value names, e.g. ["%cst_16_i32"] or ["%for:5"]
	OpCode     string     // base mnemonic/opcode, e.g. "constant", "cuda_tile.mmaf_scaled", "return"
	Operands   []string   // operand value names, e.g. ["%arg0", "%arg1"]
	Type       string     // operation result type or signature, e.g. "tile<i32>"
	Attributes string     // inline attributes, e.g. "<i32: 16>" or "{inbounds = true}"
	Loc        Location   // resolved source location
	LocRef     string     // raw location symbol, e.g. "#loc10"
	Raw        string     // full raw text of the operation line
}
