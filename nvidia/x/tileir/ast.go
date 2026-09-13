package tileir

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

// Module represents a parsed or decoded Tile IR translation unit.
type Module struct {
	Name      string
	Producer  string
	Functions []*Function
	Globals   []*Global
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

// AllOperations returns a slice of all operations across all functions in the module.
func (m *Module) AllOperations() []*Operation {
	if m == nil {
		return nil
	}
	var ops []*Operation
	for _, fn := range m.Functions {
		ops = append(ops, fn.Operations...)
	}
	return ops
}

// Function represents a kernel entry point or device function in Tile IR.
type Function struct {
	Name       string
	Entry      bool // true for kernel entry points (cuda_tile.entry)
	Params     []*Param
	Returns    []string
	Blocks     []*Block
	Operations []*Operation
	Loc        Location
	LocRef     string // raw location symbol, e.g. "#loc1"
	StartLine  int
	EndLine    int
	OptHints   string
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

// FilterOps returns operations matching a predicate function.
func (f *Function) FilterOps(predicate func(*Operation) bool) []*Operation {
	if f == nil {
		return nil
	}
	var res []*Operation
	for _, op := range f.Operations {
		if predicate(op) {
			res = append(res, op)
		}
	}
	return res
}

// Param represents a function parameter.
type Param struct {
	Name    string // e.g. "%arg0"
	Type    string // e.g. "tile<ptr<f32>>"
	TypeVal Type   // parsed type representation if available
}

// Block represents a basic block within a function.
type Block struct {
	Name       string
	Operations []*Operation
}

// Global represents a module-level global variable.
type Global struct {
	Name          string
	Type          string
	TypeVal       Type
	ConstantIndex int
	Loc           Location
}

// Operation represents an individual SSA statement in Tile IR.
type Operation struct {
	Line       int      // 1-based line number in Tile IR text
	Results    []string // SSA result value names, e.g. ["%cst_16_i32"]
	OpCode     string   // mnemonic or opcode, e.g. "cuda_tile.mma_f" or "constant"
	Operands   []string // operand value names, e.g. ["%arg0", "%arg1"]
	Type       string   // result type string, e.g. "tile<16x16xf32>"
	TypeVal    Type     // parsed type value if resolved
	Attributes string   // inline attributes, e.g. "{inbounds = true}"
	Loc        Location // resolved source location
	LocRef     string   // raw location symbol, e.g. "#loc10"
	Raw        string   // full raw text of the operation line
}

// IsTMA reports whether this operation represents an asynchronous TMA memory transfer.
func (op *Operation) IsTMA() bool {
	if op == nil {
		return false
	}
	code := strings.ToLower(op.OpCode)
	return strings.Contains(code, "load_view_tko") ||
		strings.Contains(code, "store_view_tko") ||
		strings.Contains(code, "make_tensor_view") ||
		strings.Contains(code, "make_partition_view")
}

// IsMMA reports whether this operation represents a Tensor Core matrix multiplication.
func (op *Operation) IsMMA() bool {
	if op == nil {
		return false
	}
	code := strings.ToLower(op.OpCode)
	return strings.Contains(code, "mma_f") ||
		strings.Contains(code, "mma_i") ||
		strings.Contains(code, "mmaf_scaled")
}

// IsConstant reports whether this operation creates a constant value.
func (op *Operation) IsConstant() bool {
	if op == nil {
		return false
	}
	return strings.Contains(op.OpCode, "constant")
}

func (op *Operation) String() string {
	if op == nil {
		return ""
	}
	if op.Raw != "" {
		return op.Raw
	}
	var b strings.Builder
	if len(op.Results) > 0 {
		b.WriteString(strings.Join(op.Results, ", "))
		b.WriteString(" = ")
	}
	b.WriteString(op.OpCode)
	if op.Attributes != "" {
		b.WriteString(" ")
		b.WriteString(op.Attributes)
	}
	if len(op.Operands) > 0 {
		b.WriteString(" ")
		b.WriteString(strings.Join(op.Operands, ", "))
	}
	if op.Type != "" {
		b.WriteString(" : ")
		b.WriteString(op.Type)
	}
	return b.String()
}
