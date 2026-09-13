package cudaptx

import (
	"fmt"
	"strings"
)

// Location represents a source position from a .loc directive, with the file
// resolved through the module's .file table. The zero value means no location.
// Authority: docs/ptx/11-directives.md:1686-1791
type Location struct {
	File         string // resolved filename, e.g. "vectorAdd.cu"
	FileIndex    int    // .file index as written (1-based)
	Line         int    // 1-based source line
	Column       int    // 1-based source column
	Intermediate bool   // true if from .loc_intermediate directive (PTX ISA 9.4)
}

// IsDataType reports whether typ (with or without leading dot) is a recognized
// PTX fundamental, packed, or sub-byte data type (such as .f32, .b8, or .ue5m3x2).
func IsDataType(typ string) bool {
	typ = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(typ)), ".")
	_, ok := ptxDataTypes[typ]
	return ok
}

var ptxDataTypes = map[string]struct{}{
	// Fundamental integer types
	"s8": {}, "s16": {}, "s32": {}, "s64": {},
	"u8": {}, "u16": {}, "u32": {}, "u64": {},
	// Untyped bit fields
	"b8": {}, "b16": {}, "b32": {}, "b64": {}, "b128": {},
	// Floating point types
	"f16": {}, "f16x2": {}, "f32": {}, "f64": {},
	"bf16": {}, "bf16x2": {}, "tf32": {},
	// Sub-byte & microscaling formats (PTX ISA 8.x / 9.4)
	"e4m3": {}, "e4m3x2": {}, "e5m2": {}, "e5m2x2": {},
	"e2m3": {}, "e3m2": {}, "e2m1": {},
	"ue4m3": {}, "ue4m3x2": {},
	"ue5m3": {}, "ue5m3x2": {}, // Microscaling sub-byte formats
	// Predicate type
	"pred": {},
}

func (l Location) String() string {
	if l.File == "" && l.FileIndex == 0 {
		return ""
	}
	file := l.File
	if file == "" {
		file = fmt.Sprintf("<file %d>", l.FileIndex)
	}
	if l.Column > 0 {
		return fmt.Sprintf("%s:%d:%d", file, l.Line, l.Column)
	}
	return fmt.Sprintf("%s:%d", file, l.Line)
}

// Param represents a kernel or function parameter declaration.
// Authority: docs/ptx/11-directives.md:412-645
type Param struct {
	Name  string // e.g. "in1" or "addone_param_0"
	Space string // "param", "reg"
	Type  string // e.g. "u64", "f32", "b8"
	Align int    // .align byte requirement, or 0 if unspecified
	Size  int    // array element count for vector/array params, 0 if scalar
}

// Tuning retains kernel resource and launch bounds directives.
// Authority: docs/ptx/11-directives.md:925-1218
type Tuning struct {
	MaxNReg      int    // .maxnreg limit
	MaxNTID      [3]int // .maxntid [nx, ny, nz]
	ReqNTID      [3]int // .reqntid [nx, ny, nz]
	MinNCTAPerSM int    // .minnctapersm
}

// Instruction represents a single PTX instruction statement.
// Authority: docs/ptx/04-syntax.md:37-55
type Instruction struct {
	Line      int        // 1-based line number of statement's first token
	Predicate string     // guard predicate, e.g. "@%p1", "@!%p2"
	Opcode    string     // full opcode token, e.g. "fma.rn.f32"
	Mnemonic  string     // base mnemonic, e.g. "fma"
	Operands  string     // comma-separated operand text, unparsed
	Loc             Location   // nearest lexically preceding .loc; zero if none
	IntermediateLoc Location   // nearest lexically preceding .loc_intermediate; zero if none
	InlinedAt       []Location // .inlined_at chain, innermost first; nil if not inlined
	Raw             string     // full raw statement text
}

// Kernel represents a parsed .entry kernel or .func device function in PTX.
// Authority: docs/ptx/11-directives.md:403-715
type Kernel struct {
	Name         string        // kernel or function name
	Entry        bool          // true for .entry (kernel), false for .func (device function)
	Linkage      string        // "visible", "extern", "weak", "common", or ""
	Body         bool          // true if function has a body, false for bodiless declaration
	Returns      []Param       // return parameters for .func
	Params       []Param       // input parameters
	Tuning       Tuning        // occupancy and launch bounds tuning directives
	StartLine    int           // 1-based start line of function header
	EndLine      int           // 1-based end line of function body
	Instructions []Instruction // instruction sequence in execution order
}

// Module represents a fully parsed PTX source module.
// Authority: docs/ptx/11-directives.md:15-402
type Module struct {
	Version     string         // e.g. "8.8" or "8.0"
	Target      string         // e.g. "sm_90"
	AddressSize int            // 32 or 64
	Files       map[int]string // .file index -> filename map
	Kernels     []*Kernel      // list of kernel and function definitions/declarations
	kernelMap   map[string]*Kernel
}

// Kernel returns the kernel or function by name, preferring definitions over
// declarations if both exist. Returns nil if not found.
func (m *Module) Kernel(name string) *Kernel {
	if m.kernelMap == nil {
		m.kernelMap = make(map[string]*Kernel, len(m.Kernels))
		for _, k := range m.Kernels {
			if existing, ok := m.kernelMap[k.Name]; !ok || (k.Body && !existing.Body) {
				m.kernelMap[k.Name] = k
			}
		}
	}
	return m.kernelMap[name]
}
