package cudaptx

import (
	"fmt"
)

// Location represents a source position from a .loc directive, with the file
// resolved through the module's .file table. The zero value means no location.
// Authority: docs/ptx/11-directives.md:1686-1791
type Location struct {
	File      string // resolved filename, e.g. "vectorAdd.cu"
	FileIndex int    // .file index as written (1-based)
	Line      int    // 1-based source line
	Column    int    // 1-based source column
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
	Loc       Location   // nearest lexically preceding .loc; zero if none
	InlinedAt []Location // .inlined_at chain, innermost first; nil if not inlined
	Raw       string     // full raw statement text
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
