package cudaptx_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// parseFixture parses one of the real nvcc outputs in testdata. See
// testdata/README for how they are generated and what each covers.
func parseFixture(t *testing.T, name string) *cudaptx.Module {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	m, err := cudaptx.Parse(f)
	if err != nil {
		t.Fatalf("Parse(%s): %v", name, err)
	}
	return m
}

func TestFixtures(t *testing.T) {
	type kernel struct {
		name    string
		entry   bool
		body    bool
		linkage string
		returns int
		params  int
		insts   int
	}
	tests := []struct {
		file        string
		version     string
		target      string
		addressSize int
		files       map[int]string
		kernels     []kernel
	}{{
		file:        "minimal.sm90.ptx",
		version:     "9.0",
		target:      "sm_90",
		addressSize: 64,
		files:       map[int]string{},
		kernels: []kernel{
			{"addone", true, true, "visible", 0, 1, 9},
		},
	}, {
		file:        "vector.sm90.ptx",
		version:     "9.0",
		target:      "sm_90",
		addressSize: 64,
		files:       map[int]string{1: "/tmp/cudaptx-fixtures/vector.cu"},
		kernels: []kernel{
			{"vec4", true, true, "visible", 0, 3, 21},
		},
	}, {
		file:        "lexer.sm90.ptx",
		version:     "9.0",
		target:      "sm_90",
		addressSize: 64,
		files:       map[int]string{1: "/tmp/cudaptx-fixtures/lexer.cu"},
		kernels: []kernel{
			{"externalHelper", false, false, "extern", 1, 1, 0},
			{"notInlined", false, true, "visible", 1, 1, 7},
			{"guarded", true, true, "visible", 0, 4, 19},
			{"bounded", true, true, "visible", 0, 1, 9},
			{"shaped3d", true, true, "visible", 0, 1, 14},
			{"byvalue", true, true, "visible", 0, 2, 34},
			{"calls", true, true, "visible", 0, 1, 11},
		},
	}, {
		// Same source, different architecture: byvalue lowers to one
		// instruction fewer. The parser must depend on neither.
		file:        "lexer.sm121a.ptx",
		version:     "9.0",
		target:      "sm_121a",
		addressSize: 64,
		files:       map[int]string{1: "/tmp/cudaptx-fixtures/lexer.cu"},
		kernels: []kernel{
			{"externalHelper", false, false, "extern", 1, 1, 0},
			{"notInlined", false, true, "visible", 1, 1, 7},
			{"guarded", true, true, "visible", 0, 4, 19},
			{"bounded", true, true, "visible", 0, 1, 9},
			{"shaped3d", true, true, "visible", 0, 1, 14},
			{"byvalue", true, true, "visible", 0, 2, 33},
			{"calls", true, true, "visible", 0, 1, 11},
		},
	}}

	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			m := parseFixture(t, tt.file)
			if m.Version != tt.version {
				t.Errorf("Version = %q, want %q", m.Version, tt.version)
			}
			if m.Target != tt.target {
				t.Errorf("Target = %q, want %q", m.Target, tt.target)
			}
			if m.AddressSize != tt.addressSize {
				t.Errorf("AddressSize = %d, want %d", m.AddressSize, tt.addressSize)
			}
			if len(m.Files) != len(tt.files) {
				t.Errorf("Files = %v, want %v", m.Files, tt.files)
			}
			for idx, want := range tt.files {
				if got := m.Files[idx]; got != want {
					t.Errorf("Files[%d] = %q, want %q", idx, got, want)
				}
			}
			if len(m.Kernels) != len(tt.kernels) {
				var got []string
				for _, k := range m.Kernels {
					got = append(got, k.Name)
				}
				t.Fatalf("parsed %d kernels %v, want %d", len(m.Kernels), got, len(tt.kernels))
			}
			for i, want := range tt.kernels {
				k := m.Kernels[i]
				got := kernel{k.Name, k.Entry, k.Body, k.Linkage, len(k.Returns), len(k.Params), len(k.Instructions)}
				if got != want {
					t.Errorf("kernel %d = %+v, want %+v", i, got, want)
				}
			}
		})
	}
}

// TestFileTableFollowsItsUses pins the property that motivates deferred
// filename resolution: nvcc emits .file at the end of the module, after every
// .loc that references it. The fixture is checked too, so that a regenerated
// fixture cannot silently stop covering this.
func TestFileTableFollowsItsUses(t *testing.T) {
	const name = "lexer.sm90.ptx"
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	firstLoc, lastFile := -1, -1
	for i, line := range strings.Split(string(data), "\n") {
		switch {
		case strings.HasPrefix(strings.TrimSpace(line), ".loc") && firstLoc < 0:
			firstLoc = i
		case strings.HasPrefix(strings.TrimSpace(line), ".file"):
			lastFile = i
		}
	}
	if firstLoc < 0 || lastFile < 0 {
		t.Fatalf("%s: found .loc at %d and .file at %d; fixture no longer covers deferred resolution", name, firstLoc, lastFile)
	}
	if lastFile < firstLoc {
		t.Fatalf("%s: .file at line %d precedes the first .loc at line %d; fixture no longer covers deferred resolution", name, lastFile+1, firstLoc+1)
	}

	m := parseFixture(t, name)
	k := m.Kernel("guarded")
	if k == nil {
		t.Fatal("kernel guarded not found")
	}
	for _, inst := range k.Instructions {
		if inst.Loc.FileIndex != 0 && inst.Loc.File == "" {
			t.Fatalf("%s: instruction %q has FileIndex %d but unresolved File", name, inst.Opcode, inst.Loc.FileIndex)
		}
	}
}

func TestGuardedKernelDetail(t *testing.T) {
	m := parseFixture(t, "lexer.sm90.ptx")
	k := m.Kernel("guarded")
	if k == nil {
		t.Fatal("kernel guarded not found")
	}

	// Guard predicate: docs/ptx/04-syntax.md:82-85.
	var bra *cudaptx.Instruction
	for i := range k.Instructions {
		if k.Instructions[i].Mnemonic == "bra" {
			bra = &k.Instructions[i]
			break
		}
	}
	if bra == nil {
		t.Fatal("no bra instruction")
	}
	if bra.Predicate != "@%p1" {
		t.Errorf("bra Predicate = %q, want %q", bra.Predicate, "@%p1")
	}
	if bra.Operands != "$L__BB1_2" {
		t.Errorf("bra Operands = %q, want %q", bra.Operands, "$L__BB1_2")
	}

	// An inlined __forceinline__ callee: .loc points into the callee and
	// inlined_at at the call site (docs/ptx/11-directives.md:1686-1791).
	var fma *cudaptx.Instruction
	for i := range k.Instructions {
		if k.Instructions[i].Mnemonic == "fma" {
			fma = &k.Instructions[i]
			break
		}
	}
	if fma == nil {
		t.Fatal("no fma instruction")
	}
	if fma.Opcode != "fma.rn.f32" {
		t.Errorf("fma Opcode = %q, want %q", fma.Opcode, "fma.rn.f32")
	}
	if fma.Loc.Line != 6 || fma.Loc.Column != 3 {
		t.Errorf("fma Loc = %v, want line 6 column 3", fma.Loc)
	}
	if len(fma.InlinedAt) != 1 {
		t.Fatalf("fma InlinedAt = %v, want one frame", fma.InlinedAt)
	}
	if fma.InlinedAt[0].Line != 15 || fma.InlinedAt[0].File != "/tmp/cudaptx-fixtures/lexer.cu" {
		t.Errorf("fma InlinedAt[0] = %v, want lexer.cu line 15", fma.InlinedAt[0])
	}

	// Instructions that are not inlined carry no frames.
	for _, inst := range k.Instructions {
		if inst.Mnemonic != "fma" && inst.InlinedAt != nil {
			t.Errorf("%s has InlinedAt %v, want none", inst.Opcode, inst.InlinedAt)
		}
	}

	// Struct-by-value parameters: .param .align 4 .b8 name[36].
	bv := m.Kernel("byvalue")
	if bv == nil {
		t.Fatal("kernel byvalue not found")
	}
	want := []cudaptx.Param{
		{Name: "byvalue_param_0", Space: "param", Type: "b8", Align: 4, Size: 36},
		{Name: "byvalue_param_1", Space: "param", Type: "u64"},
	}
	if len(bv.Params) != len(want) {
		t.Fatalf("byvalue Params = %+v, want %+v", bv.Params, want)
	}
	for i := range want {
		if bv.Params[i] != want[i] {
			t.Errorf("byvalue Params[%d] = %+v, want %+v", i, bv.Params[i], want[i])
		}
	}
}

// TestMultiLineCallStatement checks that a statement split across five lines
// becomes one instruction. nvcc writes call.uni that way
// (testdata/lexer.sm90.ptx).
func TestMultiLineCallStatement(t *testing.T) {
	m := parseFixture(t, "lexer.sm90.ptx")
	k := m.Kernel("calls")
	if k == nil {
		t.Fatal("kernel calls not found")
	}
	var calls []cudaptx.Instruction
	for _, inst := range k.Instructions {
		if inst.Mnemonic == "call" {
			calls = append(calls, inst)
		}
	}
	if len(calls) != 1 {
		t.Fatalf("got %d call instructions, want 1: %+v", len(calls), calls)
	}
	if calls[0].Opcode != "call.uni" {
		t.Errorf("Opcode = %q, want %q", calls[0].Opcode, "call.uni")
	}
	if got, want := calls[0].Operands, "(retval0), notInlined, ( param0 )"; got != want {
		t.Errorf("Operands = %q, want %q", got, want)
	}
}

// TestVectorOperandsAreNotBlocks checks that a brace-delimited operand list is
// not read as a block delimiter.
func TestVectorOperandsAreNotBlocks(t *testing.T) {
	m := parseFixture(t, "vector.sm90.ptx")
	k := m.Kernel("vec4")
	if k == nil {
		t.Fatal("kernel vec4 not found")
	}
	var ld, st *cudaptx.Instruction
	for i := range k.Instructions {
		switch k.Instructions[i].Opcode {
		case "ld.global.v4.f32":
			ld = &k.Instructions[i]
		case "st.global.v4.f32":
			st = &k.Instructions[i]
		}
	}
	if ld == nil || st == nil {
		t.Fatalf("missing v4 instructions: ld=%v st=%v", ld, st)
	}
	if got, want := ld.Operands, "{%f1, %f2, %f3, %f4}, [%rd5]"; got != want {
		t.Errorf("ld Operands = %q, want %q", got, want)
	}
	if got, want := st.Operands, "[%rd7], {%f12, %f11, %f10, %f9}"; got != want {
		t.Errorf("st Operands = %q, want %q", got, want)
	}
}

// TestTuningBetweenHeaderAndBody covers the tuning directives, which nvcc
// writes unterminated between a header's ")" and its "{"
// (docs/ptx/11-directives.md:925-1218).
func TestTuningBetweenHeaderAndBody(t *testing.T) {
	m := parseFixture(t, "lexer.sm90.ptx")
	tests := []struct {
		kernel string
		want   cudaptx.Tuning
	}{
		{"bounded", cudaptx.Tuning{MaxNTID: [3]int{256, 1, 1}, MinNCTAPerSM: 4}},
		{"shaped3d", cudaptx.Tuning{MaxNTID: [3]int{64, 1, 1}}},
		{"guarded", cudaptx.Tuning{}},
	}
	for _, tt := range tests {
		k := m.Kernel(tt.kernel)
		if k == nil {
			t.Fatalf("kernel %s not found", tt.kernel)
		}
		if k.Tuning != tt.want {
			t.Errorf("%s Tuning = %+v, want %+v", tt.kernel, k.Tuning, tt.want)
		}
	}
}

// TestEveryMnemonicIsAKeyword audits the lexer against the generated ISA table.
// A mnemonic that is not a reserved keyword means the statement splitter
// produced something that is not an instruction.
func TestEveryMnemonicIsAKeyword(t *testing.T) {
	entries, err := filepath.Glob(filepath.Join("testdata", "*.ptx"))
	if err != nil || len(entries) == 0 {
		t.Fatalf("no fixtures: %v", err)
	}
	for _, path := range entries {
		m := parseFixture(t, filepath.Base(path))
		for _, k := range m.Kernels {
			for _, inst := range k.Instructions {
				if !cudaptx.IsInstructionKeyword(inst.Mnemonic) {
					t.Errorf("%s: %s: mnemonic %q is not a PTX %s keyword (raw %q)",
						filepath.Base(path), k.Name, inst.Mnemonic, cudaptx.PTXISAVersion, inst.Raw)
				}
			}
		}
	}
}

// The cases below use hand-written PTX only for input nvcc does not emit: block
// comments, preprocessor lines, a header and its brace on one line, a negated
// guard predicate, and the two tuning directives. Each was checked against all
// four fixtures before being written here, and anything a compiler does emit is
// asserted from testdata instead -- register and local declarations, for one,
// are covered there by the instruction counts in TestFixtures.
func TestHandWrittenEdgeCases(t *testing.T) {
	tests := []struct {
		name  string
		ptx   string
		check func(*testing.T, *cudaptx.Module)
	}{{
		name: "BlockCommentSpanningLines",
		ptx: `.version 8.0
.target sm_90
.visible .entry k() {
	/* a comment
	   that spans lines and contains ; and { and }
	   and a fake instruction: add.f32 %f1, %f2, %f3;
	*/
	ret;
}`,
		check: func(t *testing.T, m *cudaptx.Module) {
			k := m.Kernel("k")
			if k == nil {
				t.Fatal("kernel k not found")
			}
			if len(k.Instructions) != 1 || k.Instructions[0].Mnemonic != "ret" {
				t.Errorf("Instructions = %+v, want one ret", k.Instructions)
			}
		},
	}, {
		name: "PreprocessorLinesSkipped",
		ptx: `#line 1 "k.cu"
.version 8.0
#define UNUSED 1
.visible .entry k() {
	ret;
}`,
		check: func(t *testing.T, m *cudaptx.Module) {
			if m.Version != "8.0" {
				t.Errorf("Version = %q, want 8.0", m.Version)
			}
			if k := m.Kernel("k"); k == nil || len(k.Instructions) != 1 {
				t.Errorf("kernel k = %+v, want one instruction", k)
			}
		},
	}, {
		name: "HeaderAndBraceOnOneLine",
		ptx: `.version 8.0
.visible .entry k(.param .u64 p) { ret; }`,
		check: func(t *testing.T, m *cudaptx.Module) {
			k := m.Kernel("k")
			if k == nil {
				t.Fatal("kernel k not found")
			}
			if !k.Body || len(k.Params) != 1 || k.Params[0].Name != "p" {
				t.Errorf("kernel = %+v, want body with param p", k)
			}
		},
	}, {
		// @!p is in the ISA (04-syntax.md:85) but unreachable from nvcc: across 27
		// C constructs x 7 architectures x 6 option sets, every guard was plain
		// @%p, because NVVM inverts the setp comparison rather than the guard.
		name: "NegatedPredicate",
		ptx: `.version 8.0
.visible .entry k() {
	@!%p2 bra $L1;
	ret;
}`,
		check: func(t *testing.T, m *cudaptx.Module) {
			k := m.Kernel("k")
			if len(k.Instructions) != 2 {
				t.Fatalf("Instructions = %+v, want 2", k.Instructions)
			}
			if got := k.Instructions[0].Predicate; got != "@!%p2" {
				t.Errorf("Predicate = %q, want %q", got, "@!%p2")
			}
		},
	}, {
		// .maxnreg and .reqntid are not emitted by nvcc; see testdata/README.
		name: "MaxNRegAndReqNTID",
		ptx: `.version 8.0
.visible .entry k()
.maxnreg 64
.reqntid 16, 16
{
	ret;
}`,
		check: func(t *testing.T, m *cudaptx.Module) {
			k := m.Kernel("k")
			if k == nil {
				t.Fatal("kernel k not found")
			}
			want := cudaptx.Tuning{MaxNReg: 64, ReqNTID: [3]int{16, 16, 1}}
			if k.Tuning != want {
				t.Errorf("Tuning = %+v, want %+v", k.Tuning, want)
			}
		},
	}, {
		name: "BodilessDeclaration",
		ptx: `.version 8.0
.extern .func (.param .b32 r) helper (.param .b32 a);
.visible .entry k() {
	ret;
}`,
		check: func(t *testing.T, m *cudaptx.Module) {
			h := m.Kernel("helper")
			if h == nil {
				t.Fatal("helper not found")
			}
			if h.Body || h.Entry || h.Linkage != "extern" || len(h.Returns) != 1 || len(h.Params) != 1 {
				t.Errorf("helper = %+v, want bodiless extern func with 1 return and 1 param", h)
			}
			if len(h.Instructions) != 0 {
				t.Errorf("helper Instructions = %+v, want none", h.Instructions)
			}
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := cudaptx.ParseString(tt.ptx)
			if err != nil {
				t.Fatalf("ParseString: %v", err)
			}
			tt.check(t, m)
		})
	}
}

func TestPTX94Keywords(t *testing.T) {
	// 1. Check instruction keyword spcompress
	if !cudaptx.IsInstructionKeyword("spcompress") {
		t.Errorf("expected spcompress to be recognized as an instruction keyword")
	}

	// 2. Check data type .ue5m3x2
	for _, dt := range []string{".ue5m3x2", "ue5m3x2", ".ue4m3x2", ".e4m3", ".e5m2", ".f32"} {
		if !cudaptx.IsDataType(dt) {
			t.Errorf("expected %q to be recognized as a PTX data type", dt)
		}
	}

	// 3. Check parsing of .loc_intermediate and PTX 9.4 keywords
	ptxSrc := `.version 9.4
.target sm_100
.address_size 64
.file 1 "kernel.cu"
.visible .entry test_ptx94() {
	.loc_intermediate 1 42 8
	spcompress.b32 %r1, %r2, %r3;
	ret;
}`

	m, err := cudaptx.ParseString(ptxSrc)
	if err != nil {
		t.Fatalf("ParseString failed: %v", err)
	}

	k := m.Kernel("test_ptx94")
	if k == nil {
		t.Fatal("kernel test_ptx94 not found")
	}
	if len(k.Instructions) != 2 {
		t.Fatalf("expected 2 instructions, got %d", len(k.Instructions))
	}

	inst := k.Instructions[0]
	if inst.Mnemonic != "spcompress" {
		t.Errorf("inst Mnemonic = %q, want spcompress", inst.Mnemonic)
	}
	if !inst.Loc.Intermediate {
		t.Errorf("expected inst.Loc.Intermediate to be true")
	}
	if inst.Loc.Line != 42 || inst.Loc.Column != 8 {
		t.Errorf("inst.Loc = %v, want line 42 col 8", inst.Loc)
	}

	// 4. Check paired .loc and .loc_intermediate
	ptxDual := `.version 9.4
.target sm_100
.address_size 64
.file 1 "kernel.cu"
.file 2 "kernel.ir"
.visible .entry test_dual() {
	.loc 1 10 2
	.loc_intermediate 2 42 8
	spcompress.b32 %r1, %r2, %r3;
	ret;
}`
	mDual, err := cudaptx.ParseString(ptxDual)
	if err != nil {
		t.Fatalf("ParseString dual failed: %v", err)
	}
	kDual := mDual.Kernel("test_dual")
	instDual := kDual.Instructions[0]
	if instDual.Loc.File != "kernel.cu" || instDual.Loc.Line != 10 || instDual.Loc.Column != 2 {
		t.Errorf("instDual.Loc = %v, want kernel.cu:10:2", instDual.Loc)
	}
	if instDual.IntermediateLoc.File != "kernel.ir" || instDual.IntermediateLoc.Line != 42 || instDual.IntermediateLoc.Column != 8 {
		t.Errorf("instDual.IntermediateLoc = %v, want kernel.ir:42:8", instDual.IntermediateLoc)
	}
}
