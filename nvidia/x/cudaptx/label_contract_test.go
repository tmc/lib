package cudaptx_test

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func TestPTXLabelPositions(t *testing.T) {
	const text = `.visible .entry k() {
start: alias: @%p bra.uni forward;
{
back: add.u32 %r1, %r1, 1;
@!%p bra start;
}
forward: bra back;
ret;
end:
}
.visible .func f() { start: ret; }
`
	m, err := cudaptx.ParseString(text)
	if err != nil {
		t.Fatal(err)
	}
	want := []cudaptx.Label{
		{Name: "start", Line: 2, Instruction: 0},
		{Name: "alias", Line: 2, Instruction: 0},
		{Name: "back", Line: 4, Instruction: 1},
		{Name: "forward", Line: 7, Instruction: 3},
		{Name: "end", Line: 9, Instruction: 5},
	}
	if got := m.Kernels[0].Labels; !reflect.DeepEqual(got, want) {
		t.Fatalf("labels = %+v, want %+v", got, want)
	}
	if got := m.Kernels[1].Labels; !reflect.DeepEqual(got, []cudaptx.Label{{Name: "start", Line: 11, Instruction: 0}}) {
		t.Fatalf("function labels = %+v", got)
	}
	if got := len(m.Kernels[0].Instructions); got != 5 {
		t.Fatalf("labels changed instruction count: got %d, want 5", got)
	}
	if err := branchesResolve(m); err != nil {
		t.Fatal(err)
	}
}

func TestPTXLabelContract(t *testing.T) {
	tests := []struct {
		name, body, wantError string
	}{
		{"forward", "bra next; next: ret;", ""},
		{"backward", "loop: add.u32 %r, %r, 1; bra loop;", ""},
		{"percent identifier", "%loop: bra %loop;", ""},
		{"predicated uniform", "@!%p bra.uni next; next: ret;", ""},
		{"same instruction aliases", "a: b: bra a;", ""},
		{"unused end label", "ret; end:", ""},
		{"comment between label and instruction", "next: /* : ignored */ bra next;", ""},
		{"missing", "bra missing; ret;", `unresolved branch target "missing"`},
		{"predicated missing", "@%p bra.uni missing; ret;", `unresolved branch target "missing"`},
		{"empty target", "bra; ret;", "requires one direct label target"},
		{"multiple targets", "bra a, b; a: ret; b: ret;", "requires one direct label target"},
		{"expression target", "bra a+1; a: ret;", "requires one direct label target"},
		{"register target", "bra %r; ret;", `unresolved branch target "%r"`},
		{"numeric target", "bra 1; ret;", "requires one direct label target"},
		{"end target", "bra end; end:", `branch target "end" has no instruction`},
		{"duplicate", "a: ret; a: ret;", `duplicate label "a"`},
		{"duplicate in nested block", "a: ret; { a: ret; }", `duplicate label "a"`},
		{"empty label", ": ret;", `invalid label ""`},
		{"numeric label", "1bad: ret;", `invalid label "1bad"`},
		{"dotted label", "bad.name: ret;", `invalid label "bad.name"`},
		{"punctuation label", "bad-name: ret;", `invalid label "bad-name"`},
		// These forms are deliberately outside direct bra/bra.uni resolution.
		{"indirect table not resolved", "brx.idx %r, targets; ret;", ""},
		{"other branch modifier not resolved", "bra.other unknown; ret;", ""},
		{"call target not resolved", "call external_function, (); ret;", ""},
		{"double colon operand preserved", "ld.shared::cta.u32 %r, [%a]; ret;", ""},
	}
	parsers := []struct {
		name  string
		parse func(string) (*cudaptx.Module, error)
	}{
		{"string", cudaptx.ParseString},
		{"bytes", func(s string) (*cudaptx.Module, error) { return cudaptx.ParseBytes([]byte(s)) }},
		{"reader", func(s string) (*cudaptx.Module, error) { return cudaptx.Parse(strings.NewReader(s)) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, p := range parsers {
				t.Run(p.name, func(t *testing.T) {
					m, err := p.parse(".visible .entry k() { " + tt.body + " }")
					if tt.wantError != "" {
						if err == nil || !strings.Contains(err.Error(), tt.wantError) {
							t.Fatalf("Parse error = %v, want %q", err, tt.wantError)
						}
						if m != nil {
							t.Fatalf("failed validation returned a partial module: %+v", m)
						}
						return
					}
					if err != nil {
						t.Fatal(err)
					}
					if err := branchesResolve(m); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}

func TestPTXLabelScope(t *testing.T) {
	for _, text := range []string{
		".visible .entry a() { bra target; }\n.visible .entry b() { target: ret; }",
		".extern .func target();\n.visible .entry a() { bra target; }",
	} {
		if _, err := cudaptx.ParseString(text); err == nil || !strings.Contains(err.Error(), `unresolved branch target "target"`) {
			t.Fatalf("cross-function or external target accepted: %v", err)
		}
	}
	// Reusing a label name in separate function bodies is valid.
	if _, err := cudaptx.ParseString(".visible .entry a() { target: bra target; }\n.visible .entry b() { target: bra target; }"); err != nil {
		t.Fatal(err)
	}
}

func TestPTXLabelFuzzInvariant(t *testing.T) {
	m, err := cudaptx.ParseString(".visible .entry k() { target: bra target; }")
	if err != nil {
		t.Fatal(err)
	}
	if err := branchesResolve(m); err != nil {
		t.Fatal(err)
	}
	m.Kernels[0].Instructions[0].Operands = "missing"
	// Check the registered harness, not only the helper: removing its actual
	// registration must make this control fail.
	rejected := false
	for _, invariant := range ptxDecoder.Invariants {
		rejected = rejected || invariant(m) != nil
	}
	if !rejected {
		t.Fatal("registered fuzz invariants accepted an unresolved direct branch")
	}
	m.Kernels[0].Instructions[0].Operands = "target"
	m.Kernels[0].Labels[0].Instruction = len(m.Kernels[0].Instructions)
	if branchesResolve(m) == nil {
		t.Fatal("branch invariant accepted a target past the last instruction")
	}
	m.Kernels[0].Labels[0].Instruction = -1
	if branchesResolve(m) == nil {
		t.Fatal("branch invariant accepted a negative instruction index")
	}
	m.Kernels[0].Labels[0].Instruction = 0
	m.Kernels[0].Labels = append(m.Kernels[0].Labels, m.Kernels[0].Labels[0])
	if branchesResolve(m) == nil {
		t.Fatal("branch invariant accepted duplicate labels")
	}
}

func TestPTXLabelsOutsideFunctions(t *testing.T) {
	// A module-scope token is not a function label, and Parse still accepts
	// unmodelled module text. Empty input still describes an empty module.
	for _, text := range []string{"", "outside:\n.visible .entry k() { ret; }"} {
		m, err := cudaptx.ParseString(text)
		if err != nil || m == nil {
			t.Fatalf("Parse(%q) = %v, %v", text, m, err)
		}
	}
}

func ExampleLabel() {
	m, err := cudaptx.ParseString(".visible .entry k() { loop: bra loop; }")
	if err != nil {
		fmt.Println(err)
		return
	}
	label := m.Kernels[0].Labels[0]
	fmt.Println(label.Name, label.Instruction)
	// Output: loop 0
}

func TestPTXLabelQuotedText(t *testing.T) {
	literals := []struct{ name, directive string }{
		{"review separator", `.pragma "prefix;phantom:";`},
		{"braces", `.pragma "{prefix;phantom:}";`},
		{"escaped quote", `.pragma "prefix\";phantom:";`},
		{"escaped backslash", `.pragma "prefix\\;phantom:";`},
		{"comment markers", `.pragma "// /* prefix;phantom: */";`},
		{"function words", `.pragma " .entry phantom() { phantom: ret; }";`},
		{"device function words", `.pragma " .func phantom() { phantom: ret; }";`},
		{"continued quote", ".pragma \"prefix\n// phantom: } ;\n.maxntid 256\n\";"},
	}
	parsers := []struct {
		name  string
		parse func(string) (*cudaptx.Module, error)
	}{
		{"string", cudaptx.ParseString},
		{"bytes", func(s string) (*cudaptx.Module, error) { return cudaptx.ParseBytes([]byte(s)) }},
		{"reader", func(s string) (*cudaptx.Module, error) { return cudaptx.Parse(strings.NewReader(s)) }},
	}
	for _, tt := range literals {
		t.Run(tt.name, func(t *testing.T) {
			for _, p := range parsers {
				t.Run(p.name, func(t *testing.T) {
					m, err := p.parse(".entry k() { " + tt.directive + " ret; }")
					if err != nil || len(m.Kernels) != 1 {
						t.Fatalf("quoted directive changed function framing: %v, %v", m, err)
					}
					k := m.Kernels[0]
					if len(k.Labels) != 0 || len(k.Instructions) != 1 || k.Instructions[0].Opcode != "ret" || k.EndLine == 0 {
						t.Fatalf("literal contents changed executable framing: %+v", k)
					}
					m, err = p.parse(".entry k() { " + tt.directive + " bra phantom; ret; }")
					if err == nil || m != nil || !strings.Contains(err.Error(), `unresolved branch target "phantom"`) {
						t.Fatalf("quoted text resolved a missing target: %v, %v", m, err)
					}
				})
			}
		})
	}
}

func TestPTXLabelInlineBlocks(t *testing.T) {
	tests := []struct {
		name, body   string
		labels       []cudaptx.Label
		instructions int
	}{
		{
			name: "review forward backward aliases",
			body: "first:other:@!%p bra.uni last;{middle:bra first;}last:bra middle;ret;unused:",
			labels: []cudaptx.Label{
				{Name: "first", Line: 1, Instruction: 0},
				{Name: "other", Line: 1, Instruction: 0},
				{Name: "middle", Line: 1, Instruction: 1},
				{Name: "last", Line: 1, Instruction: 2},
				{Name: "unused", Line: 1, Instruction: 4},
			},
			instructions: 4,
		},
		{
			name: "adjacent blocks",
			body: "first:bra last;{middle:bra first;}{other:bra middle;}last:bra other;",
			labels: []cudaptx.Label{
				{Name: "first", Line: 1, Instruction: 0},
				{Name: "middle", Line: 1, Instruction: 1},
				{Name: "other", Line: 1, Instruction: 2},
				{Name: "last", Line: 1, Instruction: 3},
			},
			instructions: 4,
		},
		{
			name:         "vector then block",
			body:         "ld.global.v2.f32 {%f0, %f1}, [%rd]; {target:bra target;} ret;",
			labels:       []cudaptx.Label{{Name: "target", Line: 1, Instruction: 1}},
			instructions: 3,
		},
		{
			name:         "continued vector then block",
			body:         "ld.global.v2.f32 {\n%f0,\n%f1\n}, [%rd];\n{target:bra target;} ret;",
			labels:       []cudaptx.Label{{Name: "target", Line: 5, Instruction: 1}},
			instructions: 3,
		},
		{
			name:         "operand text cannot declare a label",
			body:         "future.op {1;phantom:}, %r; real:bra real;",
			labels:       []cudaptx.Label{{Name: "real", Line: 1, Instruction: 1}},
			instructions: 2,
		},
		{
			name:         "space before colon",
			body:         "first : bra first;",
			labels:       []cudaptx.Label{{Name: "first", Line: 1, Instruction: 0}},
			instructions: 1,
		},
		{
			name:         "continued label",
			body:         "first\n: bra first;",
			labels:       []cudaptx.Label{{Name: "first", Line: 1, Instruction: 0}},
			instructions: 1,
		},
		{
			name:         "comment before colon",
			body:         "first/* comment */ : bra first;",
			labels:       []cudaptx.Label{{Name: "first", Line: 1, Instruction: 0}},
			instructions: 1,
		},
		{
			name:         "declaration initializer",
			body:         ".param .b8 data[2] = {1, 2}; real:bra real;",
			labels:       []cudaptx.Label{{Name: "real", Line: 1, Instruction: 0}},
			instructions: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text := ".entry k() { " + tt.body + " }"
			for _, parse := range []func(string) (*cudaptx.Module, error){
				cudaptx.ParseString,
				func(s string) (*cudaptx.Module, error) { return cudaptx.ParseBytes([]byte(s)) },
				func(s string) (*cudaptx.Module, error) { return cudaptx.Parse(strings.NewReader(s)) },
			} {
				m, err := parse(text)
				if err != nil || len(m.Kernels) != 1 {
					t.Fatalf("Parse = %v, %v", m, err)
				}
				k := m.Kernels[0]
				if !reflect.DeepEqual(k.Labels, tt.labels) || len(k.Instructions) != tt.instructions || k.EndLine == 0 {
					t.Fatalf("framing = labels%+v instructions%+v end%d, want labels%+v instructions%d", k.Labels, k.Instructions, k.EndLine, tt.labels, tt.instructions)
				}
				if err := branchesResolve(m); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
	// A directive-looking line inside a block comment cannot start an opaque
	// source section and swallow the next real function or its labels.
	m, err := cudaptx.ParseString("/*\n.nv_intermediate_source_section\n*/\n.entry k() { real:bra real; }")
	if err != nil || len(m.Kernels) != 1 || len(m.Kernels[0].Labels) != 1 || m.Kernels[0].Labels[0].Name != "real" {
		t.Fatalf("comment changed function framing: %v, %v", m, err)
	}
}
