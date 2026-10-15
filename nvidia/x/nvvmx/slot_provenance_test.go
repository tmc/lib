package nvvmx

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/nvvm"
)

const transportIR = `@s = internal addrspace(3) global i32 undef, align 4
define void @kernel() {
 %slot = alloca ptr, align 8
 store ptr addrspacecast (ptr addrspace(3) @s to ptr), ptr %slot, align 8
 %p = load ptr, ptr %slot, align 8
 %v = load i32, ptr %p, align 4
 ret void
}
`

func TestSlotProvenanceCompileTransport(t *testing.T) {
	if nvvm.Loaded() {
		t.Fatal("NVVM loaded")
	}
	t.Cleanup(func() {
		if nvvm.Loaded() {
			t.Error("loaded NVVM")
		}
	})
	for _, failure := range []string{"", "compile", "close", "log"} {
		name := failure
		if name == "" {
			name = "success"
		}
		t.Run(name, func(t *testing.T) {
			p := &fakeProgram{fail: failure, output: []byte("authored PTX transport bytes")}
			input := []Input{{Name: "owned.ll", Data: []byte(transportIR)}}
			result, err := compileWithOptions(input, func() (program, error) { return p, nil }, WithAddressSpaceInference())
			if (err != nil) != (failure != "") {
				t.Fatal(err)
			}
			if result == nil {
				t.Fatal("lost Result owner")
			}
			if failure != "" {
				if !errors.Is(err, errInjected) {
					t.Fatal("lost primary error", err)
				}
				if _, err := result.AddressSpaceProvenance(); err == nil {
					t.Fatal("failed compile published provenance")
				}
				if failure == "close" {
					if result.resources.h == nil || result.resources.pins == nil {
						t.Fatal("cleanup owner lost")
					}
					p.fail = ""
					if err := result.Close(); err != nil {
						t.Fatal(err)
					}
				}
				return
			}
			proof, err := result.AddressSpaceProvenance()
			if err != nil {
				t.Fatal(err)
			}
			if len(proof.Inputs) != 1 || proof.Inputs[0].Name != "owned.ll" || proof.PTXSHA256 != fmt.Sprintf("%x", sha256.Sum256(result.PTX)) {
				t.Fatal(proof)
			}
			prepared := proof.Inputs[0].Provenance
			if prepared.OriginalSHA256 != fmt.Sprintf("%x", sha256.Sum256(input[0].Data)) || prepared.PreparedSHA256 != fmt.Sprintf("%x", sha256.Sum256(p.inputs[0])) || prepared.PreparedSHA256 == prepared.OriginalSHA256 {
				t.Fatal("input identities not observed", prepared)
			}
			if !bytes.Contains(p.inputs[0], []byte("alloca ptr addrspace(3)")) {
				t.Fatal("compile bypassed actual inference")
			}
			proof.Inputs[0].Provenance.Slots[0].Slot = "forged"
			again, err := result.AddressSpaceProvenance()
			if err != nil || again.Inputs[0].Provenance.Slots[0].Slot == "forged" {
				t.Fatal("getter leaked ownership")
			}
			result.PTX[0] ^= 1
			if _, err := result.AddressSpaceProvenance(); err == nil {
				t.Fatal("modified output accepted")
			}
		})
	}
}
func TestSlotProvenanceDefaultOffAndLTO(t *testing.T) {
	for _, opts := range [][]Option{nil, {WithAddressSpaceInference(), WithGenLTO()}} {
		p := &fakeProgram{output: []byte("owned bytes")}
		r, err := compileWithOptions([]Input{{Name: "input", Data: []byte(transportIR)}}, func() (program, error) { return p, nil }, opts...)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := r.AddressSpaceProvenance(); err == nil {
			t.Fatal("unset inference or LTO is PTX evidence")
		}
		if opts == nil && !bytes.Equal(p.inputs[0], []byte(transportIR)) {
			t.Fatal("default changed input")
		}
	}
}
func TestSlotProvenanceOwnedOrigins(t *testing.T) {
	_, r, err := InferAddressSpaces([]byte(transportIR))
	if err != nil {
		t.Fatal(err)
	}
	p, err := r.Provenance()
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Slots) != 1 || p.Slots[0].DeclarationLine != 3 || len(p.Slots[0].Uses) != 1 || p.Slots[0].Uses[0].IRLine != 6 {
		t.Fatal(p)
	}
	p.Slots[0].Uses[0].Value = "forged"
	again, err := r.Provenance()
	if err != nil || again.Slots[0].Uses[0].Value != "%p" {
		t.Fatal("origin ownership lost")
	}
	r.Slots["@kernel/%slot"] = SlotInfo{}
	if _, err := r.Provenance(); err == nil {
		t.Fatal("mutated public map accepted")
	}
}
func TestSlotProvenanceMetadataBudget(t *testing.T) {
	ir := transportIR + strings.Repeat("!1 = !DILocation(line: 1, column: 1, scope: !1)\n", 5000)
	if _, r, err := InferAddressSpaces([]byte(ir)); err != nil || r.Slots["@kernel/%slot"].Effective != AddrSpaceShared {
		t.Fatal("irrelevant metadata changed inference", err)
	}
}

func ExampleInferenceReport_Provenance() {
	_, report, _ := InferAddressSpaces([]byte(transportIR))
	provenance, _ := report.Provenance()
	fmt.Println(provenance.Slots[0].Slot, provenance.Slots[0].Uses[0].Value)
	// Output: @kernel/%slot %p
}
func ExampleResult_AddressSpaceProvenance() {
	var result Result
	_, err := result.AddressSpaceProvenance()
	fmt.Println(err != nil)
	// Output: true
}
func ExampleSourceLocation() {
	var location SourceLocation
	fmt.Println(location.Line, location.Column)
	// Output: 0 0
}
func ExampleSlotUse() {
	use := SlotUse{IRLine: 6, Value: "%p", Operation: "load"}
	fmt.Println(use.Value, use.Operation)
	// Output: %p load
}
func ExampleSlotOrigin() {
	origin := SlotOrigin{Slot: "@kernel/%slot", Reason: "no terminal memory use"}
	fmt.Println(origin.Reason)
	// Output: no terminal memory use
}
func ExampleSlotProvenance() {
	_, report, _ := InferAddressSpaces([]byte(transportIR))
	provenance, _ := report.Provenance()
	fmt.Println(provenance.OriginalSHA256 != provenance.PreparedSHA256)
	// Output: true
}
func ExampleInferredInput() {
	input := InferredInput{Name: "kernel.ll"}
	fmt.Println(input.Name, len(input.Provenance.Slots))
	// Output: kernel.ll 0
}
func ExampleCompilationProvenance() {
	var provenance CompilationProvenance
	fmt.Println(len(provenance.Inputs))
	// Output: 0
}

func TestSlotProvenanceCompileRoute(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("source location unavailable")
	}
	f, err := parser.ParseFile(token.NewFileSet(), filepath.Join(filepath.Dir(file), "options.go"), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "CompileWithOptions" {
			continue
		}
		if len(fn.Body.List) != 1 {
			t.Fatal("public compile bypassed per-call transport")
		}
		ret, ok := fn.Body.List[0].(*ast.ReturnStmt)
		if !ok || len(ret.Results) != 1 {
			t.Fatal("compile route lost returned result/error")
		}
		call, ok := ret.Results[0].(*ast.CallExpr)
		if !ok || len(call.Args) != 3 || call.Ellipsis == token.NoPos {
			t.Fatal("compile route changed")
		}
		target, ok := call.Fun.(*ast.Ident)
		if !ok || target.Name != "compileWithOptions" {
			t.Fatal("public compile bypassed inference transport")
		}
		for i, want := range []string{"inputs", "newProgram", "opts"} {
			id, ok := call.Args[i].(*ast.Ident)
			if !ok || id.Name != want {
				t.Fatal("public compile changed actual inputs/factory/options")
			}
		}
		return
	}
	t.Fatal("missing public CompileWithOptions")
}

func TestSlotProvenanceDependentSlotsUnavailable(t *testing.T) {
	ir := strings.Replace(transportIR, " %v =", " %other = alloca ptr, align 8\n store ptr %p, ptr %other, align 8\n %q = load ptr, ptr %other, align 8\n %w = load i32, ptr %q, align 4\n %v =", 1)
	_, r, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatal(err)
	}
	proof, err := r.Provenance()
	if err != nil {
		t.Fatal(err)
	}
	if len(proof.Slots) != 2 {
		t.Fatal("dependent slots not actually inferred", proof)
	}
	for _, slot := range proof.Slots {
		if slot.Reason != "dependent slot" {
			t.Fatal("dependent origin became unique", slot)
		}
	}
}
