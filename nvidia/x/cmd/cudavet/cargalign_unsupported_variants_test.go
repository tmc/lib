package main

import (
	"os"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
)

func TestUnsupportedVariantsReported(t *testing.T) {
	defer func(v bool) { reportUnsupported = v }(reportUnsupported)
	reportUnsupported = true
	analysistest.Run(t, analysistest.TestData(), cargalignAnalyzer, "cargalignunsupportedvariants", "cargalignsupportedempty")
}

func TestUnsupportedVariantsSilentWithoutFlag(t *testing.T) {
	saved := reportUnsupported
	reportUnsupported = false
	defer func() { reportUnsupported = saved }()
	body, err := os.ReadFile("testdata/src/cargalignunsupportedvariants/variants.go")
	if err != nil {
		t.Fatal(err)
	}
	pass, diagnostics := cargalignPass(t, cargalignTypes(t), string(body))
	if _, err := runCargalign(pass); err != nil {
		t.Fatal(err)
	}
	if len(*diagnostics) != 0 {
		t.Fatalf("unsupported flag off: %+v", *diagnostics)
	}
}

func TestRawSchemaImplicitTotal(t *testing.T) {
	source := `package fixture
 import "github.com/tmc/lib/nvidia/x/cudakernel"
 var _ = cudakernel.Schema{Params: []cudakernel.Param{{Name:"n", Type:"u32"}}}
 `
	pass, diagnostics := cargalignPass(t, cargalignTypes(t), source)
	if _, err := runCargalign(pass); err != nil {
		t.Fatal(err)
	}
	if len(*diagnostics) != 1 || (*diagnostics)[0].Message != "layout declares 0 total parameter bytes; cudakernel.Schema computes 8" {
		t.Fatalf("implicit total: %+v", *diagnostics)
	}
}

func TestCargalignAliases(t *testing.T) {
	saved := reportUnsupported
	reportUnsupported = true
	defer func() { reportUnsupported = saved }()
	cases := []struct{ name, body, want string }{
		{"schema", `type S = cudakernel.Schema; var _ = S{Params:[]cudakernel.Param{{Name:"x",Type:"u32"}},TotalParamBytes:4}`, "layout declares 4 total parameter bytes; cudakernel.Schema computes 8"},
		{"parameter", `type P = cudakernel.Param; var _,_ = cudakernel.NewSchema(P{Name:"x",Type:"u32",Offset:4})`, "parameter x declares offset 4; cudakernel.Schema places it at offset 0"},
		{"marked-named", `type Small struct{N uint32}
//cuda:kernelparams
type A = Small`, "kernel parameter block A is 4 bytes in Go; cudakernel.Schema computes 8 total parameter bytes"},
		{"marked-anonymous", `//cuda:kernelparams
type A = struct{N uint32}`, "kernel parameter block A is 4 bytes in Go; cudakernel.Schema computes 8 total parameter bytes"},
		{"schema-clean", `type S = cudakernel.Schema; var _ = S{Params:[]cudakernel.Param{{Name:"x",Type:"u32"}},TotalParamBytes:8}`, ""},
		{"parameter-clean", `type P = cudakernel.Param; var _,_ = cudakernel.NewSchema(P{Name:"x",Type:"u32",Offset:0})`, ""},
		{"marked-named-clean", `type Small struct{N uint64}
//cuda:kernelparams
type A = Small`, ""},
		{"marked-anonymous-clean", `//cuda:kernelparams
type A = struct{N uint64}`, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pass, diagnostics := cargalignPass(t, cargalignTypes(t), "package fixture\nimport \"github.com/tmc/lib/nvidia/x/cudakernel\"\nvar _ cudakernel.Schema\n"+tc.body)
			if _, err := runCargalign(pass); err != nil {
				t.Fatal(err)
			}
			if tc.want == "" {
				if len(*diagnostics) != 0 {
					t.Fatalf("clean alias diagnostics=%+v", *diagnostics)
				}
				return
			}
			if len(*diagnostics) != 1 || (*diagnostics)[0].Message != tc.want {
				t.Fatalf("alias diagnostics=%+v; want %q", *diagnostics, tc.want)
			}
		})
	}
}

func TestCargalignDirectFunctions(t *testing.T) {
	cases := []struct{ name, source, want string }{
		{"parenthesized", `package fixture
 import "github.com/tmc/lib/nvidia/x/cudakernel"
 var _,_ = (cudakernel.NewSchema)(cudakernel.Param{Name:"x",Type:"u32",Offset:4})`, "parameter x declares offset 4; cudakernel.Schema places it at offset 0"},
		{"dot-import", `package fixture
 import . "github.com/tmc/lib/nvidia/x/cudakernel"
 var _,_ = NewSchema(Param{Name:"x",Type:"u32",Offset:4})`, "parameter x declares offset 4; cudakernel.Schema places it at offset 0"},
		{"parenthesized-clean", `package fixture
 import "github.com/tmc/lib/nvidia/x/cudakernel"
 var _,_ = (cudakernel.NewSchema)(cudakernel.Param{Name:"x",Type:"u32"})`, ""},
		{"dot-import-clean", `package fixture
 import . "github.com/tmc/lib/nvidia/x/cudakernel"
 var _,_ = NewSchema(Param{Name:"x",Type:"u32"})`, ""},
		{"local-name", `package fixture
 import "github.com/tmc/lib/nvidia/x/cudakernel"
 func NewSchema(p cudakernel.Param){}
 func f(){NewSchema(cudakernel.Param{Name:"x",Type:"u32",Offset:4})}`, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pass, diagnostics := cargalignPass(t, cargalignTypes(t), tc.source)
			if _, err := runCargalign(pass); err != nil {
				t.Fatal(err)
			}
			if tc.want == "" {
				if len(*diagnostics) != 0 {
					t.Fatalf("clean direct call diagnostics=%+v", *diagnostics)
				}
				return
			}
			if len(*diagnostics) != 1 || (*diagnostics)[0].Message != tc.want {
				t.Fatalf("direct call diagnostics=%+v; want %q", *diagnostics, tc.want)
			}
		})
	}
}
