package main

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestRawSchemaImplicitOffsets(t *testing.T) {
	params := []cudakernel.Param{
		{Name: "n", Type: "u32", ByteWidth: 4, Alignment: 4},
		{Name: "implicit", Type: "ptr", ByteWidth: 8, Alignment: 8, IsPointer: true},
	}
	written := cudakernel.Schema{Params: params, TotalParamBytes: 16}
	if err := written.Validate(); err == nil || !strings.Contains(err.Error(), `"implicit" offset 0 overlaps prior end 4`) {
		t.Fatalf("raw Schema.Validate() = %v, want overlapping implicit zero", err)
	}
	computed, err := cudakernel.NewSchema(params...)
	if err != nil {
		t.Fatal(err)
	}
	want := fmt.Sprintf("parameter implicit declares offset 0; cudakernel.Schema places it at offset %d", computed.Params[1].Offset)
	found := false
	for _, result := range analysistest.Run(t, analysistest.TestData(), cargalignAnalyzer, "cargalign", "cargalignclean") {
		for _, diagnostic := range result.Diagnostics {
			if diagnostic.Message == want {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("analyzer did not report %q", want)
	}
}
