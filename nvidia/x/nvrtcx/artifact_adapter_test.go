package nvrtcx_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestCompiledModuleToArtifact(t *testing.T) {
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "y", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "x", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "a", Type: "f32"},
	)
	if err != nil {
		t.Fatalf("NewSchema: %v", err)
	}

	// 1. PTX artifact adapter
	ptxModule := &nvrtcx.CompiledModule{}
	ptxModule.SetJITOptions(cudakernel.JITOptions{MaxRegisters: 32, OptLevel: 4})

	// Inject PTX output using export or simulate compiled module
	// By testing ToArtifact with output kinds
	testCases := []struct {
		name         string
		output       nvrtcx.OutputKind
		expectedKind cudakernel.Kind
		requiresLink bool
	}{
		{"PTX", nvrtcx.OutputPTX, cudakernel.KindPTX, false},
		{"LTOIR", nvrtcx.OutputLTOIR, cudakernel.KindLTOIR, true},
		{"TileIR", nvrtcx.OutputTileIR, cudakernel.KindTileIR, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cm := nvrtcx.ExportNewCompiledModule(tc.output, "dummy-data", []byte("raw-bytes-here"))
			cm.SetJITOptions(cudakernel.JITOptions{MaxRegisters: 64, OptLevel: 3})

			input := schema
			input.Params = append([]cudakernel.Param(nil), schema.Params...)
			art, err := cm.ToArtifact("test_kernel", input, "sm_121a")
			if err != nil {
				t.Fatalf("ToArtifact failed: %v", err)
			}
			if art.Kind() != tc.expectedKind {
				t.Errorf("art.Kind = %v, want %v", art.Kind(), tc.expectedKind)
			}
			if art.RequiresLink() != tc.requiresLink {
				t.Errorf("art.RequiresLink() = %v, want %v", art.RequiresLink(), tc.requiresLink)
			}
			if art.JITOptions().MaxRegisters != 64 || art.JITOptions().OptLevel != 3 {
				t.Errorf("art.JITOptions() mismatch: %+v", art.JITOptions())
			}
			cm.SetJITOptions(cudakernel.JITOptions{MaxRegisters: 16, OptLevel: 1})
			input.Params[0].Name = "changed caller schema"
			view := art.Schema()
			view.Params[0].Name = "changed view"
			if art.JITOptions().MaxRegisters != 64 || art.Schema().Params[0].Name != "y" {
				t.Fatal("adapter artifact changed with module or schema view")
			}
		})
	}
}
