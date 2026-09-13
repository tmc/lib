package tileir_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/tileir"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

func TestAttachRemarks(t *testing.T) {
	const tileIR = `
cuda_tile.module @matmul_mod {
  cuda_tile.entry @matmul_kernel(%A: !cuda_tile.tile<ptr<f16>>, %B: !cuda_tile.tile<ptr<f16>>, %C: !cuda_tile.tile<ptr<f32>>) loc(#loc1) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token loc(#loc2)
    %vA = cuda_tile.load_view_tko %A, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16> loc(#loc3)
    %cst = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32> loc(#loc4)
    %acc = cuda_tile.mma_f %vA, %vA, %cst : (!cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf32>) -> !cuda_tile.tile<64x64xf32> loc(#loc5)
    cuda_tile.return loc(#loc6)
  }
}

#loc1 = loc("kernel.cu":10:1)
#loc2 = loc("kernel.cu":12:5)
#loc3 = loc("kernel.cu":14:10)
#loc4 = loc("kernel.cu":16:5)
#loc5 = loc("kernel.cu":20:10)
#loc6 = loc("kernel.cu":22:1)
`

	const yamlRemarks = `--- !Passed
Pass: Tensor-core
Name: RemarkTensorCoreMMA
DebugLoc: { File: 'kernel.cu', Line: 20, Column: 10 }
Function: matmul_kernel
Args:
  - Remark: 'MMA operation successfully optimized to use Tensor Cores'
  - Instruction: 'wgmma.mma_async'
--- !Passed
Pass: Memory
Name: RemarkMemoryLoadInstructionSelected
DebugLoc: { File: 'kernel.cu', Line: 14, Column: 10 }
Function: matmul_kernel
Args:
  - Remark: 'TMA Load instruction selected'
  - Instruction: 'TMA Load'
`

	mod, err := tileir.ParseString(tileIR)
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}

	remarks, err := tileiras.ParseRemarksYAML([]byte(yamlRemarks))
	if err != nil {
		t.Fatalf("ParseRemarksYAML: %v", err)
	}

	if len(remarks) != 2 {
		t.Fatalf("got %d remarks, want 2", len(remarks))
	}

	mod.AttachRemarks(remarks)

	fn := mod.FindFunction("matmul_kernel")
	if fn == nil {
		t.Fatal("matmul_kernel not found")
	}

	if len(fn.Remarks) != 2 {
		t.Errorf("fn.Remarks has %d remarks, want 2", len(fn.Remarks))
	}

	// Verify MMA operation got the Tensor Core remark
	var mmaOp *tileir.Operation
	for _, op := range fn.Operations {
		if op.IsMMA() {
			mmaOp = op
			break
		}
	}
	if mmaOp == nil {
		t.Fatal("MMA operation not found")
	}
	if len(mmaOp.Remarks) != 1 {
		t.Fatalf("MMA op has %d remarks, want 1", len(mmaOp.Remarks))
	}
	if !mmaOp.Remarks[0].IsTensorCoreMMA() {
		t.Errorf("expected remark to report IsTensorCoreMMA() = true")
	}
	if mmaOp.Remarks[0].Instruction != "wgmma.mma_async" {
		t.Errorf("Instruction = %q, want wgmma.mma_async", mmaOp.Remarks[0].Instruction)
	}

	// Verify TMA load operation got the Memory remark
	var loadOp *tileir.Operation
	for _, op := range fn.Operations {
		if op.IsTMA() {
			loadOp = op
			break
		}
	}
	if loadOp == nil {
		t.Fatal("TMA operation not found")
	}
	if len(loadOp.Remarks) != 1 {
		t.Fatalf("TMA load op has %d remarks, want 1", len(loadOp.Remarks))
	}
	if loadOp.Remarks[0].Pass != "Memory" {
		t.Errorf("Pass = %q, want Memory", loadOp.Remarks[0].Pass)
	}
}

func TestAttachRemarksNil(t *testing.T) {
	var mod *tileir.Module
	// Should not panic on nil module
	mod.AttachRemarks([]tileiras.Remark{{Line: 10}})

	// Should not panic on empty remarks
	mod2 := &tileir.Module{}
	mod2.AttachRemarks(nil)
}
