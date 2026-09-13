package tileiras_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/tileiras"
)

const sampleRemarksYAML = `--- !Passed
Pass:            Memory
Name:            RemarkMemoryLoadInstructionSelected
DebugLoc:        { File: '/tmp/tile_matrix.cu', Line: 10, Column: 37 }
Function:        matrix
Args:
  - RemarkId:    4
  - Remark:      Load instruction selected
  - Instruction: TMA Load instruction
  - Shape:       '[16, 16]'
...
--- !Passed
Pass:            Memory
Name:            RemarkMemoryStoreInstructionSelected
DebugLoc:        { File: '/tmp/tile_matrix.cu', Line: 12, Column: 10 }
Function:        matrix
Args:
  - RemarkId:    2
  - Remark:      Store instruction selected
  - Instruction: TMA Store instruction
  - Shape:       '[16, 16]'
...
--- !Failure
Pass:            Memory
Name:            RemarkMemoryTmaLoadFailed
DebugLoc:        { File: '/tmp/tile_matrix.cu', Line: 14, Column: 5 }
Function:        matrix
Args:
  - RemarkId:    5
  - Remark:      Failed to select TMA load instruction
  - Reason:      non-contiguous stride along dim 1
  - Shape:       '[16, 32]'
...
--- !Failure
Pass:            Tensor-core
Name:            RemarkTensorCoreMMA
DebugLoc:        { File: '/tmp/tile_matrix.cu', Line: 16, Column: 9 }
Function:        matrix
Args:
  - RemarkId:    3
  - Remark:      MMA operation failed to optimize to use Tensor Cores, it is using FMA instructions instead
  - Instruction: FMA
  - Shape:       '[1, 1, 1]'
...
--- !Passed
Pass:            Tensor-core
Name:            RemarkTensorCoreMMA
DebugLoc:        { File: '/tmp/tile_gemm.cu', Line: 20, Column: 4 }
Function:        gemm
Args:
  - RemarkId:    6
  - Remark:      MMA operation successfully optimized to use Tensor Cores
  - Instruction: Tensor Core MMA
  - Shape:       '[128, 64, 16]'
...
`

func TestParseRemarksYAML(t *testing.T) {
	remarks, err := tileiras.ParseRemarksYAML([]byte(sampleRemarksYAML))
	if err != nil {
		t.Fatalf("ParseRemarksYAML failed: %v", err)
	}

	if len(remarks) != 5 {
		t.Fatalf("len(remarks) = %d, want 5", len(remarks))
	}

	// 1. TMA Load Passed
	r0 := remarks[0]
	if r0.Kind != "Passed" || r0.Pass != "Memory" {
		t.Errorf("r0 = %v, want Passed/Memory", r0)
	}
	if r0.File != "/tmp/tile_matrix.cu" || r0.Line != 10 || r0.Column != 37 {
		t.Errorf("r0 loc = %s:%d:%d, want /tmp/tile_matrix.cu:10:37", r0.File, r0.Line, r0.Column)
	}
	if r0.Instruction != "TMA Load instruction" {
		t.Errorf("r0 instruction = %q, want 'TMA Load instruction'", r0.Instruction)
	}

	// 2. TMA Load Failed
	r2 := remarks[2]
	if !r2.IsTMAFailure() {
		t.Errorf("expected r2 to be TMA failure, got name %q remark %q", r2.Name, r2.Remark)
	}
	if r2.Reason != "non-contiguous stride along dim 1" {
		t.Errorf("r2 reason = %q, want 'non-contiguous stride along dim 1'", r2.Reason)
	}

	// 3. Tensor Core Failure (FMA fallback)
	r3 := remarks[3]
	if !r3.IsTensorCoreFailure() {
		t.Errorf("expected r3 to be TensorCore failure, got %v", r3)
	}

	// 4. Tensor Core Success
	r4 := remarks[4]
	if !r4.IsTensorCoreMMA() {
		t.Errorf("expected r4 to be TensorCore success, got %v", r4)
	}

	res := &tileiras.Result{Remarks: remarks}
	tmaFails := res.TMAFailures()
	if len(tmaFails) != 1 || tmaFails[0].Name != "RemarkMemoryTmaLoadFailed" {
		t.Errorf("TMAFailures = %v, want 1 remark (RemarkMemoryTmaLoadFailed)", tmaFails)
	}

	tcFails := res.TensorCoreFailures()
	if len(tcFails) != 1 || tcFails[0].Name != "RemarkTensorCoreMMA" {
		t.Errorf("TensorCoreFailures = %v, want 1 remark", tcFails)
	}
}
