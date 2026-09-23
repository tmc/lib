package main

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func TestInjectTileIRLocations(t *testing.T) {
	const sampleTileIR = `// Sample Tile IR
cuda_tile.module @deepseek_v4 {
  cuda_tile.entry @ds4_q8_0_matvec(...) {
    cuda_tile.return
  }
}
`
	const samplePTX = `.version 8.0
.target sm_90
.address_size 64

.file 1 "kernels.cu"

.visible .entry ds4_q8_0_matvec(
	.param .u64 W
)
{
	.reg .b32 %r<4>;
	ld.param.u64 %rd1, [W];
	ret;
}
`

	injected := injectTileIRLocations(samplePTX, sampleTileIR)
	if !strings.Contains(injected, `.file	2 "kernels.tileir"`) {
		t.Errorf("missing .file 2 directive in injected PTX:\n%s", injected)
	}
	if !strings.Contains(injected, ".loc_intermediate 2 3 1") {
		t.Errorf("missing .loc_intermediate 2 3 1 directive in injected PTX:\n%s", injected)
	}

	// Verify cudaptx.ParseString parses it and recognizes .loc_intermediate
	parsed, err := cudaptx.ParseString(injected)
	if err != nil {
		t.Fatalf("cudaptx.ParseString failed on injected PTX: %v", err)
	}
	k := parsed.Kernel("ds4_q8_0_matvec")
	if k == nil {
		t.Fatal("kernel ds4_q8_0_matvec not found in parsed PTX")
	}
	if len(k.Instructions) == 0 {
		t.Fatal("no instructions in parsed kernel")
	}

	foundIntermediate := false
	for _, inst := range k.Instructions {
		if inst.IntermediateLoc.Line == 3 && inst.IntermediateLoc.FileIndex == 2 && inst.IntermediateLoc.Intermediate {
			foundIntermediate = true
			break
		}
	}
	if !foundIntermediate {
		t.Errorf("expected instruction to have IntermediateLoc Line=3 FileIndex=2 Intermediate=true, got: %+v", k.Instructions[0].IntermediateLoc)
	}
}
