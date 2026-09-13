package tileirdisasm_test

import (
	"context"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileirdisasm"
)

const sampleModule = `// Producer: Cuda compilation tools, release 13.4, V13.4.59
entry @bf16_splat() {
  %cst_bf16 = constant <bf16: 5.500000e+00> : tile<bf16> loc(#loc)
  return loc(#loc)
} loc(#loc)
#loc = loc("kernel.cu":5:2)
`

const sampleMatrixModule = `// Producer: Cuda compilation tools, release 13.4
entry @matrix(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>, %arg2: tile<ptr<f32>>) {
  %cst_16_i32 = constant <i32: 16> : tile<i32> loc(#loc2)
  %0 = make_token : token loc(#loc3)
  %assume = assume div_by<16>, %arg0 : tile<ptr<f32>> loc(#loc4)
  %for:5 = for %loopIdx in (%cst_0_i32 to %cst_2_i32, step %cst_1_i32) : tile<i32> iter_values(%iterArg0 = %bcast) -> (tile<16x16xf32>, token) {
    %31 = join_tokens %iterArg2, %iterArg1 : token loc(#loc5)
    %sum = cuda_tile.mmaf_scaled %arg0, %arg1 : tile<16x16xf32> loc(#loc6)
  }
  return loc(#loc7)
}
#loc2 = loc("tile_matrix.cu":10:37)
#loc3 = loc("tile_matrix.cu":11:2)
#loc4 = loc("tile_matrix.cu":12:15)
#loc5 = loc("tile_matrix.cu":14:5)
#loc6 = loc("tile_matrix.cu":15:10)
#loc7 = loc("tile_matrix.cu":18:1)
`

func TestParseModule(t *testing.T) {
	mod, err := tileirdisasm.ParseString(sampleModule)
	if err != nil {
		t.Fatalf("ParseString failed: %v", err)
	}

	if !strings.Contains(mod.Producer, "13.4") {
		t.Errorf("Producer = %q, want version 13.4", mod.Producer)
	}

	fn := mod.FindFunction("bf16_splat")
	if fn == nil {
		t.Fatal("expected to find function @bf16_splat")
	}

	if !fn.Entry {
		t.Error("expected fn.Entry to be true")
	}

	if len(fn.Operations) != 2 {
		t.Fatalf("len(fn.Operations) = %d, want 2", len(fn.Operations))
	}

	op0 := fn.Operations[0]
	if len(op0.Results) != 1 || op0.Results[0] != "%cst_bf16" {
		t.Errorf("op0.Results = %v, want ['%%cst_bf16']", op0.Results)
	}
	if op0.OpCode != "constant" {
		t.Errorf("op0.OpCode = %q, want 'constant'", op0.OpCode)
	}
	if op0.Type != "tile<bf16>" {
		t.Errorf("op0.Type = %q, want 'tile<bf16>'", op0.Type)
	}
	if op0.Loc.File != "kernel.cu" || op0.Loc.Line != 5 || op0.Loc.Column != 2 {
		t.Errorf("op0.Loc = %v, want kernel.cu:5:2", op0.Loc)
	}

	op1 := fn.Operations[1]
	if op1.OpCode != "return" {
		t.Errorf("op1.OpCode = %q, want 'return'", op1.OpCode)
	}
}

func TestParseMatrixModule(t *testing.T) {
	mod, err := tileirdisasm.ParseString(sampleMatrixModule)
	if err != nil {
		t.Fatalf("ParseString failed: %v", err)
	}

	fn := mod.FindFunction("matrix")
	if fn == nil {
		t.Fatal("expected function @matrix")
	}

	if len(fn.Params) != 3 {
		t.Fatalf("len(fn.Params) = %d, want 3", len(fn.Params))
	}
	if fn.Params[0].Name != "%arg0" || fn.Params[0].Type != "tile<ptr<f32>>" {
		t.Errorf("param[0] = %+v, want %%arg0 : tile<ptr<f32>>", fn.Params[0])
	}

	var mmaOp *tileirdisasm.Operation
	for _, op := range fn.Operations {
		if op.OpCode == "cuda_tile.mmaf_scaled" {
			mmaOp = op
			break
		}
	}
	if mmaOp == nil {
		t.Fatal("missing cuda_tile.mmaf_scaled operation")
	}

	if len(mmaOp.Results) != 1 || mmaOp.Results[0] != "%sum" {
		t.Errorf("mmaOp.Results = %v, want ['%%sum']", mmaOp.Results)
	}
	if mmaOp.Loc.File != "tile_matrix.cu" || mmaOp.Loc.Line != 15 {
		t.Errorf("mmaOp.Loc = %v, want tile_matrix.cu:15", mmaOp.Loc)
	}

	// Verify OperationAtLine
	targetOp := fn.OperationAtLine(mmaOp.Line)
	if targetOp != mmaOp {
		t.Errorf("OperationAtLine(%d) = %v, want %v", mmaOp.Line, targetOp, mmaOp)
	}
}

func TestExtractFromELFInvalid(t *testing.T) {
	_, err := tileirdisasm.ExtractFromELF([]byte("not an elf"))
	if err == nil {
		t.Error("expected error for invalid ELF bytes")
	}
}

func TestDisassembleEmpty(t *testing.T) {
	ctx := context.Background()
	_, err := tileirdisasm.Disassemble(ctx, nil)
	if err == nil {
		t.Error("expected error on nil bytecode")
	}
}
