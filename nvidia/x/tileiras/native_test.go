package tileiras_test

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/tileir"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

// The native gate is opt-in. Unset, it skips; set, an unavailable assembler
// is a failure rather than a skip, so a run that was meant to qualify the
// toolchain cannot quietly report success without having used it.
const nativeEnv = "TILEIRAS_NATIVE"

// nativeModule is a minimal Tile IR kernel: one token, one tensor-view load,
// one store. It uses only operations x/tileir claims to model.
const nativeModule = `
cuda_tile.module @native_probe {
  cuda_tile.entry @copy_tile(%A: !cuda_tile.tile<ptr<f16>>, %C: !cuda_tile.tile<ptr<f16>>) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %vA = cuda_tile.load_view_tko %A, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16>
    %tok1 = cuda_tile.store_view_tko %vA, %C, %tok0 : (!cuda_tile.tile<64x64xf16>, !cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.token
    cuda_tile.return
  }
}
`

// TestAssembleNative is the gate this package did not have: it runs the real
// assembler over bytecode this repo produced and requires the assembler to
// accept it.
//
// It replaces a test that could not fail. That one resolved the tool with
// exec.LookPath, which misses the assembler in every CUDA installation
// directory that [tileiras.Assemble] searches -- so it skipped even on a host
// where the tool is present -- and it discarded both return values, asserting
// nothing on the runs where it did not skip.
//
// Availability here is decided by calling Assemble and reading ErrNotFound,
// which is the same resolution Assemble performs. A gate that decides
// availability by its own separate means can disagree with the code it
// guards, which is how the previous one came to skip on a host that had the
// tool.
func TestAssembleNative(t *testing.T) {
	enabled := os.Getenv(nativeEnv) != ""
	if !enabled {
		t.Skipf("%s unset; set it to run the real assembler", nativeEnv)
	}

	mod, err := tileir.ParseString(nativeModule)
	if err != nil {
		t.Fatalf("parse Tile IR: %v", err)
	}
	bc, err := mod.EncodeBytecode()
	if err != nil {
		t.Fatalf("encode bytecode: %v", err)
	}
	if !tileiras.IsBytecode(bc) {
		t.Fatal("encoder produced output without the Tile IR magic header")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	res, err := tileiras.Assemble(ctx, bc, tileiras.WithGPUName("sm_90"), tileiras.WithRemarks("all"))
	if errors.Is(err, tileiras.ErrNotFound) {
		t.Fatalf("%s is set but no assembler is available: %v", nativeEnv, err)
	}
	if err != nil {
		t.Fatalf("assembling encoder output: %v", err)
	}
	if len(res.CUBIN) == 0 {
		t.Error("assembly reported success with an empty CUBIN")
	}
	t.Logf("assembled %d bytes, %d remarks, %d TMA failures",
		len(res.CUBIN), len(res.Remarks), len(res.TMAFailures()))
}
