package tileir_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileir"
)

func TestVerifyValidKernel(t *testing.T) {
	const validCode = `
cuda_tile.module @valid_module {
  cuda_tile.entry @gemm(%A: !cuda_tile.tile<ptr<f16>>, %B: !cuda_tile.tile<ptr<f16>>, %C: !cuda_tile.tile<ptr<f32>>) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %vA = cuda_tile.load_view_tko %A, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x32xf16>
    %vB = cuda_tile.load_view_tko %B, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<32x64xf16>
    %cst = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>
    %acc = cuda_tile.mma_f %vA, %vB, %cst : (!cuda_tile.tile<64x32xf16>, !cuda_tile.tile<32x64xf16>, !cuda_tile.tile<64x64xf32>) -> !cuda_tile.tile<64x64xf32>
    cuda_tile.return
  }
}
`
	mod, err := tileir.ParseString(validCode)
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}

	fn := mod.FindFunction("gemm")
	if fn == nil {
		t.Fatal("gemm not found")
	}

	hazards, shapes := fn.Verify()
	if len(hazards) != 0 {
		t.Errorf("got %d unexpected hazards: %v", len(hazards), hazards)
	}
	if len(shapes) != 0 {
		t.Errorf("got %d unexpected shape errors: %v", len(shapes), shapes)
	}
}

func TestVerifyTokenHazards(t *testing.T) {
	tests := []struct {
		name        string
		code        string
		wantHazard  tileir.HazardKind
		wantMessage string
	}{
		{
			name: "unconsumed token",
			code: `cuda_tile.module @m {
  cuda_tile.entry @unconsumed() {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %tok_dead = cuda_tile.make_token : !cuda_tile.token
    %vA = cuda_tile.load_view_tko %arg0, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<16x16xf16>
    cuda_tile.return
  }
}`,
			wantHazard:  tileir.HazardUnconsumedToken,
			wantMessage: "never consumed or joined",
		},
		{
			name: "unsynchronized read after write RAW",
			code: `cuda_tile.module @m {
  cuda_tile.entry @raw_hazard(%view: !cuda_tile.tile<ptr<f32>>) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %tok1 = cuda_tile.make_token : !cuda_tile.token
    %val = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<16x16xf32>
    %out = cuda_tile.store_view_tko %val, %view, %tok0 : (!cuda_tile.tile<16x16xf32>, !cuda_tile.tile<ptr<f32>>, !cuda_tile.token) -> !cuda_tile.token
    %read = cuda_tile.load_view_tko %view, %tok1 : (!cuda_tile.tile<ptr<f32>>, !cuda_tile.token) -> !cuda_tile.tile<16x16xf32>
    cuda_tile.return
  }
}`,
			wantHazard:  tileir.HazardUnsynchronizedRead,
			wantMessage: "without synchronizing on prior write",
		},
		{
			name: "dangling async transfer at return",
			code: `cuda_tile.module @m {
  cuda_tile.entry @dangling(%view: !cuda_tile.tile<ptr<f32>>) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %val = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<16x16xf32>
    %tok_pending = cuda_tile.store_view_tko %val, %view, %tok0 : (!cuda_tile.tile<16x16xf32>, !cuda_tile.tile<ptr<f32>>, !cuda_tile.token) -> !cuda_tile.token
    cuda_tile.return
  }
}`,
			wantHazard:  tileir.HazardDanglingAsyncTransfer,
			wantMessage: "has pending token",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mod, err := tileir.ParseString(tt.code)
			if err != nil {
				t.Fatalf("ParseString: %v", err)
			}
			hazards, _ := mod.Verify()
			if len(hazards) == 0 {
				t.Fatalf("expected hazard %v, got none", tt.wantHazard)
			}
			found := false
			for _, h := range hazards {
				if h.Kind == tt.wantHazard && strings.Contains(h.Message, tt.wantMessage) {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("hazards = %v, want hazard %v with message containing %q", hazards, tt.wantHazard, tt.wantMessage)
			}
		})
	}
}

func TestVerifyShapes(t *testing.T) {
	tests := []struct {
		name       string
		code       string
		wantReason string
	}{
		{
			name: "mma inner dimension mismatch",
			code: `cuda_tile.module @m {
  cuda_tile.entry @bad_inner() {
    %A = cuda_tile.constant <f16: 1.0> : !cuda_tile.tile<64x32xf16>
    %B = cuda_tile.constant <f16: 1.0> : !cuda_tile.tile<16x64xf16>
    %C = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>
    %acc = cuda_tile.mma_f %A, %B, %C : (!cuda_tile.tile<64x32xf16>, !cuda_tile.tile<16x64xf16>, !cuda_tile.tile<64x64xf32>) -> !cuda_tile.tile<64x64xf32>
    cuda_tile.return
  }
}`,
			wantReason: "MMA inner dimension mismatch: 32 != 16",
		},
		{
			name: "mma accumulator mismatch",
			code: `cuda_tile.module @m {
  cuda_tile.entry @bad_acc() {
    %A = cuda_tile.constant <f16: 1.0> : !cuda_tile.tile<64x32xf16>
    %B = cuda_tile.constant <f16: 1.0> : !cuda_tile.tile<32x64xf16>
    %C = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<32x32xf32>
    %acc = cuda_tile.mma_f %A, %B, %C : (!cuda_tile.tile<64x32xf16>, !cuda_tile.tile<32x64xf16>, !cuda_tile.tile<32x32xf32>) -> !cuda_tile.tile<64x64xf32>
    cuda_tile.return
  }
}`,
			wantReason: "MMA accumulator shape [32, 32] does not match result dimensions [64, 64]",
		},
		{
			name: "reshape element count mismatch",
			code: `cuda_tile.module @m {
  cuda_tile.entry @bad_reshape() {
    %src = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<16x16xf32>
    %res = cuda_tile.reshape %src : (!cuda_tile.tile<16x16xf32>) -> !cuda_tile.tile<8x16xf32>
    cuda_tile.return
  }
}`,
			wantReason: "reshape total elements mismatch: 256 != 128",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mod, err := tileir.ParseString(tt.code)
			if err != nil {
				t.Fatalf("ParseString: %v", err)
			}
			_, shapeErrors := mod.Verify()
			if len(shapeErrors) == 0 {
				t.Fatalf("expected shape error, got none")
			}
			found := false
			for _, se := range shapeErrors {
				if strings.Contains(se.Reason, tt.wantReason) {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("shapeErrors = %v, want error containing %q", shapeErrors, tt.wantReason)
			}
		})
	}
}

func TestVerifyNilSafety(t *testing.T) {
	var fn *tileir.Function
	h1, s1 := fn.Verify()
	if len(h1) != 0 || len(s1) != 0 {
		t.Errorf("(*Function)(nil).Verify() = %v, %v, want nil", h1, s1)
	}

	var mod *tileir.Module
	h2, s2 := mod.Verify()
	if len(h2) != 0 || len(s2) != 0 {
		t.Errorf("(*Module)(nil).Verify() = %v, %v, want nil", h2, s2)
	}
}
