package tileir_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileir"
)

func TestTypeSystem(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		wantStr     string
		wantKind    tileir.TypeKind
		wantIsTile  bool
		wantIsFloat bool
		wantIsInt   bool
	}{
		{
			name:        "i32 primitive",
			input:       "i32",
			wantStr:     "i32",
			wantKind:    tileir.KindInt32,
			wantIsTile:  false,
			wantIsFloat: false,
			wantIsInt:   true,
		},
		{
			name:        "bf16 primitive",
			input:       "bf16",
			wantStr:     "bf16",
			wantKind:    tileir.KindBFloat16,
			wantIsTile:  false,
			wantIsFloat: true,
			wantIsInt:   false,
		},
		{
			name:        "f8E4M3FN primitive",
			input:       "f8E4M3FN",
			wantStr:     "f8E4M3FN",
			wantKind:    tileir.KindFloat8E4M3,
			wantIsTile:  false,
			wantIsFloat: true,
			wantIsInt:   false,
		},
		{
			name:        "tile 2D f32",
			input:       "!cuda_tile.tile<8x16xf32>",
			wantStr:     "!cuda_tile.tile<8x16xf32>",
			wantKind:    tileir.KindTile,
			wantIsTile:  true,
			wantIsFloat: false,
			wantIsInt:   false,
		},
		{
			name:        "tile scalar bf16",
			input:       "tile<bf16>",
			wantStr:     "!cuda_tile.tile<bf16>",
			wantKind:    tileir.KindTile,
			wantIsTile:  true,
			wantIsFloat: false,
			wantIsInt:   false,
		},
		{
			name:        "pointer to f32",
			input:       "!cuda_tile.ptr<f32>",
			wantStr:     "!cuda_tile.ptr<f32>",
			wantKind:    tileir.KindPointer,
			wantIsTile:  false,
			wantIsFloat: false,
			wantIsInt:   false,
		},
		{
			name:        "tensor view 3D",
			input:       "tensor_view<4x8x16xf16>",
			wantStr:     "!cuda_tile.tensor_view<4x8x16xf16>",
			wantKind:    tileir.KindTensorView,
			wantIsTile:  false,
			wantIsFloat: false,
			wantIsInt:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			typ, err := tileir.ParseType(tt.input)
			if err != nil {
				t.Fatalf("ParseType(%q) error: %v", tt.input, err)
			}
			if got := typ.Kind(); got != tt.wantKind {
				t.Errorf("Kind() = %v, want %v", got, tt.wantKind)
			}
			if got := typ.String(); got != tt.wantStr {
				t.Errorf("String() = %q, want %q", got, tt.wantStr)
			}
			if got := typ.IsTile(); got != tt.wantIsTile {
				t.Errorf("IsTile() = %v, want %v", got, tt.wantIsTile)
			}
			if got := typ.IsFloat(); got != tt.wantIsFloat {
				t.Errorf("IsFloat() = %v, want %v", got, tt.wantIsFloat)
			}
			if got := typ.IsInt(); got != tt.wantIsInt {
				t.Errorf("IsInt() = %v, want %v", got, tt.wantIsInt)
			}
		})
	}
}

func TestTextualParser(t *testing.T) {
	const sample = `
// Module comments
cuda_tile.module @gemm_kernels {
  cuda_tile.entry @matmul_tma(%A: !cuda_tile.tile<ptr<f16>>, %B: !cuda_tile.tile<ptr<f16>>, %C: !cuda_tile.tile<ptr<f32>>) loc(#loc1) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token loc(#loc2)
    %vA = cuda_tile.load_view_tko %A, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16> loc(#loc3)
    %vB = cuda_tile.load_view_tko %B, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16> loc(#loc4)
    %cst = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32> loc(#loc5)
    %acc = cuda_tile.mma_f %vA, %vB, %cst : (!cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf32>) -> !cuda_tile.tile<64x64xf32> loc(#loc6)
    cuda_tile.return loc(#loc7)
  }
}

#loc1 = loc("kernel.cu":10:1)
#loc2 = loc("kernel.cu":12:5)
#loc3 = loc("kernel.cu":14:10)
#loc4 = loc("kernel.cu":16:10)
#loc5 = loc("kernel.cu":18:5)
#loc6 = loc("kernel.cu":20:10)
#loc7 = loc("kernel.cu":22:1)
`

	mod, err := tileir.ParseString(sample)
	if err != nil {
		t.Fatalf("ParseString error: %v", err)
	}

	if mod.Name != "gemm_kernels" {
		t.Errorf("mod.Name = %q, want gemm_kernels", mod.Name)
	}
	if len(mod.Functions) != 1 {
		t.Fatalf("got %d functions, want 1", len(mod.Functions))
	}

	fn := mod.FindFunction("matmul_tma")
	if fn == nil {
		t.Fatal("FindFunction(matmul_tma) returned nil")
	}
	if !fn.Entry {
		t.Error("fn.Entry = false, want true")
	}
	if len(fn.Params) != 3 {
		t.Errorf("len(Params) = %d, want 3", len(fn.Params))
	}
	if len(fn.Operations) != 6 {
		t.Fatalf("len(Operations) = %d, want 6", len(fn.Operations))
	}

	// Check TMA operations
	tmaOps := fn.FilterOps(func(op *tileir.Operation) bool {
		return op.IsTMA()
	})
	if len(tmaOps) != 2 {
		t.Errorf("got %d TMA ops, want 2", len(tmaOps))
	}

	// Check MMA operations
	mmaOps := fn.FilterOps(func(op *tileir.Operation) bool {
		return op.IsMMA()
	})
	if len(mmaOps) != 1 {
		t.Errorf("got %d MMA ops, want 1", len(mmaOps))
	}

	// Check locations
	if fn.Loc.File != "kernel.cu" || fn.Loc.Line != 10 {
		t.Errorf("fn.Loc = %v, want kernel.cu:10", fn.Loc)
	}
	if mmaOps[0].Loc.Line != 20 {
		t.Errorf("mma op line = %d, want 20", mmaOps[0].Loc.Line)
	}

	// Round-trip printing
	printed := mod.String()
	if !strings.Contains(printed, "cuda_tile.entry @matmul_tma") {
		t.Errorf("printed module missing entry: %s", printed)
	}
}

func TestBytecodeGoldens(t *testing.T) {
	goldenDir := os.Getenv("HOME") + "/go/src/github.com/NVIDIA/cuda-tile/test/Bytecode/encoding_stability/goldens"
	files, err := filepath.Glob(filepath.Join(goldenDir, "*.golden.tileirbc"))
	if err != nil || len(files) == 0 {
		t.Skip("cuda-tile golden files not found, skipping")
	}

	for _, file := range files {
		name := filepath.Base(file)
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(file)
			if err != nil {
				t.Fatalf("failed to read %s: %v", file, err)
			}

			if !tileir.IsTileIRBytecode(data) {
				t.Fatalf("%s is not recognized as Tile IR bytecode", name)
			}

			mod, err := tileir.DecodeBytecodeBytes(data)
			if err != nil {
				t.Fatalf("DecodeBytecodeBytes(%s) failed: %v", name, err)
			}

			if mod.Version.Major != 13 || mod.Version.Minor != 4 {
				t.Errorf("unexpected version: %v", mod.Version)
			}

			if len(mod.Functions) == 0 {
				t.Errorf("no functions decoded in %s", name)
			}

			// Convert to AST module
			astMod := mod.ToModule()
			if astMod == nil {
				t.Fatal("ToModule() returned nil")
			}
			if len(astMod.Functions) != len(mod.Functions) {
				t.Errorf("AST function count mismatch: %d vs %d", len(astMod.Functions), len(mod.Functions))
			}
		})
	}
}

func TestSubByteTypeAliases(t *testing.T) {
	tests := []struct {
		input    string
		wantKind tileir.TypeKind
	}{
		{"fp8", tileir.KindFloat8E4M3},
		{"fp8_e4m3", tileir.KindFloat8E4M3},
		{"e4m3", tileir.KindFloat8E4M3},
		{"e4m3fn", tileir.KindFloat8E4M3},
		{"fp8_e5m2", tileir.KindFloat8E5M2},
		{"e5m2", tileir.KindFloat8E5M2},
		{"fp8_e8m0", tileir.KindFloat8E8M0},
		{"e8m0", tileir.KindFloat8E8M0},
		{"fp4", tileir.KindFloat4E2M1},
		{"f4", tileir.KindFloat4E2M1},
		{"e2m1", tileir.KindFloat4E2M1},
		{"f4e2m1", tileir.KindFloat4E2M1},
		{"fp8_e5m3", tileir.KindFloat8E5M3},
		{"e5m3", tileir.KindFloat8E5M3},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			typ, err := tileir.ParseType(tt.input)
			if err != nil {
				t.Fatalf("ParseType(%q): %v", tt.input, err)
			}
			if typ.Kind() != tt.wantKind {
				t.Errorf("Kind() = %v, want %v", typ.Kind(), tt.wantKind)
			}
		})
	}
}

func TestEncodeBytecodeRoundTrip(t *testing.T) {
	const textSample = `cuda_tile.module @subbyte_kernels {
  cuda_tile.entry @gemm_fp4(%A: !cuda_tile.tile<ptr<f4E2M1FN>>, %B: !cuda_tile.tile<ptr<f4E2M1FN>>, %C: !cuda_tile.tile<ptr<f32>>) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %vA = cuda_tile.load_view_tko %A, %tok0 : (!cuda_tile.tile<ptr<f4E2M1FN>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf4E2M1FN>
    %vB = cuda_tile.load_view_tko %B, %tok0 : (!cuda_tile.tile<ptr<f4E2M1FN>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf4E2M1FN>
    %cst = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>
    %acc = cuda_tile.mma_f %vA, %vB, %cst : (!cuda_tile.tile<64x64xf4E2M1FN>, !cuda_tile.tile<64x64xf4E2M1FN>, !cuda_tile.tile<64x64xf32>) -> !cuda_tile.tile<64x64xf32>
    cuda_tile.return
  }
}
`
	mod, err := tileir.ParseString(textSample)
	if err != nil {
		t.Fatalf("ParseString failed: %v", err)
	}

	bcBytes, err := mod.EncodeBytecode()
	if err != nil {
		t.Fatalf("EncodeBytecode failed: %v", err)
	}

	if !tileir.IsTileIRBytecode(bcBytes) {
		t.Fatal("encoded bytes not recognized by IsTileIRBytecode")
	}

	decoded, err := tileir.DecodeBytecodeBytes(bcBytes)
	if err != nil {
		t.Fatalf("DecodeBytecodeBytes failed: %v", err)
	}

	if len(decoded.Functions) != 1 {
		t.Fatalf("decoded %d functions, want 1", len(decoded.Functions))
	}
	fn := decoded.Functions[0]
	if fn.Name != "gemm_fp4" {
		t.Errorf("fn.Name = %q, want gemm_fp4", fn.Name)
	}
	if !fn.IsEntry {
		t.Errorf("fn.IsEntry = false, want true")
	}
}

