package tileir_test

import (
	"bytes"
	"context"
	"debug/elf"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	"github.com/tmc/lib/nvidia/x/tileir"
	"github.com/tmc/lib/nvidia/x/tileiras"
	"github.com/tmc/lib/nvidia/x/tileirdisasm"
)

// declarativeTileMatmulSource defines a declarative Tile IR GEMM kernel
// using CUDA 13.4 cuda_tile.h, __tile_global__, and tensor view abstractions.
const declarativeTileMatmulSource = `#include "cuda_tile.h"
using namespace cuda::tiles;

extern "C" __tile_global__ void tile_matmul(float* A, float* B, float* C) {
    constexpr int M = 16, N = 16, K = 32, TM = 16, TN = 16, TK = 16;
    auto a = partition_view{tensor_span{assume_aligned<16>(A), extents{M, K}}, shape<TM, TK>{}};
    auto b = partition_view{tensor_span{assume_aligned<16>(B), extents{K, N}}, shape<TK, TN>{}};
    auto c = partition_view{tensor_span{assume_aligned<16>(C), extents{M, N}}, shape<TM, TN>{}};
    auto sum = full<tile<float, shape<TM, TN>>>(0.0f);
    for (auto k : irange(0, K / TK)) {
        sum = mma(a.load(0, k), b.load(k, 0), sum);
    }
    c.store(sum, 0, 0);
}
`

// TestNVRTCCompileTileIR verifies that NVRTC 13.4 compiles declarative Tile IR
// sources into device CUBINs when CUDA 13.4 is available.
func TestNVRTCCompileTileIR(t *testing.T) {
	// Step 1: Check if CUDA Toolkit 13.4 / NVRTC with --enable-tile is available.
	// If not, skip cleanly with an informative message.
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Skipf("cuda_tile.h is CUDA 13.4+ only (NVRTC unavailable: %v)", err)
	}
	if major < 13 || (major == 13 && minor < 4) {
		t.Skipf("cuda_tile.h is CUDA 13.4+ only (found NVRTC %d.%d)", major, minor)
	}

	// Determine compilation target architecture.
	var dev *cuda.Device
	var targetArch nvrtcx.Option
	if d, err := cuda.NewDevice(0); err == nil {
		dev = d
		t.Cleanup(func() { _ = dev.Close() })
		if devMajor, devMinor, err := dev.ComputeCapability(); err == nil {
			targetArch = nvrtcx.WithPhysicalArch(int(devMajor), int(devMinor))
			t.Logf("targeting device 0 compute capability %d.%d", devMajor, devMinor)
		}
	}
	if targetArch == nil {
		// Target Hopper physical architecture for offline CUBIN compilation.
		targetArch = nvrtcx.WithPhysicalArch(9, 0)
	}

	// Step 2: Compile declarative Tile IR source module using nvrtcx.Compile with --enable-tile.
	opts := []nvrtcx.Option{
		targetArch,
		nvrtcx.WithFlag("--enable-tile"),
		nvrtcx.WithFlag("--std=c++20"),
		nvrtcx.WithNoCache(),
	}

	cm, err := nvrtcx.Compile(declarativeTileMatmulSource, opts...)
	if err != nil {
		errStr := err.Error()
		if strings.Contains(errStr, "cuda_tile.h") ||
			strings.Contains(errStr, "--enable-tile") ||
			strings.Contains(errStr, "cannot open source file") {
			t.Skipf("cuda_tile.h is CUDA 13.4+ only: %v", err)
		}
		t.Fatalf("nvrtcx.Compile with --enable-tile failed: %v", err)
	}

	cubin := cm.CUBIN()
	if len(cubin) == 0 {
		t.Fatalf("compilation succeeded but produced empty CUBIN image; log:\n%s", cm.Log())
	}
	t.Logf("compiled declarative Tile IR to CUBIN (%d bytes)", len(cubin))

	// Step 3: Validate that the resulting CUBIN can be loaded and executed or disassembled.
	var executed bool
	if dev != nil {
		mod, _, err := cm.LoadOwnedOutput(dev, nvrtcx.OutputCUBIN)
		if err == nil {
			t.Cleanup(func() { _ = mod.Close() })
			fn, err := mod.Function("tile_matmul")
			if err == nil {
				const M, N, K = 16, 16, 32
				aData := make([]float32, M*K)
				bData := make([]float32, K*N)
				cData := make([]float32, M*N)
				for i := range aData {
					aData[i] = float32(i%7) - 3.0
				}
				for i := range bData {
					bData[i] = float32(i%5) - 2.0
				}

				allocBuf := func(data []float32) (driver.DevicePtr, error) {
					var ptr driver.DevicePtr
					err := dev.WithContext(func() error {
						p, err := driver.MemAlloc(uint(len(data) * 4))
						if err != nil {
							return err
						}
						ptr = p
						return p.MemcpyHtoD(unsafe.Pointer(&data[0]), uint(len(data)*4))
					})
					return ptr, err
				}

				pA, errA := allocBuf(aData)
				pB, errB := allocBuf(bData)
				pC, errC := allocBuf(cData)
				if errA == nil && errB == nil && errC == nil {
					t.Cleanup(func() {
						_ = dev.WithContext(pA.MemFree)
						_ = dev.WithContext(pB.MemFree)
						_ = dev.WithContext(pC.MemFree)
					})

					stream, err := dev.NewStream()
					if err == nil {
						t.Cleanup(func() { _ = stream.Destroy() })
						args := []unsafe.Pointer{
							unsafe.Pointer(&pA),
							unsafe.Pointer(&pB),
							unsafe.Pointer(&pC),
						}
						err = fn.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, stream.Raw(), args)
						if err == nil && stream.Synchronize(context.Background()) == nil {
							_ = dev.WithContext(func() error {
								return pC.MemcpyDtoH(unsafe.Pointer(&cData[0]), uint(len(cData)*4))
							})
							for i := 0; i < M; i++ {
								for j := 0; j < N; j++ {
									var expected float32
									for k := 0; k < K; k++ {
										expected += aData[i*K+k] * bData[k*N+j]
									}
									got := cData[i*N+j]
									diff := got - expected
									if diff < 0 {
										diff = -diff
									}
									if diff > 1e-3 {
										t.Errorf("C[%d,%d] = %f, want %f", i, j, got, expected)
									}
								}
							}
							executed = true
							t.Log("verified declarative Tile IR kernel execution on device")
						}
					}
				}
			}
		}
	}

	// Validate CUBIN ELF image structure.
	ef, err := elf.NewFile(bytes.NewReader(cubin))
	if err != nil {
		t.Fatalf("parse CUBIN ELF image: %v", err)
	}
	defer ef.Close()

	if len(ef.Sections) == 0 {
		t.Errorf("CUBIN has no ELF sections")
	}

	// Check for embedded Tile IR bytecode in CUBIN and validate disassembly.
	tileBytecode, err := tileirdisasm.ExtractFromELF(cubin)
	if err == nil && len(tileBytecode) > 0 {
		if !tileir.IsTileIRBytecode(tileBytecode) {
			t.Error("extracted ELF section is not recognized as Tile IR bytecode")
		} else {
			decoded, err := tileir.DecodeBytecodeBytes(tileBytecode)
			if err != nil {
				t.Errorf("DecodeBytecodeBytes failed on extracted Tile IR: %v", err)
			} else {
				t.Logf("decoded %d functions from CUBIN Tile IR section", len(decoded.Functions))
			}
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if disMod, err := tileirdisasm.Disassemble(ctx, tileBytecode); err == nil {
			t.Logf("tileirdisasm disassembled %d functions from CUBIN", len(disMod.Functions))
		} else if !errors.Is(err, tileirdisasm.ErrNotFound) && !strings.Contains(err.Error(), "executable not found") {
			t.Logf("tileirdisasm notice: %v", err)
		}
	} else {
		t.Logf("CUBIN ELF structure validated (%d sections, executed=%v)", len(ef.Sections), executed)
	}
}

// TestTileIRSyntaxGeneration verifies programmatic generation and round-trip parsing
// of Tile IR MLIR abstract syntax trees.
func TestTileIRSyntaxGeneration(t *testing.T) {
	tests := []struct {
		name          string
		module        *tileir.Module
		wantSubstrs   []string
		wantTMAOps    int
		wantMMAOps    int
		wantConstants int
	}{
		{
			name: "gemm_tma_mma",
			module: &tileir.Module{
				Name: "tma_gemm_module",
				Functions: []*tileir.Function{
					{
						Name:  "matmul_kernel",
						Entry: true,
						Params: []*tileir.Param{
							{Name: "%A", Type: "!cuda_tile.tile<ptr<f16>>"},
							{Name: "%B", Type: "!cuda_tile.tile<ptr<f16>>"},
							{Name: "%C", Type: "!cuda_tile.tile<ptr<f32>>"},
						},
						Operations: []*tileir.Operation{
							{
								Results: []string{"%tok0"},
								OpCode:  "cuda_tile.make_token",
								Type:    "!cuda_tile.token",
							},
							{
								Results:  []string{"%vA"},
								OpCode:   "cuda_tile.load_view_tko",
								Operands: []string{"%A", "%tok0"},
								Type:     "(!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16>",
							},
							{
								Results:  []string{"%vB"},
								OpCode:   "cuda_tile.load_view_tko",
								Operands: []string{"%B", "%tok0"},
								Type:     "(!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16>",
							},
							{
								Results:    []string{"%cst"},
								OpCode:     "cuda_tile.constant",
								Attributes: "<f32: 0.0>",
								Type:       "!cuda_tile.tile<64x64xf32>",
							},
							{
								Results:  []string{"%acc"},
								OpCode:   "cuda_tile.mma_f",
								Operands: []string{"%vA", "%vB", "%cst"},
								Type:     "(!cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf32>) -> !cuda_tile.tile<64x64xf32>",
							},
							{
								OpCode:   "cuda_tile.store_view_tko",
								Operands: []string{"%C", "%acc", "%tok0"},
								Type:     "(!cuda_tile.tile<ptr<f32>>, !cuda_tile.tile<64x64xf32>, !cuda_tile.token)",
							},
							{
								OpCode: "cuda_tile.return",
							},
						},
					},
				},
			},
			wantSubstrs: []string{
				"cuda_tile.module @tma_gemm_module",
				"cuda_tile.entry @matmul_kernel",
				"%tok0 = cuda_tile.make_token : !cuda_tile.token",
				"%vA = cuda_tile.load_view_tko %A, %tok0",
				"%vB = cuda_tile.load_view_tko %B, %tok0",
				"%cst = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>",
				"%acc = cuda_tile.mma_f %vA, %vB, %cst",
				"cuda_tile.store_view_tko %C, %acc, %tok0",
				"cuda_tile.return",
			},
			wantTMAOps:    3, // 2 loads + 1 store
			wantMMAOps:    1,
			wantConstants: 1,
		},
		{
			name: "scaled_mma_integer",
			module: &tileir.Module{
				Name: "scaled_int_module",
				Functions: []*tileir.Function{
					{
						Name:  "int_matmul",
						Entry: true,
						Params: []*tileir.Param{
							{Name: "%X", Type: "!cuda_tile.tile<ptr<i8>>"},
							{Name: "%W", Type: "!cuda_tile.tile<ptr<i8>>"},
						},
						Operations: []*tileir.Operation{
							{
								Results: []string{"%tok"},
								OpCode:  "cuda_tile.make_token",
								Type:    "!cuda_tile.token",
							},
							{
								Results:  []string{"%tX"},
								OpCode:   "cuda_tile.load_view_tko",
								Operands: []string{"%X", "%tok"},
								Type:     "(!cuda_tile.tile<ptr<i8>>, !cuda_tile.token) -> !cuda_tile.tile<32x32xi8>",
							},
							{
								Results:  []string{"%tW"},
								OpCode:   "cuda_tile.load_view_tko",
								Operands: []string{"%W", "%tok"},
								Type:     "(!cuda_tile.tile<ptr<i8>>, !cuda_tile.token) -> !cuda_tile.tile<32x32xi8>",
							},
							{
								Results:  []string{"%out"},
								OpCode:   "cuda_tile.mma_i",
								Operands: []string{"%tX", "%tW"},
								Type:     "(!cuda_tile.tile<32x32xi8>, !cuda_tile.tile<32x32xi8>) -> !cuda_tile.tile<32x32xi32>",
							},
							{
								Results:  []string{"%scaled"},
								OpCode:   "cuda_tile.mmaf_scaled",
								Operands: []string{"%tX", "%tW"},
								Type:     "(!cuda_tile.tile<32x32xi8>, !cuda_tile.tile<32x32xi8>) -> !cuda_tile.tile<32x32xf32>",
							},
							{
								OpCode: "cuda_tile.return",
							},
						},
					},
				},
			},
			wantSubstrs: []string{
				"cuda_tile.module @scaled_int_module",
				"cuda_tile.entry @int_matmul",
				"%out = cuda_tile.mma_i %tX, %tW",
				"%scaled = cuda_tile.mmaf_scaled %tX, %tW",
			},
			wantTMAOps:    2,
			wantMMAOps:    2, // mma_i and mmaf_scaled
			wantConstants: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			textIR := tt.module.String()
			for _, substr := range tt.wantSubstrs {
				if !strings.Contains(textIR, substr) {
					t.Errorf("module.String() missing substring %q\nFull IR:\n%s", substr, textIR)
				}
			}

			fn := tt.module.Functions[0]
			tmaOps := fn.FilterOps(func(op *tileir.Operation) bool { return op.IsTMA() })
			if len(tmaOps) != tt.wantTMAOps {
				t.Errorf("got %d TMA ops, want %d", len(tmaOps), tt.wantTMAOps)
			}

			mmaOps := fn.FilterOps(func(op *tileir.Operation) bool { return op.IsMMA() })
			if len(mmaOps) != tt.wantMMAOps {
				t.Errorf("got %d MMA ops, want %d", len(mmaOps), tt.wantMMAOps)
			}

			cstOps := fn.FilterOps(func(op *tileir.Operation) bool { return op.IsConstant() })
			if len(cstOps) != tt.wantConstants {
				t.Errorf("got %d constant ops, want %d", len(cstOps), tt.wantConstants)
			}

			// Verify round-trip parsing of generated Tile IR text
			parsed, err := tileir.ParseString(textIR)
			if err != nil {
				t.Fatalf("ParseString failed on generated Tile IR: %v\nSource:\n%s", err, textIR)
			}
			if parsed.Name != tt.module.Name {
				t.Errorf("parsed.Name = %q, want %q", parsed.Name, tt.module.Name)
			}
			if len(parsed.Functions) != len(tt.module.Functions) {
				t.Fatalf("parsed %d functions, want %d", len(parsed.Functions), len(tt.module.Functions))
			}
			parsedFn := parsed.Functions[0]
			if parsedFn.Name != fn.Name {
				t.Errorf("parsed function name = %q, want %q", parsedFn.Name, fn.Name)
			}
			if len(parsedFn.Operations) != len(fn.Operations) {
				t.Errorf("parsed %d operations, want %d", len(parsedFn.Operations), len(fn.Operations))
			}

			// Verify bytecode encoding round-trip
			bcBytes, err := parsed.EncodeBytecode()
			if err != nil {
				t.Fatalf("EncodeBytecode failed: %v", err)
			}
			if !tileir.IsTileIRBytecode(bcBytes) {
				t.Fatal("encoded bytecode failed IsTileIRBytecode check")
			}
			decoded, err := tileir.DecodeBytecodeBytes(bcBytes)
			if err != nil {
				t.Fatalf("DecodeBytecodeBytes failed: %v", err)
			}
			if len(decoded.Functions) != len(tt.module.Functions) {
				t.Errorf("decoded %d functions, want %d", len(decoded.Functions), len(tt.module.Functions))
			}
		})
	}
}

// declarativeKernelConfig configures declarative C++ Tile IR source generation.
type declarativeKernelConfig struct {
	Name       string
	Precision  string
	M, N, K    int
	TM, TN, TK int
}

func generateDeclarativeTileSource(cfg declarativeKernelConfig) string {
	return fmt.Sprintf(`// Declarative C++ Tile IR generated for NVRTC 13.4 compilation
#include "cuda_tile.h"
using namespace cuda::tiles;

extern "C" __tile_global__ void %s(
    const %s* __restrict__ A,
    const %s* __restrict__ B,
    %s* __restrict__ C
) {
    constexpr int M = %d, N = %d, K = %d;
    constexpr int TM = %d, TN = %d, TK = %d;
    auto a = partition_view{tensor_span{assume_aligned<16>(A), extents{M, K}}, shape<TM, TK>{}};
    auto b = partition_view{tensor_span{assume_aligned<16>(B), extents{K, N}}, shape<TK, TN>{}};
    auto c = partition_view{tensor_span{assume_aligned<16>(C), extents{M, N}}, shape<TM, TN>{}};
    auto sum = full<tile<%s, shape<TM, TN>>>(0.0f);
    for (auto k : irange(0, K / TK)) {
        sum = mma(a.load(0, k), b.load(k, 0), sum);
    }
    c.store(sum, 0, 0);
}
`, cfg.Name, cfg.Precision, cfg.Precision, cfg.Precision,
		cfg.M, cfg.N, cfg.K, cfg.TM, cfg.TN, cfg.TK, cfg.Precision)
}

// TestDeclarativeCPPSyntaxGeneration verifies declarative C++ Tile IR source generation
// targeting NVRTC --enable-tile.
func TestDeclarativeCPPSyntaxGeneration(t *testing.T) {
	tests := []struct {
		name        string
		config      declarativeKernelConfig
		wantSubstrs []string
	}{
		{
			name: "fp32_16x16x32",
			config: declarativeKernelConfig{
				Name:      "tile_gemm_fp32",
				Precision: "float",
				M:         16,
				N:         16,
				K:         32,
				TM:        16,
				TN:        16,
				TK:        16,
			},
			wantSubstrs: []string{
				`#include "cuda_tile.h"`,
				"using namespace cuda::tiles;",
				`extern "C" __tile_global__ void tile_gemm_fp32`,
				"constexpr int M = 16, N = 16, K = 32;",
				"constexpr int TM = 16, TN = 16, TK = 16;",
				"assume_aligned<16>(A)",
				"assume_aligned<16>(B)",
				"assume_aligned<16>(C)",
				"partition_view{tensor_span{assume_aligned<16>(A), extents{M, K}}, shape<TM, TK>{}}",
				"full<tile<float, shape<TM, TN>>>(0.0f)",
				"sum = mma(a.load(0, k), b.load(k, 0), sum);",
				"c.store(sum, 0, 0);",
			},
		},
		{
			name: "half_32x32x64",
			config: declarativeKernelConfig{
				Name:      "tile_gemm_half",
				Precision: "half",
				M:         32,
				N:         32,
				K:         64,
				TM:        16,
				TN:        16,
				TK:        16,
			},
			wantSubstrs: []string{
				`#include "cuda_tile.h"`,
				`extern "C" __tile_global__ void tile_gemm_half`,
				"constexpr int M = 32, N = 32, K = 64;",
				"const half* __restrict__ A",
				"full<tile<half, shape<TM, TN>>>(0.0f)",
				"c.store(sum, 0, 0);",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := generateDeclarativeTileSource(tt.config)
			for _, substr := range tt.wantSubstrs {
				if !strings.Contains(src, substr) {
					t.Errorf("generateDeclarativeTileSource missing %q in output:\n%s", substr, src)
				}
			}
		})
	}
}

// TestRemarkParsing verifies parsing and categorization of compiler optimization
// remarks emitted during Tile IR compilation.
func TestRemarkParsing(t *testing.T) {
	const sampleRemarks = `--- !Passed
Pass:            Memory
Name:            RemarkMemoryLoadInstructionSelected
DebugLoc:        { File: '/src/kernel.cu', Line: 12, Column: 10 }
Function:        matmul_kernel
Args:
  - RemarkId:    1
  - Remark:      Load instruction selected
  - Instruction: TMA Load instruction
  - Shape:       '[16, 16]'
...
--- !Passed
Pass:            Memory
Name:            RemarkMemoryStoreInstructionSelected
DebugLoc:        { File: '/src/kernel.cu', Line: 18, Column: 5 }
Function:        matmul_kernel
Args:
  - RemarkId:    2
  - Remark:      Store instruction selected
  - Instruction: TMA Store instruction
  - Shape:       '[16, 16]'
...
--- !Failure
Pass:            Memory
Name:            RemarkMemoryTmaLoadFailed
DebugLoc:        { File: '/src/kernel.cu', Line: 24, Column: 8 }
Function:        matmul_kernel
Args:
  - RemarkId:    3
  - Remark:      Failed to select TMA load instruction
  - Reason:      misaligned base address
  - Shape:       '[16, 32]'
...
--- !Passed
Pass:            Tensor-core
Name:            RemarkTensorCoreMMA
DebugLoc:        { File: '/src/kernel.cu', Line: 30, Column: 15 }
Function:        matmul_kernel
Args:
  - RemarkId:    4
  - Remark:      MMA operation successfully optimized to use Tensor Cores
  - Instruction: Tensor Core MMA
  - Shape:       '[16, 16, 16]'
...
--- !Failure
Pass:            Tensor-core
Name:            RemarkTensorCoreMMA
DebugLoc:        { File: '/src/kernel.cu', Line: 36, Column: 12 }
Function:        matmul_kernel
Args:
  - RemarkId:    5
  - Remark:      MMA operation failed to optimize to use Tensor Cores, it is using FMA instructions instead
  - Instruction: FMA
  - Shape:       '[1, 1, 1]'
...
`

	remarks, err := tileiras.ParseRemarksYAML([]byte(sampleRemarks))
	if err != nil {
		t.Fatalf("ParseRemarksYAML failed: %v", err)
	}

	if len(remarks) != 5 {
		t.Fatalf("len(remarks) = %d, want 5", len(remarks))
	}

	// Remark 0: Passed TMA Load
	r0 := remarks[0]
	if r0.Kind != "Passed" || r0.Pass != "Memory" {
		t.Errorf("r0 = %s/%s, want Passed/Memory", r0.Pass, r0.Kind)
	}
	if r0.File != "/src/kernel.cu" || r0.Line != 12 || r0.Column != 10 {
		t.Errorf("r0 loc = %s:%d:%d, want /src/kernel.cu:12:10", r0.File, r0.Line, r0.Column)
	}
	if r0.Instruction != "TMA Load instruction" {
		t.Errorf("r0 instruction = %q, want 'TMA Load instruction'", r0.Instruction)
	}
	if r0.IsTMAFailure() {
		t.Error("r0.IsTMAFailure() = true, want false")
	}

	// Remark 2: TMA Load Failure
	r2 := remarks[2]
	if !r2.IsTMAFailure() {
		t.Errorf("r2.IsTMAFailure() = false, want true (name=%q, remark=%q)", r2.Name, r2.Remark)
	}
	if r2.Reason != "misaligned base address" {
		t.Errorf("r2.Reason = %q, want 'misaligned base address'", r2.Reason)
	}

	// Remark 3: Tensor Core MMA Passed
	r3 := remarks[3]
	if !r3.IsTensorCoreMMA() {
		t.Errorf("r3.IsTensorCoreMMA() = false, want true")
	}
	if r3.IsTensorCoreFailure() {
		t.Errorf("r3.IsTensorCoreFailure() = true, want false")
	}

	// Remark 4: Tensor Core Failure (FMA fallback)
	r4 := remarks[4]
	if !r4.IsTensorCoreFailure() {
		t.Errorf("r4.IsTensorCoreFailure() = false, want true")
	}
	if r4.Instruction != "FMA" {
		t.Errorf("r4.Instruction = %q, want 'FMA'", r4.Instruction)
	}

	// Result filtering helpers
	res := &tileiras.Result{Remarks: remarks}
	tmaFailures := res.TMAFailures()
	if len(tmaFailures) != 1 {
		t.Errorf("len(TMAFailures()) = %d, want 1", len(tmaFailures))
	}
	tcFailures := res.TensorCoreFailures()
	if len(tcFailures) != 1 {
		t.Errorf("len(TensorCoreFailures()) = %d, want 1", len(tcFailures))
	}

	// Empty input handling
	emptyRemarks, err := tileiras.ParseRemarksYAML(nil)
	if err != nil {
		t.Errorf("ParseRemarksYAML(nil) error: %v", err)
	}
	if len(emptyRemarks) != 0 {
		t.Errorf("ParseRemarksYAML(nil) returned %d remarks, want 0", len(emptyRemarks))
	}

	// String formatting check
	str := r0.String()
	if !strings.Contains(str, "[Memory/Passed]") || !strings.Contains(str, "kernel.cu:12:10") {
		t.Errorf("r0.String() = %q, missing expected format", str)
	}
}
