package qualify

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

type mockPTXASCompiler struct {
	fn func(iter int, cudaSource, arch string) (string, error)
	n  int
}

func (m *mockPTXASCompiler) CompilePTXAS(_ context.Context, cudaSource, arch string) (ptxasinfo.Report, error) {
	m.n++
	if strings.TrimSpace(cudaSource) == "" {
		return ptxasinfo.Report{}, fmt.Errorf("empty cudaSource on call %d", m.n)
	}
	logText, err := m.fn(m.n, cudaSource, arch)
	if err != nil {
		return ptxasinfo.Report{}, err
	}
	rep, err := ptxasinfo.Parse(strings.NewReader(logText))
	if err != nil {
		return ptxasinfo.Report{}, fmt.Errorf("parse mock ptxas log (call %d): %w", m.n, err)
	}
	return *rep, nil
}

func TestRefinePlanByPTXASLoop(t *testing.T) {
	t.Run("MultiIterationConvergence", func(t *testing.T) {
		plan := &hopper.Plan{
			Name: "loop_converge",
			Inputs: []hopper.Tensor{
				{Name: "x", DType: hopper.BF16, Shape: []int{256, 128}},
				{Name: "w0", DType: hopper.BF16, Shape: []int{5120, 128}},
				{Name: "w1", DType: hopper.BF16, Shape: []int{128, 128}},
				{Name: "sx", DType: hopper.BF16, Shape: []int{2, 2048}},
				{Name: "sw", DType: hopper.BF16, Shape: []int{4, 2048}},
			},
			Temps: []hopper.Tensor{
				{Name: "t0", DType: hopper.F32, Shape: []int{256, 5120}},
				{Name: "t1", DType: hopper.F32, Shape: []int{256, 128}},
			},
			Output: hopper.Tensor{Name: "out", DType: hopper.F32, Shape: []int{2, 4}},
			Kernels: []hopper.PlanKernel{
				{
					Tile: &hopper.TileGEMM{
						Name:        "tile_proj",
						N:           5120,
						K:           128,
						Weight:      hopper.WeightBF16,
						Out:         hopper.F32,
						BM:          128,
						BN:          128,
						Stages:      3,
						WGMMA:       true,
						Swizzle128B: true,
					},
					M:    256,
					Args: []string{"x", "w0", "t0"},
				},
				{
					Tile: &hopper.TileGEMM{
						Name:   "tile_ffn",
						N:      128,
						K:      128,
						Weight: hopper.WeightBF16,
						Out:    hopper.F32,
						BM:     128,
						BN:     64,
						Stages: 2,
						WGMMA:  true,
					},
					M:    256,
					Args: []string{"x", "w1", "t1"},
				},
				{
					GEMM: &hopper.SkinnyGEMM{
						Name:         "skinny_head",
						N:            4,
						K:            2048,
						MaxM:         2,
						Weight:       hopper.WeightBF16,
						Out:          hopper.F32,
						RowsPerBlock: 2,
						Warps:        4,
						Unroll:       4,
						Prefetch:     2,
					},
					M:    2,
					Args: []string{"sx", "sw", "out"},
				},
			},
		}

		compiler := &mockPTXASCompiler{
			fn: func(iter int, _ string, arch string) (string, error) {
				if arch != "sm_90a" {
					return "", fmt.Errorf("unexpected arch %q", arch)
				}
				switch iter {
				case 1:
					// Iteration 1:
					// - tile_proj (BM=128, BN=128, Stages=3, Grid=80): spills 24B store / 32B load -> steps down BN 128 -> 64.
					// - tile_ffn (WGMMA=true, WarpSpecialized=false): C7519 WGMMA serialized -> promotes to WarpSpecialized=true.
					// - skinny_head (Prefetch=2, RowsPerBlock=2, Unroll=4): spills 16B store / 16B load -> disables Prefetch=0 and steps down RowsPerBlock 2 -> 1.
					return `
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z9tile_projiPKtS0_Pf' for 'sm_90a'
ptxas info    : Function properties for _Z9tile_projiPKtS0_Pf
    32 bytes stack frame, 24 bytes spill stores, 32 bytes spill loads
ptxas info    : Used 184 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : (C7519) Potential Performance Loss: wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path in the function '_Z8tile_ffniPKtS0_Pf' in line 142
ptxas info    : Compiling entry function '_Z8tile_ffniPKtS0_Pf' for 'sm_90a'
ptxas info    : Function properties for _Z8tile_ffniPKtS0_Pf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 120 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z11skinny_headILi2EEvPKtS1_Pf' for 'sm_90a'
ptxas info    : Function properties for _Z11skinny_headILi2EEvPKtS1_Pf
    16 bytes stack frame, 16 bytes spill stores, 16 bytes spill loads
ptxas info    : Used 96 registers, 256 bytes smem, 384 bytes cmem[0]
`, nil
				case 2:
					// Iteration 2:
					// - tile_proj (BM=128, BN=64, Stages=3): now has GridBlocks=160 on 132 SMs (1.212 waves -> BottleneckWaveTail) AND still spills 8B -> steps down Stages 3 -> 2 and enables StreamK=132.
					// - tile_ffn (WarpSpecialized=true): still reports C7519 -> fallback disables WGMMA=false.
					// - skinny_head (Prefetch=0, RowsPerBlock=1, Unroll=4): still spills 8B -> steps down Unroll 4 -> 2.
					return `
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z9tile_projiPKtS0_Pf' for 'sm_90a'
ptxas info    : Function properties for _Z9tile_projiPKtS0_Pf
    16 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads
ptxas info    : Used 148 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : (C7519) Potential Performance Loss: wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path in the function '_Z8tile_ffniPKtS0_Pf' in line 188
ptxas info    : Compiling entry function '_Z8tile_ffniPKtS0_Pf' for 'sm_90a'
ptxas info    : Function properties for _Z8tile_ffniPKtS0_Pf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 128 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z11skinny_headILi2EEvPKtS1_Pf' for 'sm_90a'
ptxas info    : Function properties for _Z11skinny_headILi2EEvPKtS1_Pf
    8 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads
ptxas info    : Used 80 registers, 256 bytes smem, 384 bytes cmem[0]
`, nil
				default:
					// Iteration 3: all three kernels are clean (0 spills, no C7519, StreamK=132 eliminates wave tail).
					return `
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z9tile_projiPKtS0_PfS1_' for 'sm_90a'
ptxas info    : Function properties for _Z9tile_projiPKtS0_PfS1_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 120 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z8tile_ffniPKtS0_Pf' for 'sm_90a'
ptxas info    : Function properties for _Z8tile_ffniPKtS0_Pf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 96 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z11skinny_headILi2EEvPKtS1_Pf' for 'sm_90a'
ptxas info    : Function properties for _Z11skinny_headILi2EEvPKtS1_Pf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 64 registers, 256 bytes smem, 384 bytes cmem[0]
`, nil
				}
			},
		}

		res, err := RefinePlanByPTXASLoop(t.Context(), plan, compiler, PTXASLoopConfig{
			Arch:          "sm_90a",
			Device:        "NVIDIA H200",
			DeviceSMs:     132,
			MaxIterations: 4,
			Gate: StaticGate{
				NoSpills:     true,
				MaxRegisters: 192,
				MinCTAsPerSM: 1,
			},
		})
		if err != nil {
			t.Fatalf("RefinePlanByPTXASLoop: %v", err)
		}
		if !res.Converged {
			t.Fatalf("expected Converged=true, got false (iterations: %+v)", res.Iterations)
		}
		if len(res.Iterations) != 3 {
			t.Fatalf("len(res.Iterations) = %d, want 3", len(res.Iterations))
		}

		// Verify Iteration 1 telemetry.
		it1 := res.Iterations[0]
		if it1.Iteration != 1 || it1.Clean || it1.TotalSpillBytes != 88 || it1.SerializedKernels != 1 {
			t.Fatalf("unexpected iteration 1 record: %+v", it1)
		}
		if !slices.Contains(it1.Bottlenecks["tile_proj"], BottleneckRegisterSpill) {
			t.Fatalf("iteration 1 tile_proj bottlenecks = %v, want register_spill", it1.Bottlenecks["tile_proj"])
		}
		if !slices.Contains(it1.Bottlenecks["tile_ffn"], BottleneckWGMMASerial) {
			t.Fatalf("iteration 1 tile_ffn bottlenecks = %v, want wgmma_serialization", it1.Bottlenecks["tile_ffn"])
		}
		if !slices.Contains(it1.Bottlenecks["skinny_head"], BottleneckRegisterSpill) {
			t.Fatalf("iteration 1 skinny_head bottlenecks = %v, want register_spill", it1.Bottlenecks["skinny_head"])
		}

		// Verify Iteration 2 telemetry (wave-tail exposed on tile_proj after BN 128 -> 64).
		it2 := res.Iterations[1]
		if it2.Iteration != 2 || it2.Clean || it2.TotalSpillBytes != 32 || it2.SerializedKernels != 1 {
			t.Fatalf("unexpected iteration 2 record: %+v", it2)
		}
		for _, want := range []BottleneckKind{BottleneckRegisterSpill, BottleneckWaveTail} {
			if !slices.Contains(it2.Bottlenecks["tile_proj"], want) {
				t.Fatalf("iteration 2 tile_proj bottlenecks = %v, missing %v", it2.Bottlenecks["tile_proj"], want)
			}
		}

		// Verify Iteration 3 telemetry (clean fixed point).
		it3 := res.Iterations[2]
		if it3.Iteration != 3 || !it3.Clean || it3.TotalSpillBytes != 0 || it3.SerializedKernels != 0 || len(it3.Actions) != 0 {
			t.Fatalf("unexpected iteration 3 record: %+v", it3)
		}

		// Verify refined schedule knobs.
		gotProj := res.RefinedPlan.Kernels[0].Tile
		if gotProj.BN != 64 || gotProj.Stages != 2 || gotProj.StreamK != 132 {
			t.Fatalf("refined tile_proj = %+v, want BN=64 Stages=2 StreamK=132", *gotProj)
		}
		if len(res.RefinedPlan.Kernels[0].Args) != 4 {
			t.Fatalf("expected StreamK workspace arg appended to tile_proj, got %v", res.RefinedPlan.Kernels[0].Args)
		}
		gotFFN := res.RefinedPlan.Kernels[1].Tile
		if gotFFN.WGMMA || gotFFN.WarpSpecialized {
			t.Fatalf("refined tile_ffn = %+v, want WGMMA=false after fallback", *gotFFN)
		}
		gotSkinny := res.RefinedPlan.Kernels[2].GEMM
		if gotSkinny.Prefetch != 0 || gotSkinny.RowsPerBlock != 1 || gotSkinny.Unroll != 2 {
			t.Fatalf("refined skinny_head = %+v, want Prefetch=0 RowsPerBlock=1 Unroll=2", *gotSkinny)
		}

		// Verify original plan was not mutated.
		if plan.Kernels[0].Tile.BN != 128 || plan.Kernels[0].Tile.Stages != 3 || plan.Kernels[0].Tile.StreamK != 0 {
			t.Fatalf("original plan was mutated: %+v", *plan.Kernels[0].Tile)
		}
	})

	t.Run("AlreadyCleanFixedPointOnIteration1", func(t *testing.T) {
		plan := &hopper.Plan{
			Name: "clean_plan",
			Inputs: []hopper.Tensor{
				{Name: "x", DType: hopper.BF16, Shape: []int{256, 128}},
				{Name: "w", DType: hopper.BF16, Shape: []int{256, 128}},
			},
			Output: hopper.Tensor{Name: "out", DType: hopper.F32, Shape: []int{256, 256}},
			Kernels: []hopper.PlanKernel{
				{
					Tile: &hopper.TileGEMM{
						Name:        "clean_tile",
						N:           256,
						K:           128,
						Weight:      hopper.WeightBF16,
						Out:         hopper.F32,
						BM:          128,
						BN:          128,
						Stages:      2,
						WGMMA:       true,
						Swizzle128B: true,
					},
					M:    256,
					Args: []string{"x", "w", "out"},
				},
			},
		}

		compiler := &mockPTXASCompiler{
			fn: func(_ int, _ string, _ string) (string, error) {
				return `
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z10clean_tileiPKtS0_Pf' for 'sm_90a'
ptxas info    : Function properties for _Z10clean_tileiPKtS0_Pf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 112 registers, 1024 bytes smem, 384 bytes cmem[0]
`, nil
			},
		}

		res, err := RefinePlanByPTXASLoop(t.Context(), plan, compiler, PTXASLoopConfig{
			Arch:      "sm_90a",
			Device:    "NVIDIA H200",
			DeviceSMs: 132,
			Gate:      StaticGate{NoSpills: true, MaxRegisters: 168, MinCTAsPerSM: 2},
		})
		if err != nil {
			t.Fatalf("RefinePlanByPTXASLoop: %v", err)
		}
		if !res.Converged {
			t.Fatal("expected Converged=true on iteration 1")
		}
		if len(res.Iterations) != 1 {
			t.Fatalf("len(res.Iterations) = %d, want 1", len(res.Iterations))
		}
		if !res.Iterations[0].Clean || len(res.Iterations[0].Actions) != 0 || len(res.Iterations[0].Bottlenecks) != 0 {
			t.Fatalf("unexpected iteration 1 record: %+v", res.Iterations[0])
		}
		if compiler.n != 1 {
			t.Fatalf("compiler invoked %d times, want 1", compiler.n)
		}
	})

	t.Run("BoundedTerminationWhenUnsatisfiable", func(t *testing.T) {
		plan := &hopper.Plan{
			Name: "unsat_plan",
			Inputs: []hopper.Tensor{
				{Name: "x", DType: hopper.BF16, Shape: []int{128, 128}},
				{Name: "w", DType: hopper.BF16, Shape: []int{128, 128}},
			},
			Output: hopper.Tensor{Name: "out", DType: hopper.F32, Shape: []int{128, 128}},
			Kernels: []hopper.PlanKernel{
				{
					Tile: &hopper.TileGEMM{
						Name:        "unsat_tile",
						N:           128,
						K:           128,
						Weight:      hopper.WeightBF16,
						Out:         hopper.F32,
						BM:          128,
						BN:          128,
						Stages:      3,
						WGMMA:       true,
						Swizzle128B: true,
					},
					M:    128,
					Args: []string{"x", "w", "out"},
				},
			},
		}

		// Compiler persistently reports spills and 220 registers (> MaxRegisters 128) on every iteration.
		compiler := &mockPTXASCompiler{
			fn: func(_ int, _ string, _ string) (string, error) {
				return `
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z10unsat_tileiPKtS0_Pf' for 'sm_90a'
ptxas info    : Function properties for _Z10unsat_tileiPKtS0_Pf
    64 bytes stack frame, 32 bytes spill stores, 32 bytes spill loads
ptxas info    : Used 220 registers, 1024 bytes smem, 384 bytes cmem[0]
`, nil
			},
		}

		res, err := RefinePlanByPTXASLoop(t.Context(), plan, compiler, PTXASLoopConfig{
			Arch:          "sm_90a",
			Device:        "NVIDIA H200",
			DeviceSMs:     132,
			MaxIterations: 2,
			Gate:          StaticGate{NoSpills: true, MaxRegisters: 128, MinCTAsPerSM: 2},
		})
		if err != nil {
			t.Fatalf("RefinePlanByPTXASLoop: %v", err)
		}
		if res.Converged {
			t.Fatal("expected Converged=false when constraint is never satisfied")
		}
		if len(res.Iterations) != 2 {
			t.Fatalf("len(res.Iterations) = %d, want 2 (bounded by MaxIterations)", len(res.Iterations))
		}
		for i, it := range res.Iterations {
			if it.Clean {
				t.Fatalf("iteration %d unexpectedly marked Clean", i+1)
			}
		}
		if res.RefinedPlan.Kernels[0].Tile.BN != 64 || res.RefinedPlan.Kernels[0].Tile.Stages != 2 {
			t.Fatalf("expected tile stepped down to BN=64 Stages=2 after 2 iterations, got %+v", *res.RefinedPlan.Kernels[0].Tile)
		}
	})

	t.Run("NVCCPTXASCompilerMockBinary", func(t *testing.T) {
		dir := t.TempDir()
		fakeNVCC := filepath.Join(dir, "fake_nvcc.sh")
		script := `#!/bin/sh
echo "ptxas info    : 0 bytes gmem" >&2
echo "ptxas info    : Compiling entry function '_Z7kern_k0iPf' for 'sm_121a'" >&2
echo "ptxas info    : Function properties for _Z7kern_k0iPf" >&2
echo "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads" >&2
echo "ptxas info    : Used 72 registers, 512 bytes smem, 384 bytes cmem[0]" >&2
exit 0
`
		if err := os.WriteFile(fakeNVCC, []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}
		c := NVCCPTXASCompiler{NVCC: fakeNVCC}
		rep, err := c.CompilePTXAS(t.Context(), "__global__ void kern_k0(int n, float* y) {}", "sm_121a")
		if err != nil {
			t.Fatalf("NVCCPTXASCompiler.CompilePTXAS: %v", err)
		}
		fns := rep.Functions()
		if len(fns) != 1 || fns[0].Registers != 72 || fns[0].SharedMem != 512 {
			t.Fatalf("unexpected parsed functions: %+v", fns)
		}
	})
}
