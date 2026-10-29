package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
	"github.com/tmc/lib/nvidia/x/nvl72"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
	"github.com/tmc/lib/nvidia/x/tcgen05"
)

// A kernel is one emitted translation unit the compile checks build.
// emit returns its source for arch and the dynamic shared memory one block
// uses, or ok false if the kernel is not meant for arch.
type kernel struct {
	name string
	emit func(arch string) (src string, shared int, ok bool, err error)
}

func blackwellDatacenter(arch string) bool {
	return strings.HasPrefix(arch, "sm_100") || strings.HasPrefix(arch, "sm_103")
}

func consumerBlackwell(arch string) bool { return strings.HasPrefix(arch, "sm_12") }

// tile emits g for arch. On a 99 KiB part it drops to two stages, as the
// llama-infer plans do, when three do not fit.
func tile(g hopper.TileGEMM, want func(string) bool) func(string) (string, int, bool, error) {
	return func(arch string) (string, int, bool, error) {
		if want != nil && !want(arch) {
			return "", 0, false, nil
		}
		if limit := sharedPerBlock[strings.TrimSuffix(arch, "a")]; limit > 0 && g.SharedBytes() > limit {
			g.Stages = 2
		}
		src, err := g.Source()
		return src, g.SharedBytes(), true, err
	}
}

func source(f func() (string, error), want func(string) bool) func(string) (string, int, bool, error) {
	return func(arch string) (string, int, bool, error) {
		if want != nil && !want(arch) {
			return "", 0, false, nil
		}
		src, err := f()
		return src, 0, true, err
	}
}

// kernels returns the compile suite: the e-graph tile GEMMs the llama-infer
// plans emit (BF16 and grouped NVFP4 MoE), the sm_12x FP4 mma.sync path, the
// sm_100a/sm_103a tcgen05 block-scaled GEMM, and the NVL72 multimem
// collectives at 72 ranks.
func kernels() []kernel {
	bf16 := hopper.TileGEMM{Name: "doctor_tile_bf16", N: 2304, K: 8192, Weight: hopper.WeightBF16, Out: hopper.BF16}
	moe := hopper.TileGEMM{Name: "doctor_moe_w13_nvfp4", N: 8192, K: 4096, Weight: hopper.WeightNVFP4, Out: hopper.BF16,
		Grouped: true, Gather: true, GLU: true, Epilogue: hopper.MustLambda("(mul (silu $0) $1)")}
	fp4 := hopper.TileGEMM{Name: "doctor_tile_fp4_mma", N: 2304, K: 8192, Weight: hopper.WeightNVFP4, Out: hopper.BF16, BlackwellFP4: true}
	ar := nvl72.AllReduceResidualRMSNorm{Name: "doctor_allreduce_rmsnorm", Cols: 8192, Ranks: 72, DType: nvl72.BF16}
	combine := nvl72.NVLinkMoECombine{Name: "doctor_moe_combine", Cols: 8192, TopK: 8, Ranks: 72, DType: nvl72.BF16}
	return []kernel{
		{"tile_bf16", tile(bf16, nil)},
		{"moe_w13_nvfp4", tile(moe, nil)},
		{"tile_fp4_mma", tile(fp4, consumerBlackwell)},
		{"tcgen05_scale_vec_4x", source(tcgen05.TCGen05ScaleVec4XGEMM{}.CUDAFile, blackwellDatacenter)},
		{"allreduce_rmsnorm_72", source(ar.Source, nil)},
		{"moe_combine_72", source(combine.Source, nil)},
	}
}

// compileSuite compiles every kernel meant for each arch with nvcc and
// reads ptxas's resource report. It needs a toolkit, not a GPU.
func compileSuite(ctx context.Context, nvcc string, archs []string, timeout time.Duration) []Compile {
	var cs []Compile
	dir, err := os.MkdirTemp("", "nvl72doctor")
	if err != nil {
		return []Compile{{Kernel: "suite", Error: err.Error()}}
	}
	defer os.RemoveAll(dir)
	for _, arch := range archs {
		for _, k := range kernels() {
			src, shared, ok, err := k.emit(arch)
			if !ok {
				continue
			}
			c := Compile{Kernel: k.name, Arch: arch, DynamicShared: shared}
			start := time.Now()
			if err != nil {
				c.Error = fmt.Sprintf("emit: %v", err)
			} else if err := compileOne(ctx, nvcc, dir, k.name, src, &c, timeout); err != nil {
				c.Error = err.Error()
			}
			c.Seconds = time.Since(start).Seconds()
			cs = append(cs, c)
		}
	}
	return cs
}

func compileOne(ctx context.Context, nvcc, dir, name, src string, c *Compile, timeout time.Duration) error {
	cu := filepath.Join(dir, name+".cu")
	if err := os.WriteFile(cu, []byte(src), 0o644); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	out, err := cudatool.Run(ctx, nvcc, dir, nil, "-std=c++17", "-O3", "-cubin", "-arch="+c.Arch, "-Xptxas", "-v", "-o", filepath.Join(dir, name+".cubin"), cu)
	if err != nil {
		if out.ExitCode > 0 {
			return fmt.Errorf("nvcc exit %d: %s", out.ExitCode, errorLines(append(out.Stdout, out.Stderr...), 3))
		}
		return fmt.Errorf("nvcc: %w", err)
	}
	rep, err := ptxasinfo.Parse(bytes.NewReader(append(out.Stdout, out.Stderr...)))
	if err != nil {
		return fmt.Errorf("ptxas report: %w", err)
	}
	for _, f := range rep.Functions() {
		c.SpillBytes += f.SpillStores + f.SpillLoads
		if !f.Entry {
			continue
		}
		c.Entries++
		c.MaxRegisters = max(c.MaxRegisters, f.Registers)
		c.StaticShared = max(c.StaticShared, f.SharedMem)
	}
	return nil
}

// errorLines returns the first n lines of compiler output that report an
// error, and how many more there were.
func errorLines(b []byte, n int) string {
	var lines []string
	for line := range strings.Lines(string(b)) {
		if strings.Contains(line, "error") {
			lines = append(lines, strings.TrimSpace(line))
		}
	}
	if len(lines) > n {
		return fmt.Sprintf("%s (and %d more)", strings.Join(lines[:n], "; "), len(lines)-n)
	}
	return strings.Join(lines, "; ")
}
