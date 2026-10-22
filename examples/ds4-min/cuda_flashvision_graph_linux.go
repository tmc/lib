//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// flashVisionGraph borrows a dedicated encoder stream, cuBLAS handle, immutable
// BF16 weight buffers and scratch. Its owner must serialize calls and retain
// every resource through stream completion. It never borrows language scratch.
type flashVisionGraph struct {
	blas       cublas.Handle
	stream     *cuda.Stream
	kernels    map[string]driver.Function
	weights    map[string]cudamem.Slice[uint16]
	buffer     map[string]cudamem.Slice[float32]
	activation cudamem.Slice[uint16]
}

func (g *flashVisionGraph) encode(gridH, gridW int) (cudamem.Slice[float32], error) {
	var zero cudamem.Slice[float32]
	if g == nil || g.stream == nil || gridH < 1 || gridW < 1 || gridH > 16384 || gridW > 16384 {
		return zero, fmt.Errorf("vision encoder: invalid grid")
	}
	rows, alignedRows := gridH*gridW, ((gridH+2)/3)*((gridW+2)/3)
	// Require every stage's storage before submitting any work.
	for name, count := range map[string]int{
		"patch": rows * 588, "a": rows * 1024, "b": rows * 1024,
		"qkv": rows * 3072, "q": rows * 1024, "k": rows * 1024, "v": rows * 1024,
		"attn": rows * 1024, "mlp_w1": rows * 5632, "mlp_mid": rows * 2816,
		"align_in": alignedRows * 9216, "align_a": alignedRows * 4096, "align_b": alignedRows * 4096,
	} {
		if g.buffer[name].Len() < count {
			return zero, fmt.Errorf("vision encoder: %s buffer too small", name)
		}
	}
	for _, name := range []string{"matvec_bf16", "f32_to_bf16", "round_bf16", "bias", "rms", "qkv_rope", "attention", "swiglu", "add_residual", "aligner_reorder", "gelu_bias"} {
		if g.kernels[name] == 0 {
			return zero, fmt.Errorf("vision encoder: missing %s kernel", name)
		}
	}
	// Required weights are checked by shape before any launch; the sidecar binder
	// independently owns the external GGUF name/type/dimension contract.
	need := map[string]int{"vision.patch_embed.proj.weight": 588 * 1024, "vision.patch_embed.proj.bias": 1024,
		"vision.norm.weight": 1024, "aligner.w1.weight": 9216 * 4096, "aligner.w1.bias": 4096,
		"aligner.w2.weight": 4096 * 4096, "aligner.w2.bias": 4096}
	for layer := 0; layer < 32; layer++ {
		p := fmt.Sprintf("vision.blocks.%d.", layer)
		for name, count := range map[string]int{"norm1.weight": 1024, "attn.wqkv.weight": 1024 * 3072, "attn.wqkv.bias": 3072,
			"attn.wo.weight": 1024 * 1024, "attn.wo.bias": 1024, "norm2.weight": 1024, "mlp.w1.weight": 1024 * 5632, "mlp.w2.weight": 2816 * 1024} {
			need[p+name] = count
		}
	}
	for name, count := range need {
		if g.weights[name].Len() != count {
			return zero, fmt.Errorf("vision encoder: %s weight size", name)
		}
	}
	if rows > 8 && g.activation.Len() < max(rows*2816, alignedRows*9216) {
		return zero, fmt.Errorf("vision encoder: activation scratch too small")
	}
	launch := func(name string, grid [3]uint32, block uint32, args ...any) error {
		if err := cudalaunch.Kernel(g.kernels[name]).Grid(grid[0], grid[1], grid[2]).Block(block, 1, 1).On(g.stream).Launch(args...); err != nil {
			return fmt.Errorf("vision %s: %w", name, err)
		}
		return nil
	}
	flat := func(name string, count int, args ...any) error {
		return launch(name, [3]uint32{uint32((count + 255) / 256), 1, 1}, 256, args...)
	}
	round := func(x cudamem.Slice[float32], count int) error { return flat("round_bf16", count, x, uint64(count)) }
	mat := func(dst, weight, src string, in, out, n int) error {
		return flashVisionMatmul(g.blas, g.stream, g.kernels["matvec_bf16"], g.kernels["f32_to_bf16"], g.weights[weight], g.buffer[src], g.buffer[dst], g.activation, in, out, n)
	}
	bias := func(dst, weight string, residual cudamem.Slice[float32], count, width int) error {
		return flat("bias", count, g.buffer[dst], g.weights[weight], residual, uint64(count), uint32(width))
	}
	norm := func(dst, src, weight string) error {
		return launch("rms", [3]uint32{uint32(rows), 1, 1}, 256, g.buffer[dst], g.buffer[src], g.weights[weight], uint32(1024), float32(1e-6))
	}
	if err := mat("a", "vision.patch_embed.proj.weight", "patch", 588, 1024, rows); err != nil {
		return zero, err
	}
	if err := bias("a", "vision.patch_embed.proj.bias", zero, rows*1024, 1024); err != nil {
		return zero, err
	}
	if err := round(g.buffer["a"], rows*1024); err != nil {
		return zero, err
	}
	cur, tmp := "a", "b"
	for layer := 0; layer < 32; layer++ {
		p := fmt.Sprintf("vision.blocks.%d.", layer)
		if err := norm(tmp, cur, p+"norm1.weight"); err != nil {
			return zero, err
		}
		if err := round(g.buffer[tmp], rows*1024); err != nil {
			return zero, err
		}
		if err := mat("qkv", p+"attn.wqkv.weight", tmp, 1024, 3072, rows); err != nil {
			return zero, err
		}
		if err := launch("qkv_rope", [3]uint32{uint32(rows), 16, 1}, 32, g.buffer["q"], g.buffer["k"], g.buffer["v"], g.buffer["qkv"], g.weights[p+"attn.wqkv.bias"], uint32(rows), uint32(gridW)); err != nil {
			return zero, err
		}
		if err := launch("attention", [3]uint32{uint32(rows), 16, 1}, 32, g.buffer["attn"], g.buffer["q"], g.buffer["k"], g.buffer["v"], uint32(rows)); err != nil {
			return zero, err
		}
		if err := round(g.buffer["attn"], rows*1024); err != nil {
			return zero, err
		}
		if err := mat(tmp, p+"attn.wo.weight", "attn", 1024, 1024, rows); err != nil {
			return zero, err
		}
		if err := bias(tmp, p+"attn.wo.bias", g.buffer[cur], rows*1024, 1024); err != nil {
			return zero, err
		}
		if err := round(g.buffer[tmp], rows*1024); err != nil {
			return zero, err
		}
		cur, tmp = tmp, cur
		if err := norm(tmp, cur, p+"norm2.weight"); err != nil {
			return zero, err
		}
		if err := round(g.buffer[tmp], rows*1024); err != nil {
			return zero, err
		}
		if err := mat("mlp_w1", p+"mlp.w1.weight", tmp, 1024, 5632, rows); err != nil {
			return zero, err
		}
		if err := round(g.buffer["mlp_w1"], rows*5632); err != nil {
			return zero, err
		}
		if err := flat("swiglu", rows*2816, g.buffer["mlp_mid"], g.buffer["mlp_w1"], uint64(rows*2816), uint32(2816)); err != nil {
			return zero, err
		}
		if err := mat(tmp, p+"mlp.w2.weight", "mlp_mid", 2816, 1024, rows); err != nil {
			return zero, err
		}
		if err := round(g.buffer[tmp], rows*1024); err != nil {
			return zero, err
		}
		if err := flat("add_residual", rows*1024, g.buffer[tmp], g.buffer[cur], uint64(rows*1024)); err != nil {
			return zero, err
		}
		cur, tmp = tmp, cur
	}
	if err := norm(tmp, cur, "vision.norm.weight"); err != nil {
		return zero, err
	}
	if err := round(g.buffer[tmp], rows*1024); err != nil {
		return zero, err
	}
	if err := launch("aligner_reorder", [3]uint32{36, uint32(alignedRows), 1}, 256, g.buffer["align_in"], g.buffer[tmp], uint32(gridH), uint32(gridW), uint32(alignedRows)); err != nil {
		return zero, err
	}
	if err := mat("align_a", "aligner.w1.weight", "align_in", 9216, 4096, alignedRows); err != nil {
		return zero, err
	}
	if err := flat("gelu_bias", alignedRows*4096, g.buffer["align_b"], g.buffer["align_a"], g.weights["aligner.w1.bias"], uint64(alignedRows*4096), uint32(4096)); err != nil {
		return zero, err
	}
	if err := mat("align_a", "aligner.w2.weight", "align_b", 4096, 4096, alignedRows); err != nil {
		return zero, err
	}
	if err := bias("align_a", "aligner.w2.bias", zero, alignedRows*4096, 4096); err != nil {
		return zero, err
	}
	if err := round(g.buffer["align_a"], alignedRows*4096); err != nil {
		return zero, err
	}
	return g.buffer["align_a"].Slice(0, alignedRows*4096), nil
}
