// Package cudagradcheck checks gradients computed by hand-written GPU
// kernels against float64 host oracles.
//
// [CheckGrad] compares a claimed gradient with central differences of a
// scalar host function, weighing absolute and relative error as
// torch.autograd.gradcheck does, and reports the worst element. [Compare]
// checks float32 kernel output against a float64 reference with a
// [Tolerance] sized for float32 or TF32 rounding by [Accumulated].
//
// The oracles compute forwards and backwards in float64:
//
//   - [LayerNorm] and [LayerNormBackward], with or without bias
//   - [Softmax] and [SoftmaxBackward], over rows
//   - [GELU] and [GELUBackward], the erf form
//   - [GeGLU] and [GeGLUBackward]
//   - [RoPE] and [RoPEBackward], rotate-half layout
//   - [Attention] and [AttentionBackward], with a [Mask] for full, causal,
//     sliding-window, ragged and fully masked cases
//
// Each backward is itself checked with CheckGrad against its forward, and
// its tests remove single terms from the backward to show the default
// tolerance catches the mistake.
//
// A typical kernel test runs the kernel, copies its gradient to the host,
// and checks it against the oracle's forward:
//
//	f := func(x []float64) float64 {
//		return cudagradcheck.Dot(cudagradcheck.GELU(x), dy)
//	}
//	if r, err := cudagradcheck.CheckGrad(f, x, kernelGrad, cudagradcheck.GradOptions{}); err != nil {
//		t.Fatal(err)
//	} else {
//		t.Logf("worst %v", r)
//	}
//
// For a float32 kernel gradient, pass its own tolerance, for example
// GradOptions{Tolerance: cudagradcheck.Accumulated(cudagradcheck.TF32Roundoff, k, 1)},
// or compare it with the oracle's backward using Compare.
package cudagradcheck
