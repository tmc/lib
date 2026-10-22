//go:build linux

package main

import (
	_ "embed"
	"fmt"
	"runtime"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

//go:embed cuda_flashvision.cu
var flashVisionCUDA string

// flashVisionMatmul issues the pinned encoder's two BF16 matrix routes. The
// caller owns the current context, handle, stream and all buffers until stream
// completion. Scratch holds converted activations for the large-row route.
// Nothing here changes the language model's foundation or session resources.
func flashVisionMatmul(blas cublas.Handle, stream *cuda.Stream,
	matvec, convert driver.Function, weights cudamem.Slice[uint16],
	x, out cudamem.Slice[float32], scratch cudamem.Slice[uint16],
	inDim, outDim, rows int) error {
	if inDim <= 0 || outDim <= 0 || rows <= 0 ||
		inDim > 1<<31-1 || outDim > 1<<31-1 || rows > 1<<31-1 {
		return fmt.Errorf("vision BF16 matmul: invalid shape %dx%dx%d", rows, outDim, inDim)
	}
	if uint64(weights.Len()) < uint64(inDim)*uint64(outDim) ||
		uint64(x.Len()) < uint64(inDim)*uint64(rows) ||
		uint64(out.Len()) < uint64(outDim)*uint64(rows) {
		return fmt.Errorf("vision BF16 matmul: buffer too small")
	}
	if stream == nil {
		return fmt.Errorf("vision BF16 matmul: nil stream")
	}
	if rows <= 8 {
		return cudalaunch.Kernel(matvec).Grid(uint32((outDim+7)/8), uint32(rows), 1).Block(256, 1, 1).On(stream).Launch(
			out, weights, x, uint32(inDim), uint32(outDim))
	}
	count := uint64(inDim) * uint64(rows)
	if (count+255)/256 > 1<<32-1 {
		return fmt.Errorf("vision BF16 matmul: activation launch too large")
	}
	if uint64(scratch.Len()) < count {
		return fmt.Errorf("vision BF16 matmul: activation scratch too small")
	}
	if err := cudalaunch.Kernel(convert).Grid(uint32((count+255)/256), 1, 1).Block(256, 1, 1).On(stream).Launch(scratch, x, count); err != nil {
		return fmt.Errorf("vision BF16 activation conversion: %w", err)
	}
	if err := stream.WithHandle(func(s driver.Stream) error { return blas.SetStream(driver.Stream(s)) }); err != nil {
		return fmt.Errorf("vision BF16 matmul stream: %w", err)
	}
	alpha, beta := float32(1), float32(0)
	var pin runtime.Pinner
	pin.Pin(&alpha)
	pin.Pin(&beta)
	defer pin.Unpin()
	err := blas.GemmEx(cublas.OpT, cublas.OpN, int32(outDim), int32(rows), int32(inDim), cublas.Float32Scalar(alpha), driver.DevicePtr(weights.Ptr()), cublas.R16BF, int32(inDim), driver.DevicePtr(scratch.Ptr()), cublas.R16BF, int32(inDim), cublas.Float32Scalar(beta), driver.DevicePtr(out.Ptr()), cublas.R32F, int32(outDim), cublas.Compute32F, cublas.GemmDefault)
	runtime.KeepAlive(alpha)
	runtime.KeepAlive(beta)
	if err != nil {
		return fmt.Errorf("vision BF16 matmul: %w", err)
	}
	return nil
}
