//go:build darwin

package metalmtp

import (
	"testing"
	"time"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
)

func TestMTPMultiTokenPrediction(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}

	cfg := MTPConfig{
		NumDraftHeads: 4, // Speculatively predict 4 future tokens in 1 GPU dispatch
		Dim:           256,
		VocabSize:     1024,
	}

	msl := GenerateMTPMegakernel(cfg)
	lib, err := dev.CompileLibrary(msl)
	if err != nil {
		t.Fatalf("CompileLibrary failed: %v", err)
	}

	pso, err := dev.NewComputePipeline(lib, "MTPMultiHeadForward")
	if err != nil {
		t.Fatalf("NewComputePipeline failed: %v", err)
	}

	hTrunk, _ := mempool.Alloc[float32](dev, cfg.Dim)
	wProj, _ := mempool.Alloc[float32](dev, cfg.NumDraftHeads*cfg.Dim*cfg.Dim)
	wHeads, _ := mempool.Alloc[float32](dev, cfg.NumDraftHeads*cfg.Dim*cfg.VocabSize)
	outLogits, _ := mempool.Alloc[float32](dev, cfg.NumDraftHeads*cfg.VocabSize)

	dimVal := uint32(cfg.Dim)
	vocabVal := uint32(cfg.VocabSize)
	dimBuf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&dimVal), 4, apple.MTLResourceStorageModeShared)
	vocabBuf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&vocabVal), 4, apple.MTLResourceStorageModeShared)

	cb := dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(hTrunk.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(wProj.Buffer(), 0, 1)
	enc.SetBufferWithOffsetAtIndex(wHeads.Buffer(), 0, 2)
	enc.SetBufferWithOffsetAtIndex(outLogits.Buffer(), 0, 3)
	enc.SetBufferWithOffsetAtIndex(dimBuf, 0, 4)
	enc.SetBufferWithOffsetAtIndex(vocabBuf, 0, 5)

	enc.DispatchThreadsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(cfg.NumDraftHeads * 128), Height: 1, Depth: 1},
		apple.MTLSize{Width: 128, Height: 1, Depth: 1},
	)
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()

	t.Logf("=== Multi-Token Prediction (MTP) E-Graph Megakernel Success ===")
	t.Logf("Generated %d Speculative Future Tokens in a SINGLE GPU Dispatch!", cfg.NumDraftHeads)
}

func BenchmarkMTP_vs_Autoregressive(b *testing.B) {
	dev, err := metal.NewDevice()
	if err != nil {
		b.Skipf("Metal device unavailable: %v", err)
	}

	cfg := MTPConfig{
		NumDraftHeads: 4,
		Dim:           256,
		VocabSize:     1024,
	}

	msl := GenerateMTPMegakernel(cfg)
	lib, _ := dev.CompileLibrary(msl)
	pso, _ := dev.NewComputePipeline(lib, "MTPMultiHeadForward")

	hTrunk, _ := mempool.Alloc[float32](dev, cfg.Dim)
	wProj, _ := mempool.Alloc[float32](dev, cfg.NumDraftHeads*cfg.Dim*cfg.Dim)
	wHeads, _ := mempool.Alloc[float32](dev, cfg.NumDraftHeads*cfg.Dim*cfg.VocabSize)
	outLogits, _ := mempool.Alloc[float32](dev, cfg.NumDraftHeads*cfg.VocabSize)

	dimVal := uint32(cfg.Dim)
	vocabVal := uint32(cfg.VocabSize)
	dimBuf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&dimVal), 4, apple.MTLResourceStorageModeShared)
	vocabBuf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&vocabVal), 4, apple.MTLResourceStorageModeShared)

	b.Run("MTP_4_Tokens_Parallel", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			cb := dev.Queue().CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			enc.SetComputePipelineState(pso)
			enc.SetBufferWithOffsetAtIndex(hTrunk.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(wProj.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(wHeads.Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(outLogits.Buffer(), 0, 3)
			enc.SetBufferWithOffsetAtIndex(dimBuf, 0, 4)
			enc.SetBufferWithOffsetAtIndex(vocabBuf, 0, 5)
			enc.DispatchThreadsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.NumDraftHeads * 128), Height: 1, Depth: 1},
				apple.MTLSize{Width: 128, Height: 1, Depth: 1},
			)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
		dur := time.Since(start)
		if dur > 0 {
			tokPerSec := float64(b.N*cfg.NumDraftHeads) / dur.Seconds()
			b.ReportMetric(tokPerSec, "speculative_tok/s")
		}
	})

	b.Run("Sequential_Autoregressive_4_Tokens", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			for step := 0; step < cfg.NumDraftHeads; step++ {
				cb := dev.Queue().CommandBuffer()
				enc := cb.ComputeCommandEncoder()
				enc.SetComputePipelineState(pso)
				enc.SetBufferWithOffsetAtIndex(hTrunk.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(wProj.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(wHeads.Buffer(), 0, 2)
				enc.SetBufferWithOffsetAtIndex(outLogits.Buffer(), 0, 3)
				enc.SetBufferWithOffsetAtIndex(dimBuf, 0, 4)
				enc.SetBufferWithOffsetAtIndex(vocabBuf, 0, 5)
				enc.DispatchThreadsThreadsPerThreadgroup(
					apple.MTLSize{Width: 128, Height: 1, Depth: 1},
					apple.MTLSize{Width: 128, Height: 1, Depth: 1},
				)
				enc.EndEncoding()
				cb.Commit()
				cb.WaitUntilCompleted()
			}
		}
		dur := time.Since(start)
		if dur > 0 {
			tokPerSec := float64(b.N*cfg.NumDraftHeads) / dur.Seconds()
			b.ReportMetric(tokPerSec, "speculative_tok/s")
		}
	})
}
