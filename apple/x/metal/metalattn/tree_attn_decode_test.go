//go:build darwin

package metalattn

import (
	"math"
	"strconv"
	"testing"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// referenceTreeAttention computes bitmask tree attention the straightforward
// way. It mirrors the kernel's truncation of trees wider than a row bitmask.
func referenceTreeAttention(q, kCache, vCache []float32, masks []uint64, numNodes, numHeads, headDim int, scale float32) []float32 {
	n := numNodes
	if n > TreeAttentionMaxNodes {
		n = TreeAttentionMaxNodes
	}
	out := make([]float32, numNodes*numHeads*headDim)

	for node := 0; node < numNodes; node++ {
		mask := masks[node]
		for h := 0; h < numHeads; h++ {
			qOff := (node*numHeads + h) * headDim

			scores := make([]float64, n)
			maxScore := math.Inf(-1)
			for t := 0; t < n; t++ {
				if mask>>uint(t)&1 == 0 {
					continue
				}
				kOff := (t*numHeads + h) * headDim
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(q[qOff+d]) * float64(kCache[kOff+d])
				}
				dot *= float64(scale)
				scores[t] = dot
				if dot > maxScore {
					maxScore = dot
				}
			}

			var sum float64
			for t := 0; t < n; t++ {
				if mask>>uint(t)&1 == 0 {
					scores[t] = 0
					continue
				}
				scores[t] = math.Exp(scores[t] - maxScore)
				sum += scores[t]
			}

			// An empty ancestor row is all zeros.
			inv := 0.0
			if sum > 0 {
				inv = 1.0 / sum
			}
			for d := 0; d < headDim; d++ {
				var acc float64
				for t := 0; t < n; t++ {
					if mask>>uint(t)&1 == 0 {
						continue
					}
					vOff := (t*numHeads + h) * headDim
					acc += scores[t] * float64(vCache[vOff+d])
				}
				out[qOff+d] = float32(acc * inv)
			}
		}
	}
	return out
}

// chainMasks builds ancestor bitmasks for a linear chain: node i descends from
// every node before it. Node 0 additionally gets an empty row so the
// empty-ancestor path is always exercised.
func chainMasks(numNodes int) []uint64 {
	masks := make([]uint64, numNodes)
	for i := range masks {
		if i == 0 {
			masks[i] = 0
			continue
		}
		var m uint64
		for j := 0; j <= i && j < TreeAttentionMaxNodes; j++ {
			m |= 1 << uint(j)
		}
		masks[i] = m
	}
	return masks
}

func TestTreeFlashAttentionMatchesReference(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}

	lib, err := dev.CompileLibrary(GenerateTreeAttentionMSL())
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	pso, err := dev.NewComputePipeline(lib, "TreeFlashAttention")
	if err != nil {
		t.Fatalf("NewComputePipeline: %v", err)
	}

	const numHeads = 4
	const headDim = 32
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// 63/64/65 straddle the bitmask width; 100 must be truncated, not overflow
	// the threadgroup score array.
	for _, numNodes := range []int{1, 8, 63, 64, 65, 100} {
		// Every dispatch width must give the same answer: the kernel reads its
		// own threadgroup size rather than assuming one.
		for _, tgSize := range []uint{32, 64, 128, 256} {
			name := "nodes=" + strconv.Itoa(numNodes) + "/tg=" + strconv.Itoa(int(tgSize))
			t.Run(name, func(t *testing.T) {
				qBuf, _ := metalmem.Alloc[float32](dev, numNodes*numHeads*headDim)
				kBuf, _ := metalmem.Alloc[float32](dev, numNodes*numHeads*headDim)
				vBuf, _ := metalmem.Alloc[float32](dev, numNodes*numHeads*headDim)
				outBuf, _ := metalmem.Alloc[float32](dev, numNodes*numHeads*headDim)
				maskBuf, _ := metalmem.Alloc[uint64](dev, numNodes)

				fillPseudoRandom(qBuf.HostSlice(), 21)
				fillPseudoRandom(kBuf.HostSlice(), 23)
				fillPseudoRandom(vBuf.HostSlice(), 29)
				copy(maskBuf.HostSlice(), chainMasks(numNodes))
				for i := range outBuf.HostSlice() {
					outBuf.HostSlice()[i] = float32(math.NaN())
				}

				nodesVal := int32(numNodes)
				headsVal := int32(numHeads)
				headDimVal := int32(headDim)
				scaleVal := scale
				newScalar := func(p unsafe.Pointer) apple.MTLBuffer {
					return dev.Raw().NewBufferWithBytesLengthOptions(p, 4, apple.MTLResourceStorageModeShared)
				}

				cb := dev.Queue().CommandBuffer()
				enc := cb.ComputeCommandEncoder()
				enc.SetComputePipelineState(pso)
				enc.SetBufferWithOffsetAtIndex(qBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(kBuf.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(vBuf.Buffer(), 0, 2)
				enc.SetBufferWithOffsetAtIndex(outBuf.Buffer(), 0, 3)
				enc.SetBufferWithOffsetAtIndex(maskBuf.Buffer(), 0, 4)
				enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&nodesVal)), 0, 5)
				enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&headsVal)), 0, 6)
				enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&headDimVal)), 0, 7)
				enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&scaleVal)), 0, 8)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					apple.MTLSize{Width: uint(numNodes), Height: uint(numHeads), Depth: 1},
					apple.MTLSize{Width: tgSize, Height: 1, Depth: 1},
				)
				enc.EndEncoding()
				cb.Commit()
				cb.WaitUntilCompleted()

				got := outBuf.HostSlice()
				want := referenceTreeAttention(qBuf.HostSlice(), kBuf.HostSlice(), vBuf.HostSlice(),
					maskBuf.HostSlice(), numNodes, numHeads, headDim, scale)

				const tol = 1e-5
				for i := range want {
					if math.IsNaN(float64(got[i])) {
						t.Fatalf("out[%d] not written (NaN)", i)
					}
					if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
						t.Fatalf("out[%d] (node %d, head %d, dim %d) = %v, want %v (diff %v)",
							i, i/(numHeads*headDim), (i/headDim)%numHeads, i%headDim,
							got[i], want[i], diff)
					}
				}
			})
		}
	}
}

// TestTreeFlashAttentionEmptyRowZeroed pins the empty-ancestor contract: a node
// with no ancestors must emit zeros, not an unnormalized sum.
func TestTreeFlashAttentionEmptyRowZeroed(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}

	lib, err := dev.CompileLibrary(GenerateTreeAttentionMSL())
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	pso, err := dev.NewComputePipeline(lib, "TreeFlashAttention")
	if err != nil {
		t.Fatalf("NewComputePipeline: %v", err)
	}

	const numNodes, numHeads, headDim = 4, 2, 32

	qBuf, _ := metalmem.Alloc[float32](dev, numNodes*numHeads*headDim)
	kBuf, _ := metalmem.Alloc[float32](dev, numNodes*numHeads*headDim)
	vBuf, _ := metalmem.Alloc[float32](dev, numNodes*numHeads*headDim)
	outBuf, _ := metalmem.Alloc[float32](dev, numNodes*numHeads*headDim)
	maskBuf, _ := metalmem.Alloc[uint64](dev, numNodes)

	fillPseudoRandom(qBuf.HostSlice(), 31)
	fillPseudoRandom(kBuf.HostSlice(), 37)
	fillPseudoRandom(vBuf.HostSlice(), 41)
	for i := range maskBuf.HostSlice() {
		maskBuf.HostSlice()[i] = 0 // no node has any ancestor
	}
	for i := range outBuf.HostSlice() {
		outBuf.HostSlice()[i] = 12345.0
	}

	nodesVal, headsVal, headDimVal := int32(numNodes), int32(numHeads), int32(headDim)
	scaleVal := float32(1.0 / math.Sqrt(float64(headDim)))
	newScalar := func(p unsafe.Pointer) apple.MTLBuffer {
		return dev.Raw().NewBufferWithBytesLengthOptions(p, 4, apple.MTLResourceStorageModeShared)
	}

	cb := dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(qBuf.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(kBuf.Buffer(), 0, 1)
	enc.SetBufferWithOffsetAtIndex(vBuf.Buffer(), 0, 2)
	enc.SetBufferWithOffsetAtIndex(outBuf.Buffer(), 0, 3)
	enc.SetBufferWithOffsetAtIndex(maskBuf.Buffer(), 0, 4)
	enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&nodesVal)), 0, 5)
	enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&headsVal)), 0, 6)
	enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&headDimVal)), 0, 7)
	enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&scaleVal)), 0, 8)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: numNodes, Height: numHeads, Depth: 1},
		apple.MTLSize{Width: 128, Height: 1, Depth: 1},
	)
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()

	for i, v := range outBuf.HostSlice() {
		if v != 0 {
			t.Fatalf("out[%d] = %v, want 0 for a node with no ancestors", i, v)
		}
	}
}

// TestTreeAttentionShadersConcatenate guards the shared reduction helpers: the
// package's shader strings are frequently compiled into one library together.
func TestTreeAttentionShadersConcatenate(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	src := CausalFlashAttentionMSL + "\n" + TreeAttentionMSL
	if _, err := dev.CompileLibrary(src); err != nil {
		t.Fatalf("concatenated metalattn shaders failed to compile: %v", err)
	}
}
