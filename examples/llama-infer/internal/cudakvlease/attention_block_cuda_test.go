package cudakvlease

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudalayer"
)

func TestCUDAPackedDecoderLayerComplete(t *testing.T) {
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") != "1" {
		t.Skip("set LIB_CUDA_HARDWARE_TEST=1 on a CUDA host")
	}
	for _, lengths := range [][]int{{3}, {3, 5}, {3, 5, 2, 6}} {
		t.Run(fmt.Sprintf("B%d", len(lengths)), func(t *testing.T) {
			testCUDAPackedAttentionBlock(t, lengths)
		})
	}
}

func testCUDAPackedAttentionBlock(t *testing.T, lengths []int) {
	const blockSize, qHeads, kvHeads, headDim, intermediateDim = 2, 2, 1, 4, 11
	const theta, epsilon = float32(10000), float32(1e-5)
	dim, kvDim := qHeads*headDim, kvHeads*headDim
	pages, tokens, maxLen := 0, 0, 0
	for _, length := range lengths {
		pages += (length + blockSize - 1) / blockSize
		tokens += length
		maxLen = max(maxLen, length)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	adapter, err := NewCUDA(dev, Config{NumLayers: 1, KVHeads: kvHeads, HeadDim: headDim, BlockSize: blockSize, TotalPages: pages, DType: "float32", Domain: "cuda:0"})
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := NewPackedMetadata(adapter, dev, len(lengths), tokens, (maxLen+blockSize-1)/blockSize)
	if err != nil {
		t.Fatal(err)
	}
	requests := make([]PackedRequest, len(lengths))
	for row, length := range lengths {
		positions, causal := make([]int, length), make([]int, length)
		for i := range positions {
			positions[i], causal[i] = i, i+1
		}
		requests[row] = PackedRequest{BatchRequest: BatchRequest{Key: fmt.Sprintf("row-%d", row), Generation: uint64(row + 1), Length: length, Position: length - 1}, Positions: positions, CausalLengths: causal}
	}
	batch, _, err := metadata.Admit(t.Context(), requests, blockSize, &fakeCompletion{})
	if err != nil {
		t.Fatal(err)
	}
	var admitted []Allocation
	for _, request := range requests {
		allocations, err := batch.Allocations(request.Key)
		if err != nil {
			t.Fatal(err)
		}
		admitted = append(admitted, allocations...)
	}

	input := values(tokens*dim, 17, 8, 19)
	inputNormWeight := values(dim, 11, 5, 23)
	qWeight := values(dim*dim, 13, 6, 29)
	kWeight := values(kvDim*dim, 7, 3, 31)
	vWeight := values(kvDim*dim, 9, 4, 37)
	projection := values(dim*dim, 15, 7, 41)
	outputNormWeight := values(dim, 10, 4, 17)
	gateWeight := values(intermediateDim*dim, 17, 8, 43)
	upWeight := values(intermediateDim*dim, 19, 9, 47)
	downWeight := values(dim*intermediateDim, 23, 11, 53)
	dinput := cudaAttentionSlice(t, dev, input)
	dinputNorm := cudaAttentionSlice(t, dev, make([]float32, tokens*dim))
	dinputNormWeight := cudaAttentionSlice(t, dev, inputNormWeight)
	dqWeight := cudaAttentionSlice(t, dev, qWeight)
	dkWeight := cudaAttentionSlice(t, dev, kWeight)
	dvWeight := cudaAttentionSlice(t, dev, vWeight)
	dq := cudaAttentionSlice(t, dev, make([]float32, tokens*dim))
	dk := cudaAttentionSlice(t, dev, make([]float32, tokens*kvDim))
	dv := cudaAttentionSlice(t, dev, make([]float32, tokens*kvDim))
	dqRoPE := cudaAttentionSlice(t, dev, make([]float32, tokens*dim))
	dattention := cudaAttentionSlice(t, dev, make([]float32, tokens*dim))
	dprojection := cudaAttentionSlice(t, dev, projection)
	doutputNormWeight := cudaAttentionSlice(t, dev, outputNormWeight)
	doutput := cudaAttentionSlice(t, dev, make([]float32, tokens*dim))
	doutputNorm := cudaAttentionSlice(t, dev, make([]float32, tokens*dim))
	dgateWeight := cudaAttentionSlice(t, dev, gateWeight)
	dupWeight := cudaAttentionSlice(t, dev, upWeight)
	ddownWeight := cudaAttentionSlice(t, dev, downWeight)
	dgate := cudaAttentionSlice(t, dev, make([]float32, tokens*intermediateDim))
	dup := cudaAttentionSlice(t, dev, make([]float32, tokens*intermediateDim))
	dactivation := cudaAttentionSlice(t, dev, make([]float32, tokens*intermediateDim))
	dlayerOutput := cudaAttentionSlice(t, dev, make([]float32, tokens*dim))
	for _, free := range []func() error{dinput.Free, dinputNorm.Free, dinputNormWeight.Free, dqWeight.Free, dkWeight.Free, dvWeight.Free, dq.Free, dk.Free, dv.Free, dqRoPE.Free, dattention.Free, dprojection.Free, doutputNormWeight.Free, doutput.Free, doutputNorm.Free, dgateWeight.Free, dupWeight.Free, ddownWeight.Free, dgate.Free, dup.Free, dactivation.Free, dlayerOutput.Free} {
		defer free()
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	attentionCode, err := cudaattn.CompilePaged(major, minor)
	if err != nil {
		t.Fatal(err)
	}
	attentionModule, err := attentionCode.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer attentionModule.Unload()
	layerCode, err := cudalayer.Compile(major, minor)
	if err != nil {
		t.Fatal(err)
	}
	layerModule, err := layerCode.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer layerModule.Unload()
	qkvKernel, err := layerModule.GetFunction("RMSNormQKV")
	if err != nil {
		t.Fatal(err)
	}
	ropeKernel, err := attentionModule.GetFunction("PackedPagedRoPEStoreKV")
	if err != nil {
		t.Fatal(err)
	}
	attentionKernel, err := attentionModule.GetFunction("BatchedPagedAttentionCausal")
	if err != nil {
		t.Fatal(err)
	}
	projectionKernel, err := layerModule.GetFunction("ProjectionResidualRMSNorm")
	if err != nil {
		t.Fatal(err)
	}
	gateUpKernel, err := layerModule.GetFunction("GateUpSiLU")
	if err != nil {
		t.Fatal(err)
	}
	downKernel, err := layerModule.GetFunction("DownResidual")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	op := PackedDecoderLayer{PackedAttentionBlock: PackedAttentionBlock{DecoderLayer: DecoderLayer{
		Attention:      CausalAttention{Kernel: attentionKernel, Layer: 0, QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, MaxSeqLen: maxLen, Threads: 32, Scale: float32(1 / math.Sqrt(headDim))},
		ProjectionNorm: projectionKernel, Dim: dim, Threads: 32, Epsilon: epsilon,
	}, RMSNormQKV: qkvKernel, RoPEStore: ropeKernel, RoPETheta: theta}, GateUpSiLU: gateUpKernel, DownResidual: downKernel, IntermediateDim: intermediateDim}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := batch.RunPackedDecoderLayer(canceled, stream, op, dinput, dinputNorm, dinputNormWeight, dqWeight, dkWeight, dvWeight, dq, dk, dv, dqRoPE, dattention, dprojection, doutputNormWeight, doutput, doutputNorm, dgateWeight, dupWeight, ddownWeight, dgate, dup, dactivation, dlayerOutput); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-submit cancellation = %v", err)
	}
	status, err := batch.RunPackedDecoderLayer(t.Context(), stream, op, dinput, dinputNorm, dinputNormWeight, dqWeight, dkWeight, dvWeight, dq, dk, dv, dqRoPE, dattention, dprojection, doutputNormWeight, doutput, doutputNorm, dgateWeight, dupWeight, ddownWeight, dgate, dup, dactivation, dlayerOutput)
	if err != nil {
		t.Fatal(err)
	}
	for row := range lengths {
		if status[row] != 0 {
			t.Fatalf("status[%d] = %d", row, status[row])
		}
	}
	wantOutput, wantNorm := attentionBlockReference(input, inputNormWeight, qWeight, kWeight, vWeight, projection, outputNormWeight, lengths, qHeads, kvHeads, headDim, theta, epsilon)
	wantLayerOutput := decoderMLPReference(wantOutput, wantNorm, gateWeight, upWeight, downWeight, dim, intermediateDim)
	checkLayerOutput(t, dev, doutput, wantOutput, 2e-5)
	checkLayerOutput(t, dev, doutputNorm, wantNorm, 2e-5)
	checkLayerOutput(t, dev, dlayerOutput, wantLayerOutput, 3e-5)

	firstID, firstGeneration := admitted[0].ID()
	firstUse, err := adapter.Bind(firstID, firstGeneration)
	if err != nil {
		t.Fatal(err)
	}
	firstPage, err := firstUse.Page(0)
	if err != nil {
		t.Fatal(err)
	}
	beforeK, beforeV := make([]float32, blockSize*kvDim), make([]float32, blockSize*kvDim)
	if err := dev.WithContext(func() error {
		if err := firstPage.K.CopyToHost(beforeK); err != nil {
			return err
		}
		return firstPage.V.CopyToHost(beforeV)
	}); err != nil {
		t.Fatal(err)
	}
	mutatedInput := append([]float32(nil), input...)
	for i := 0; i < dim; i++ {
		mutatedInput[i] += 10
	}
	if err := dev.WithContext(func() error { return dinput.CopyFromHost(mutatedInput) }); err != nil {
		t.Fatal(err)
	}
	device, err := batch.Device()
	if err != nil {
		t.Fatal(err)
	}
	generations := make([]uint64, tokens)
	if err := dev.WithContext(func() error {
		if err := device.TokenGenerations.CopyToHost(generations); err != nil {
			return err
		}
		generations[0]++
		return device.TokenGenerations.CopyFromHost(generations)
	}); err != nil {
		t.Fatal(err)
	}
	status, err = batch.RunPackedDecoderLayer(t.Context(), stream, op, dinput, dinputNorm, dinputNormWeight, dqWeight, dkWeight, dvWeight, dq, dk, dv, dqRoPE, dattention, dprojection, doutputNormWeight, doutput, doutputNorm, dgateWeight, dupWeight, ddownWeight, dgate, dup, dactivation, dlayerOutput)
	if err != nil {
		t.Fatal(err)
	}
	if status[0] != 1 {
		t.Fatalf("fault status[0] = %d", status[0])
	}
	afterK, afterV := make([]float32, len(beforeK)), make([]float32, len(beforeV))
	if err := dev.WithContext(func() error {
		if err := firstPage.K.CopyToHost(afterK); err != nil {
			return err
		}
		return firstPage.V.CopyToHost(afterV)
	}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < kvDim; i++ {
		if afterK[i] != beforeK[i] || afterV[i] != beforeV[i] {
			t.Fatalf("stale token mutated KV coordinate %d", i)
		}
	}
	if err := firstUse.Complete(t.Context(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, tokens*dim)
	if err := dev.WithContext(func() error { return dlayerOutput.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}
	firstOther := lengths[0]
	for i := 0; i < firstOther*dim; i++ {
		if !math.IsNaN(float64(got[i])) {
			t.Fatalf("fault output[%d] = %g, want NaN", i, got[i])
		}
	}
	for i := firstOther * dim; i < len(got); i++ {
		if diff := math.Abs(float64(got[i] - wantLayerOutput[i])); diff > 3e-5 {
			t.Fatalf("isolated output[%d] = %g, want %g", i, got[i], wantLayerOutput[i])
		}
	}
	if err := batch.Complete(t.Context(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	for i := len(admitted) - 1; i >= 0; i-- {
		if err := adapter.Retire(t.Context(), admitted[i], &fakeCompletion{}); err != nil {
			t.Fatal(err)
		}
	}
	if err := metadata.Close(); err != nil {
		t.Fatal(err)
	}
	if err := adapter.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("receipt batch=%d tokens=%d coordinates=%d tolerance=3e-5 complete_layer=true shared_weights=true heterogeneous_positions=true row_fault_isolated=true one_terminal_sync=true pins_outer_owned=true", len(lengths), tokens, len(wantLayerOutput))
}

func decoderMLPReference(residual, normalized, gateWeight, upWeight, downWeight []float32, dim, intermediate int) []float32 {
	tokens := len(residual) / dim
	output := make([]float32, len(residual))
	for token := 0; token < tokens; token++ {
		activation := make([]float32, intermediate)
		for out := 0; out < intermediate; out++ {
			var gate, up float32
			for in := 0; in < dim; in++ {
				x := normalized[token*dim+in]
				gate += x * gateWeight[out*dim+in]
				up += x * upWeight[out*dim+in]
			}
			activation[out] = gate / (1 + float32(math.Exp(float64(-gate)))) * up
		}
		for out := 0; out < dim; out++ {
			sum := residual[token*dim+out]
			for in := 0; in < intermediate; in++ {
				sum += activation[in] * downWeight[out*intermediate+in]
			}
			output[token*dim+out] = sum
		}
	}
	return output
}

func values(n, period, offset, divisor int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(i%period-offset) / float32(divisor)
	}
	return v
}

func attentionBlockReference(input, inputNormWeight, qWeight, kWeight, vWeight, projection, outputNormWeight []float32, lengths []int, qHeads, kvHeads, headDim int, theta, epsilon float32) ([]float32, []float32) {
	dim, kvDim := qHeads*headDim, kvHeads*headDim
	tokens := len(input) / dim
	q, k, v := make([]float32, tokens*dim), make([]float32, tokens*kvDim), make([]float32, tokens*kvDim)
	for token := 0; token < tokens; token++ {
		var squares float64
		for i := 0; i < dim; i++ {
			x := input[token*dim+i]
			squares += float64(x * x)
		}
		scale := float32(1 / math.Sqrt(squares/float64(dim)+float64(epsilon)))
		n := make([]float32, dim)
		for i := range n {
			n[i] = input[token*dim+i] * scale * inputNormWeight[i]
		}
		for out := 0; out < dim; out++ {
			for in := 0; in < dim; in++ {
				q[token*dim+out] += n[in] * qWeight[out*dim+in]
			}
		}
		for out := 0; out < kvDim; out++ {
			for in := 0; in < dim; in++ {
				k[token*kvDim+out] += n[in] * kWeight[out*dim+in]
				v[token*kvDim+out] += n[in] * vWeight[out*dim+in]
			}
		}
	}
	attention := make([]float32, tokens*dim)
	base := 0
	for _, length := range lengths {
		for pos := 0; pos < length; pos++ {
			applyReferenceRoPE(q[(base+pos)*dim:(base+pos+1)*dim], pos, qHeads, headDim, theta)
			applyReferenceRoPE(k[(base+pos)*kvDim:(base+pos+1)*kvDim], pos, kvHeads, headDim, theta)
			for head := 0; head < qHeads; head++ {
				kvHead := head / (qHeads / kvHeads)
				scores := make([]float64, pos+1)
				maximum := math.Inf(-1)
				for p := range scores {
					for d := 0; d < headDim; d++ {
						scores[p] += float64(q[(base+pos)*dim+head*headDim+d] * k[(base+p)*kvDim+kvHead*headDim+d])
					}
					scores[p] /= math.Sqrt(float64(headDim))
					maximum = max(maximum, scores[p])
				}
				var sum float64
				for p := range scores {
					scores[p] = math.Exp(scores[p] - maximum)
					sum += scores[p]
				}
				for d := 0; d < headDim; d++ {
					for p := range scores {
						attention[(base+pos)*dim+head*headDim+d] += float32(scores[p]/sum) * v[(base+p)*kvDim+kvHead*headDim+d]
					}
				}
			}
		}
		base += length
	}
	return projectionNormReference(attention, input, projection, outputNormWeight, dim, epsilon)
}

func applyReferenceRoPE(x []float32, position, heads, headDim int, theta float32) {
	half := headDim / 2
	for head := 0; head < heads; head++ {
		copyHead := append([]float32(nil), x[head*headDim:(head+1)*headDim]...)
		for d := 0; d < headDim; d++ {
			pair := d + half
			if d >= half {
				pair = d - half
			}
			freq := d
			if d >= half {
				freq = pair
			}
			angle := float64(position) / math.Pow(float64(theta), float64(2*freq)/float64(headDim))
			a, b := copyHead[d], copyHead[pair]
			if d < half {
				x[head*headDim+d] = a*float32(math.Cos(angle)) - b*float32(math.Sin(angle))
			} else {
				x[head*headDim+d] = b*float32(math.Sin(angle)) + a*float32(math.Cos(angle))
			}
		}
	}
}
