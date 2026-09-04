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

func TestCUDAPackedDecoderLayer(t *testing.T) {
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") != "1" {
		t.Skip("set LIB_CUDA_HARDWARE_TEST=1 on a CUDA host")
	}
	for _, lengths := range [][]int{{3}, {3, 5}, {3, 5, 2, 6}} {
		t.Run(fmt.Sprintf("B%d", len(lengths)), func(t *testing.T) {
			testCUDAPackedDecoderLayer(t, lengths)
		})
	}
}

func testCUDAPackedDecoderLayer(t *testing.T, lengths []int) {
	const blockSize, qHeads, kvHeads, headDim = 2, 2, 1, 4
	dim := qHeads * headDim
	pages := 0
	for _, length := range lengths {
		pages += (length + blockSize - 1) / blockSize
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
	metadata, err := NewPackedMetadata(adapter, dev, len(lengths), 2*len(lengths), pages)
	if err != nil {
		t.Fatal(err)
	}
	requests := make([]PackedRequest, len(lengths))
	for row, length := range lengths {
		requests[row] = PackedRequest{
			BatchRequest: BatchRequest{Key: fmt.Sprintf("row-%d", row), Generation: uint64(row + 1), Length: length, Position: length - 1},
			Positions:    []int{0, length - 1}, CausalLengths: []int{1, length},
		}
	}
	batch, _, err := metadata.Admit(t.Context(), requests, blockSize, &fakeCompletion{})
	if err != nil {
		t.Fatal(err)
	}
	var admitted []Allocation
	for row, request := range requests {
		allocations, err := batch.Allocations(request.Key)
		if err != nil {
			t.Fatal(err)
		}
		admitted = append(admitted, allocations...)
		for block, allocation := range allocations {
			id, generation := allocation.ID()
			use, err := adapter.Bind(id, generation)
			if err != nil {
				t.Fatal(err)
			}
			page, err := use.Page(0)
			if err != nil {
				t.Fatal(err)
			}
			k := make([]float32, blockSize*kvHeads*headDim)
			v := make([]float32, len(k))
			for i := range k {
				token := block*blockSize + i/headDim
				k[i] = float32((row+1)*17+token*5+i%headDim) / 31
				v[i] = float32((row+1)*11-token*3+i%headDim) / 19
			}
			if err := dev.WithContext(func() error {
				if err := page.K.CopyFromHost(k); err != nil {
					return err
				}
				return page.V.CopyFromHost(v)
			}); err != nil {
				t.Fatal(err)
			}
			if err := use.Complete(t.Context(), &fakeCompletion{}); err != nil {
				t.Fatal(err)
			}
		}
	}
	tokens := 2 * len(lengths)
	q := make([]float32, tokens*dim)
	residual := make([]float32, len(q))
	weight := make([]float32, dim*dim)
	normWeight := make([]float32, dim)
	for i := range q {
		q[i] = float32(i%13-6) / 11
		residual[i] = float32(i%7-3) / 9
	}
	for i := range weight {
		weight[i] = float32(i%11-5) / 37
	}
	for i := range normWeight {
		normWeight[i] = 0.75 + float32(i)/32
	}
	dq := cudaAttentionSlice(t, dev, q)
	dattention := cudaAttentionSlice(t, dev, make([]float32, len(q)))
	dresidual := cudaAttentionSlice(t, dev, residual)
	dweight := cudaAttentionSlice(t, dev, weight)
	dnormWeight := cudaAttentionSlice(t, dev, normWeight)
	doutput := cudaAttentionSlice(t, dev, make([]float32, len(q)))
	dnormalized := cudaAttentionSlice(t, dev, make([]float32, len(q)))
	for _, free := range []func() error{dq.Free, dattention.Free, dresidual.Free, dweight.Free, dnormWeight.Free, doutput.Free, dnormalized.Free} {
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
	attentionKernel, err := attentionModule.GetFunction("BatchedPagedAttentionCausal")
	if err != nil {
		t.Fatal(err)
	}
	layerCode, err := cudalayer.Compile(major, minor)
	if err != nil {
		t.Fatal(err)
	}
	layerModule, err := layerCode.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer layerModule.Unload()
	layerKernel, err := layerModule.GetFunction("ProjectionResidualRMSNorm")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	op := DecoderLayer{
		Attention:      CausalAttention{Kernel: attentionKernel, Layer: 0, QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, MaxSeqLen: 6, Threads: 32, Scale: float32(1 / math.Sqrt(headDim))},
		ProjectionNorm: layerKernel, Dim: dim, Threads: 32, Epsilon: 1e-5,
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := batch.RunDecoderLayer(canceled, stream, op, dq, dattention, dresidual, dweight, dnormWeight, doutput, dnormalized); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-submit cancellation = %v", err)
	}
	status, err := batch.RunDecoderLayer(t.Context(), stream, op, dq, dattention, dresidual, dweight, dnormWeight, doutput, dnormalized)
	if err != nil {
		t.Fatal(err)
	}
	for row := range lengths {
		if status[row] != 0 {
			t.Fatalf("status[%d] = %d", row, status[row])
		}
	}
	wantAttention := packedLayerAttentionReference(q, lengths, qHeads, headDim)
	wantOutput, wantNormalized := projectionNormReference(wantAttention, residual, weight, normWeight, dim, op.Epsilon)
	checkLayerOutput(t, dev, doutput, wantOutput, 4e-6)
	checkLayerOutput(t, dev, dnormalized, wantNormalized, 5e-6)

	device, err := batch.Device()
	if err != nil {
		t.Fatal(err)
	}
	generations := make([]uint64, tokens)
	if err := dev.WithContext(func() error {
		if err := device.TokenGenerations.Slice(0, tokens).CopyToHost(generations); err != nil {
			return err
		}
		generations[0]++
		return device.TokenGenerations.Slice(0, tokens).CopyFromHost(generations)
	}); err != nil {
		t.Fatal(err)
	}
	closeResult := make(chan error, 1)
	go func() { closeResult <- metadata.Close() }()
	status, err = batch.RunDecoderLayer(t.Context(), stream, op, dq, dattention, dresidual, dweight, dnormWeight, doutput, dnormalized)
	if err != nil {
		t.Fatal(err)
	}
	if status[0] != 1 {
		t.Fatalf("fault status[0] = %d", status[0])
	}
	got := make([]float32, len(q))
	if err := dev.WithContext(func() error { return dnormalized.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}
	for token := 0; token < tokens; token++ {
		row := token / 2
		for i := 0; i < dim; i++ {
			index := token*dim + i
			if row == 0 {
				if !math.IsNaN(float64(got[index])) {
					t.Fatalf("fault output[%d] = %g, want NaN", index, got[index])
				}
			} else if diff := math.Abs(float64(got[index] - wantNormalized[index])); diff > 5e-6 {
				t.Fatalf("isolated output[%d] = %g, want %g", index, got[index], wantNormalized[index])
			}
		}
	}
	if err := <-closeResult; err == nil {
		t.Fatal("Close succeeded while decoder-layer pins were live")
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
	t.Logf("receipt batch=%d tokens=%d coordinates=%d tolerance=5e-6 heterogeneous_positions=true row_fault_isolated=true one_terminal_sync=true pins_outer_owned=true", len(lengths), tokens, len(q))
}

func packedLayerAttentionReference(q []float32, lengths []int, qHeads, headDim int) []float32 {
	dim := qHeads * headDim
	out := make([]float32, len(q))
	token := 0
	for row, length := range lengths {
		for _, causal := range []int{1, length} {
			for head := 0; head < qHeads; head++ {
				scores := make([]float64, causal)
				maxScore := math.Inf(-1)
				for pos := range scores {
					var dot float64
					for d := 0; d < headDim; d++ {
						key := float32((row+1)*17+pos*5+d) / 31
						dot += float64(q[token*dim+head*headDim+d] * key)
					}
					scores[pos] = dot / math.Sqrt(float64(headDim))
					maxScore = max(maxScore, scores[pos])
				}
				var sum float64
				for pos := range scores {
					scores[pos] = math.Exp(scores[pos] - maxScore)
					sum += scores[pos]
				}
				for d := 0; d < headDim; d++ {
					var value float64
					for pos := range scores {
						v := float32((row+1)*11-pos*3+d) / 19
						value += scores[pos] / sum * float64(v)
					}
					out[token*dim+head*headDim+d] = float32(value)
				}
			}
			token++
		}
	}
	return out
}

func projectionNormReference(attention, residual, weight, normWeight []float32, dim int, epsilon float32) ([]float32, []float32) {
	output := make([]float32, len(attention))
	normalized := make([]float32, len(attention))
	for token := 0; token < len(attention)/dim; token++ {
		var squares float64
		for out := 0; out < dim; out++ {
			value := residual[token*dim+out]
			for in := 0; in < dim; in++ {
				value += attention[token*dim+in] * weight[out*dim+in]
			}
			output[token*dim+out] = value
			squares += float64(value * value)
		}
		scale := float32(1 / math.Sqrt(squares/float64(dim)+float64(epsilon)))
		for i := 0; i < dim; i++ {
			normalized[token*dim+i] = output[token*dim+i] * scale * normWeight[i]
		}
	}
	return output, normalized
}

func checkLayerOutput(t *testing.T, dev *cuda.Device, device interface {
	CopyToHost([]float32) error
}, want []float32, tolerance float64) {
	t.Helper()
	got := make([]float32, len(want))
	if err := dev.WithContext(func() error { return device.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tolerance {
			t.Fatalf("output[%d] = %g, want %g (difference %g)", i, got[i], want[i], diff)
		}
	}
}
