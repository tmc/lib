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
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDAPackedAttendCausal(t *testing.T) {
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") != "1" {
		t.Skip("set LIB_CUDA_HARDWARE_TEST=1 on a CUDA host")
	}
	for _, lengths := range [][]int{{3}, {3, 5}, {3, 5, 2, 6}} {
		t.Run(fmt.Sprintf("B%d", len(lengths)), func(t *testing.T) {
			testCUDAPackedAttendCausal(t, lengths)
		})
	}
}

func testCUDAPackedAttendCausal(t *testing.T, lengths []int) {
	const blockSize, qHeads, kvHeads, headDim = 2, 2, 1, 4
	pages := 0
	for _, n := range lengths {
		pages += (n + blockSize - 1) / blockSize
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	a, err := NewCUDA(dev, Config{NumLayers: 1, KVHeads: kvHeads, HeadDim: headDim, BlockSize: blockSize, TotalPages: pages, DType: "float32", Domain: "cuda:0"})
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := NewPackedMetadata(a, dev, len(lengths), len(lengths), pages)
	if err != nil {
		t.Fatal(err)
	}
	requests := make([]PackedRequest, len(lengths))
	for row, length := range lengths {
		requests[row] = PackedRequest{
			BatchRequest: BatchRequest{Key: fmt.Sprintf("row-%d", row), Generation: uint64(row + 1), Length: length, Position: length - 1},
			Positions:    []int{length - 1}, CausalLengths: []int{max(1, length-1)},
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
			use, err := a.Bind(id, generation)
			if err != nil {
				t.Fatal(err)
			}
			page, err := use.Page(0)
			if err != nil {
				t.Fatal(err)
			}
			k, v := make([]float32, blockSize*kvHeads*headDim), make([]float32, blockSize*kvHeads*headDim)
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
	q := make([]float32, len(lengths)*qHeads*headDim)
	for i := range q {
		q[i] = float32(i%9-4) / 7
	}
	dq := cudaAttentionSlice(t, dev, q)
	dout := cudaAttentionSlice(t, dev, make([]float32, len(q)))
	defer dq.Free()
	defer dout.Free()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := cudaattn.CompilePaged(major, minor)
	if err != nil {
		t.Fatal(err)
	}
	module, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer module.Unload()
	kernel, err := module.GetFunction("BatchedPagedAttentionCausal")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	op := CausalAttention{Kernel: kernel, Layer: 0, QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, MaxSeqLen: 6, Threads: 32, Scale: float32(1 / math.Sqrt(headDim))}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := batch.AttendCausal(canceled, stream, op, dq, dout); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-submit cancellation = %v", err)
	}
	status, err := batch.AttendCausal(t.Context(), stream, op, dq, dout)
	if err != nil {
		t.Fatal(err)
	}
	for row := range lengths {
		if status[row] != 0 {
			t.Fatalf("status[%d] = %d", row, status[row])
		}
	}
	got := make([]float32, len(q))
	if err := dev.WithContext(func() error { return dout.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}
	want := packedAttentionReference(q, lengths, qHeads, headDim)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 3e-6 {
			t.Fatalf("output[%d] = %g, want %g", i, got[i], want[i])
		}
	}
	device, err := batch.Device()
	if err != nil {
		t.Fatal(err)
	}
	bad := make([]uint64, len(lengths))
	if err := dev.WithContext(func() error {
		if err := device.TokenGenerations.Slice(0, len(lengths)).CopyToHost(bad); err != nil {
			return err
		}
		bad[0]++
		return device.TokenGenerations.Slice(0, len(lengths)).CopyFromHost(bad)
	}); err != nil {
		t.Fatal(err)
	}
	closeResult := make(chan error, 1)
	go func() { closeResult <- metadata.Close() }()
	status, err = batch.AttendCausal(t.Context(), stream, op, dq, dout)
	if err != nil {
		t.Fatal(err)
	}
	if status[0] != 1 {
		t.Fatalf("fault status[0] = %d", status[0])
	}
	for row := 1; row < len(lengths); row++ {
		if status[row] != 0 {
			t.Fatalf("isolated status[%d] = %d", row, status[row])
		}
	}
	faultOutput := make([]float32, len(q))
	if err := dev.WithContext(func() error { return dout.CopyToHost(faultOutput) }); err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(float64(faultOutput[0])) {
		t.Fatalf("fault output[0] = %g, want NaN", faultOutput[0])
	}
	rowWidth := qHeads * headDim
	for i := rowWidth; i < len(want); i++ {
		if math.Abs(float64(faultOutput[i]-want[i])) > 3e-6 {
			t.Fatalf("isolated output[%d] = %g, want %g", i, faultOutput[i], want[i])
		}
	}
	if err := <-closeResult; err == nil {
		t.Fatal("Close succeeded while attention batch pins were live")
	}
	if err := batch.Complete(t.Context(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	for i := len(admitted) - 1; i >= 0; i-- {
		if err := a.Retire(t.Context(), admitted[i], &fakeCompletion{}); err != nil {
			t.Fatal(err)
		}
	}
	if err := metadata.Close(); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("receipt batch=%d coordinates=%d tolerance=3e-6 independent_positions=true row_fault_isolated=true pins_outer_owned=true", len(lengths), len(want))
}

func cudaAttentionSlice[T any](t *testing.T, dev *cuda.Device, values []T) cudamem.Slice[T] {
	t.Helper()
	s, err := cudamem.Alloc[T](dev, len(values))
	if err != nil {
		t.Fatal(err)
	}
	if err := dev.WithContext(func() error { return s.CopyFromHost(values) }); err != nil {
		_ = s.Free()
		t.Fatal(err)
	}
	return s
}

func packedAttentionReference(q []float32, lengths []int, qHeads, headDim int) []float32 {
	out := make([]float32, len(q))
	scale := 1 / math.Sqrt(float64(headDim))
	for row, length := range lengths {
		causal := max(1, length-1)
		for head := 0; head < qHeads; head++ {
			scores := make([]float64, causal)
			maxScore := math.Inf(-1)
			for token := range scores {
				var dot float64
				for d := 0; d < headDim; d++ {
					kv := float32((row+1)*17+token*5+d) / 31
					dot += float64(q[(row*qHeads+head)*headDim+d] * kv)
				}
				scores[token] = dot * scale
				maxScore = max(maxScore, scores[token])
			}
			var sum float64
			for token := range scores {
				scores[token] = math.Exp(scores[token] - maxScore)
				sum += scores[token]
			}
			for d := 0; d < headDim; d++ {
				var value float64
				for token := range scores {
					v := float32((row+1)*11-token*3+d) / 19
					value += scores[token] / sum * float64(v)
				}
				out[(row*qHeads+head)*headDim+d] = float32(value)
			}
		}
	}
	return out
}
