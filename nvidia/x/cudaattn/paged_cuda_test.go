package cudaattn_test

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestPagedRoPEStoreAndAttention(t *testing.T) {
	for _, seqs := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("batch%d", seqs), func(t *testing.T) {
			testPagedRoPEStoreAndAttention(t, seqs)
		})
	}
}

func testPagedRoPEStoreAndAttention(t *testing.T, seqs int) {
	dev := cudatest.Require(t)
	defer dev.Close()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := cudaattn.CompilePaged(major, minor)
	if err != nil {
		t.Fatalf("CompilePaged: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer mod.Unload()
	store, err := mod.GetFunction("PagedRoPEStoreKV")
	if err != nil {
		t.Fatal(err)
	}
	attention, err := mod.GetFunction("BatchedPagedAttentionDecode")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	const (
		qHeads    = 4
		kvHeads   = 2
		headDim   = 6 // deliberately not a vector-width multiple
		blockSize = 3
		maxBlocks = 2
	)
	totalSlots := seqs*2 + 1
	seqLens := make([]int32, seqs)
	seqIndex := make([]int32, 0, seqs*5)
	positions := make([]int32, 0, seqs*5)
	blockTable := make([]int32, seqs*maxBlocks)
	for seq := 0; seq < seqs; seq++ {
		seqLen := 3
		if seq%2 != 0 {
			seqLen = 5
		}
		seqLens[seq] = int32(seqLen)
		for pos := 0; pos < seqLen; pos++ {
			seqIndex = append(seqIndex, int32(seq))
			positions = append(positions, int32(pos))
		}
		blockTable[seq*maxBlocks] = int32(totalSlots - 1 - seq*2)
		blockTable[seq*maxBlocks+1] = int32(totalSlots - 2 - seq*2)
	}
	tokens := len(seqIndex)
	q := make([]float32, tokens*qHeads*headDim)
	k := make([]float32, tokens*kvHeads*headDim)
	v := make([]float32, len(k))
	for i := range q {
		q[i] = float32((i*17)%29-14) / 19
	}
	for i := range k {
		k[i] = float32((i*11)%31-15) / 17
		v[i] = float32((i*7)%23-11) / 13
	}

	dq := deviceSlice(t, dev, q)
	dk := deviceSlice(t, dev, k)
	dv := deviceSlice(t, dev, v)
	dqr := deviceZeros[float32](t, dev, len(q))
	dkb := deviceZeros[float32](t, dev, totalSlots*blockSize*kvHeads*headDim)
	dvb := deviceZeros[float32](t, dev, totalSlots*blockSize*kvHeads*headDim)
	dtable := deviceSlice(t, dev, blockTable)
	dseq := deviceSlice(t, dev, seqIndex)
	dpos := deviceSlice(t, dev, positions)
	dstatus := deviceSlice(t, dev, []int32{0})
	for _, free := range []func() error{dq.Free, dk.Free, dv.Free, dqr.Free, dkb.Free, dvb.Free, dtable.Free, dseq.Free, dpos.Free, dstatus.Free} {
		defer free()
	}
	threads := uint32(32)
	width := qHeads * headDim
	if kvHeads*headDim > width {
		width = kvHeads * headDim
	}
	gridX := uint32((width + int(threads) - 1) / int(threads))
	if err := cudalaunch.Kernel(store).Grid(gridX, uint32(tokens)).Block(threads).On(stream).Launch(
		dq, dk, dv, dqr, dkb, dvb, dtable, dseq, dpos,
		int32(tokens), int32(maxBlocks), int32(blockSize), int32(qHeads), int32(kvHeads), int32(headDim),
		int32(seqs), int32(totalSlots), float32(10000), dstatus,
	); err != nil {
		t.Fatalf("PagedRoPEStoreKV: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("store synchronize: %v", err)
	}
	checkStatus(t, dev, dstatus)

	wantQR := ropeAll(q, positions, qHeads, headDim, 10000)
	wantK := ropeAll(k, positions, kvHeads, headDim, 10000)
	wantKB := make([]float32, dkb.Len())
	wantVB := make([]float32, dvb.Len())
	for token := 0; token < tokens; token++ {
		seq, pos := int(seqIndex[token]), int(positions[token])
		slot := int(blockTable[seq*maxBlocks+pos/blockSize])
		dst := (slot*blockSize + pos%blockSize) * kvHeads * headDim
		copy(wantKB[dst:dst+kvHeads*headDim], wantK[token*kvHeads*headDim:(token+1)*kvHeads*headDim])
		copy(wantVB[dst:dst+kvHeads*headDim], v[token*kvHeads*headDim:(token+1)*kvHeads*headDim])
	}
	checkDevice(t, dev, dqr, wantQR, 2e-6)
	checkDevice(t, dev, dkb, wantKB, 2e-6)
	checkDevice(t, dev, dvb, wantVB, 0)

	currentQ := make([]float32, seqs*qHeads*headDim)
	tokenBase := 0
	for seq, seqLen := range seqLens {
		last := tokenBase + int(seqLen) - 1
		copy(currentQ[seq*qHeads*headDim:(seq+1)*qHeads*headDim], wantQR[last*qHeads*headDim:(last+1)*qHeads*headDim])
		tokenBase += int(seqLen)
	}
	dcurrent := deviceSlice(t, dev, currentQ)
	dlens := deviceSlice(t, dev, seqLens)
	dout := deviceZeros[float32](t, dev, len(currentQ))
	defer dcurrent.Free()
	defer dlens.Free()
	defer dout.Free()
	if err := dev.WithContext(func() error { return dstatus.CopyFromHost([]int32{0}) }); err != nil {
		t.Fatal(err)
	}
	shared, _ := cudaattn.PagedSharedMemoryBytes(5, int(threads))
	if err := cudalaunch.Kernel(attention).Grid(qHeads, uint32(seqs)).Block(threads).SharedMem(uint32(shared)).On(stream).Launch(
		dcurrent, dkb, dvb, dtable, dlens, dout,
		int32(maxBlocks), int32(blockSize), int32(qHeads), int32(kvHeads), int32(headDim),
		float32(1/math.Sqrt(headDim)), int32(seqs), int32(totalSlots), int32(5), dstatus,
	); err != nil {
		t.Fatalf("BatchedPagedAttentionDecode: %v", err)
	}
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("attention synchronize: %v", err)
	}
	checkStatus(t, dev, dstatus)
	wantOut := denseAttention(currentQ, wantKB, wantVB, blockTable, seqLens, qHeads, kvHeads, headDim, blockSize, maxBlocks)
	checkDevice(t, dev, dout, wantOut, 3e-6)

	// A stale or corrupt device block table must not turn into an out-of-bounds
	// read with plausible output.
	badTable := append([]int32(nil), blockTable...)
	badTable[0] = int32(totalSlots)
	if err := dev.WithContext(func() error {
		if err := dtable.CopyFromHost(badTable); err != nil {
			return err
		}
		return dstatus.CopyFromHost([]int32{0})
	}); err != nil {
		t.Fatal(err)
	}
	if err := cudalaunch.Kernel(attention).Grid(qHeads, uint32(seqs)).Block(threads).SharedMem(uint32(shared)).On(stream).Launch(
		dcurrent, dkb, dvb, dtable, dlens, dout,
		int32(maxBlocks), int32(blockSize), int32(qHeads), int32(kvHeads), int32(headDim),
		float32(1/math.Sqrt(headDim)), int32(seqs), int32(totalSlots), int32(5), dstatus,
	); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatal(err)
	}
	status := []int32{0}
	badOut := make([]float32, len(currentQ))
	if err := dev.WithContext(func() error {
		if err := dstatus.CopyToHost(status); err != nil {
			return err
		}
		return dout.CopyToHost(badOut)
	}); err != nil {
		t.Fatal(err)
	}
	if status[0] != 1 || !math.IsNaN(float64(badOut[0])) {
		t.Fatalf("invalid block status = %d, output[0] = %g", status[0], badOut[0])
	}
}

func deviceSlice[T any](t *testing.T, dev *cuda.Device, values []T) cudamem.Slice[T] {
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

func deviceZeros[T any](t *testing.T, dev *cuda.Device, n int) cudamem.Slice[T] {
	return deviceSlice(t, dev, make([]T, n))
}

func checkStatus(t *testing.T, dev *cuda.Device, status cudamem.Slice[int32]) {
	t.Helper()
	host := []int32{0}
	if err := dev.WithContext(func() error { return status.CopyToHost(host) }); err != nil {
		t.Fatal(err)
	}
	if host[0] != 0 {
		t.Fatalf("device status = %d", host[0])
	}
}

func checkDevice(t *testing.T, dev *cuda.Device, gotDevice cudamem.Slice[float32], want []float32, tolerance float64) {
	t.Helper()
	got := make([]float32, len(want))
	if err := dev.WithContext(func() error { return gotDevice.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tolerance {
			t.Fatalf("element %d = %g, want %g (difference %g)", i, got[i], want[i], diff)
		}
	}
}

func ropeAll(src []float32, positions []int32, heads, headDim int, theta float64) []float32 {
	dst := make([]float32, len(src))
	half := headDim / 2
	width := heads * headDim
	for token, pos := range positions {
		for h := 0; h < heads; h++ {
			base := token*width + h*headDim
			for d := 0; d < half; d++ {
				angle := float64(pos) / math.Pow(theta, 2*float64(d)/float64(headDim))
				a, b := src[base+d], src[base+d+half]
				dst[base+d] = a*float32(math.Cos(angle)) - b*float32(math.Sin(angle))
				dst[base+d+half] = a*float32(math.Sin(angle)) + b*float32(math.Cos(angle))
			}
		}
	}
	return dst
}

func denseAttention(q, k, v []float32, table, lens []int32, qHeads, kvHeads, headDim, blockSize, maxBlocks int) []float32 {
	out := make([]float32, len(q))
	group := qHeads / kvHeads
	for seq, seqLen32 := range lens {
		seqLen := int(seqLen32)
		for qh := 0; qh < qHeads; qh++ {
			kh := qh / group
			scores := make([]float64, seqLen)
			maxScore := math.Inf(-1)
			for pos := 0; pos < seqLen; pos++ {
				slot := int(table[seq*maxBlocks+pos/blockSize])
				kbase := ((slot*blockSize+pos%blockSize)*kvHeads + kh) * headDim
				qbase := (seq*qHeads + qh) * headDim
				for d := 0; d < headDim; d++ {
					scores[pos] += float64(q[qbase+d] * k[kbase+d])
				}
				scores[pos] /= math.Sqrt(float64(headDim))
				maxScore = math.Max(maxScore, scores[pos])
			}
			sum := 0.0
			for pos := range scores {
				scores[pos] = math.Exp(scores[pos] - maxScore)
				sum += scores[pos]
			}
			for d := 0; d < headDim; d++ {
				for pos, score := range scores {
					slot := int(table[seq*maxBlocks+pos/blockSize])
					vbase := ((slot*blockSize+pos%blockSize)*kvHeads + kh) * headDim
					out[(seq*qHeads+qh)*headDim+d] += float32(score/sum) * v[vbase+d]
				}
			}
		}
	}
	return out
}
