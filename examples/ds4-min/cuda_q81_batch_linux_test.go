//go:build linux

package main

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDAQualificationQ81BatchQuantize(t *testing.T) {
	r := cudaSynthFoundation(t)
	const (
		batch  = 2
		cols   = 64
		stride = 96
	)
	host := make([]float32, (batch-1)*stride+cols)
	for token := range batch {
		for i := range cols {
			host[token*stride+i] = float32((token+1)*(i*29-701)) / 257
		}
	}
	in := cudaSynthUpload(t, r, host)
	out := cudaExpertAllocBytes(t, r, batch*cols/32*36)
	wantDevice := cudaExpertAllocBytes(t, r, batch*cols/32*36)
	if err := r.encodeQuantQ81Batch(in, out, cols, stride, batch); err != nil {
		t.Fatal(err)
	}
	for token := range batch {
		if err := r.encodeQuantQ81(
			in.Slice(token*stride, token*stride+cols),
			wantDevice.Slice(token*cols/32*36, (token+1)*cols/32*36), cols); err != nil {
			t.Fatal(err)
		}
	}
	got, want := make([]byte, out.Len()), make([]byte, wantDevice.Len())
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if err := wantDevice.CopyToHost(want); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("batched quantization differs from sequential rows")
	}
	for _, bad := range []struct {
		name                string
		in                  cudamem.Slice[float32]
		out                 cudamem.Slice[byte]
		cols, stride, batch int
	}{
		{"zero batch", in, out, cols, stride, 0},
		{"short input", in.Slice(0, in.Len()-1), out, cols, stride, batch},
		{"short output", in, out.Slice(0, out.Len()-1), cols, stride, batch},
		{"short stride", in, out, cols, cols - 1, batch},
	} {
		t.Run(bad.name, func(t *testing.T) {
			if err := r.encodeQuantQ81Batch(bad.in, bad.out, bad.cols, bad.stride, bad.batch); err == nil {
				t.Fatal("accepted invalid shape")
			}
		})
	}
}

func TestCUDAQualificationQ81BatchMatvec(t *testing.T) {
	r := cudaSynthFoundation(t)
	const (
		batch = 2
		cols  = 1024
		rows  = 128
	)
	weights := cudaSynthQ80(rand.New(rand.NewSource(81)), rows, cols, -8)
	tensor := cudaSynthTensor(t, r, "batched Q8_0 fixture", 8, []uint64{cols, rows}, weights)
	x := make([]float32, batch*cols)
	for i := range x {
		x[i] = float32((i*37+i/17+i/cols*29)%257-128) / 128
	}
	raw := cudaQ81ActivationReference(x)
	in := cudaSynthUpload(t, r, x)
	q := cudaExpertAllocBytes(t, r, len(raw))
	if err := r.encodeQuantQ81(in, q, cols); err != nil {
		t.Fatal(err)
	}
	out := cudaTestAlloc(t, r, batch*rows)
	sequential := cudaTestAlloc(t, r, batch*rows)
	if err := out.CopyFromHost(makeNaNs(batch * rows)); err != nil {
		t.Fatal(err)
	}
	cudaQualificationMeasure(t, r, func() error {
		return r.encodeQ80MatvecQ81Batch(tensor, q, out, batch)
	})
	got := cudaTestRead(t, out)
	cudaQualificationMeasure(t, r, func() error {
		for token := range batch {
			qStart := token * cols / 32 * 36
			if err := r.encodeQ80MatvecQ81(tensor, q.Slice(qStart, qStart+cols/32*36), sequential.Slice(token*rows, (token+1)*rows)); err != nil {
				return err
			}
		}
		return nil
	})
	wantDevice := cudaTestRead(t, sequential)
	if err := exactFloat32Bits(got, wantDevice); err != nil {
		t.Fatalf("batch differs from single-row launches: %v", err)
	}
	for token := range batch {
		qStart := token * cols / 32 * 36
		want := cudaQ81MatvecReference(t, weights, raw[qStart:qStart+cols/32*36], cols, rows)
		if err := cudaQualificationClose(got[token*rows:(token+1)*rows], want, cudaQ81ATol, cudaQ81RTol); err != nil {
			t.Fatalf("token %d differs from reference: %v", token, err)
		}
	}
	if err := r.encodeQ80MatvecQ81Batch(tensor, q, out, 0); err == nil {
		t.Fatal("accepted zero batch")
	}
	if err := r.encodeQ80MatvecQ81Batch(tensor, q.Slice(0, q.Len()-1), out, batch); err == nil {
		t.Fatal("accepted short input")
	}
	if err := r.encodeQ80MatvecQ81Batch(tensor, q, out.Slice(0, out.Len()-1), batch); err == nil {
		t.Fatal("accepted short output")
	}
}

func exactFloat32Bits(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("length %d, want %d", len(got), len(want))
	}
	for i := range got {
		gb, wb := math.Float32bits(got[i]), math.Float32bits(want[i])
		if gb != wb {
			return fmt.Errorf("value %d bits %08x, want %08x", i, gb, wb)
		}
	}
	return nil
}

func TestExactFloat32Bits(t *testing.T) {
	for _, tt := range []struct {
		name string
		got  []float32
		want []float32
		ok   bool
	}{
		{"equal", []float32{1, -2}, []float32{1, -2}, true},
		{"length", []float32{1}, []float32{1, 2}, false},
		{"signed zero", []float32{0}, []float32{math.Float32frombits(1 << 31)}, false},
		{"nan payload", []float32{math.Float32frombits(0x7fc00001)}, []float32{math.Float32frombits(0x7fc00002)}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := exactFloat32Bits(tt.got, tt.want); (err == nil) != tt.ok {
				t.Fatalf("exactFloat32Bits error = %v, want success %t", err, tt.ok)
			}
		})
	}
}

func makeNaNs(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(math.NaN())
	}
	return v
}
