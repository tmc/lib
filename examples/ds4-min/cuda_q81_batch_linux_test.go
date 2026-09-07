//go:build linux

package main

import (
	"math"
	"math/rand"
	"testing"
)

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
	if err := cudaQualificationClose(got, wantDevice, 0, 0); err != nil {
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

func makeNaNs(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(math.NaN())
	}
	return v
}
