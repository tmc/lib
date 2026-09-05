//go:build linux

package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Preregistered after the frozen candidate04 model repeat changed logits.
// This is exact repeatability, not numerical equivalence: 32 iterations in each
// mode, one common baseline, all float bits and quantized bytes checked. Queued
// mode has no host synchronization between stages. Q80 scratch snapshots use
// same-stream device copies before the next stage overwrites the scratch.
const cudaDeterminismRepeats = 32

type cudaDeterminismBuffer struct {
	name   string
	floats cudamem.Slice[float32]
	bytes  cudamem.Slice[byte]
}

type cudaDeterminismStage struct {
	name    string
	encode  func() error
	outputs []cudaDeterminismBuffer
}

func cudaDeterminismFloats(name string, d cudamem.Slice[float32]) cudaDeterminismBuffer {
	return cudaDeterminismBuffer{name: name, floats: d}
}

func cudaDeterminismBytes(name string, d cudamem.Slice[byte]) cudaDeterminismBuffer {
	return cudaDeterminismBuffer{name: name, bytes: d}
}

func cudaDeterminismPoison(t *testing.T, b cudaDeterminismBuffer, iteration int) {
	t.Helper()
	if b.floats.Len() != 0 {
		p := make([]float32, b.floats.Len())
		for i := range p {
			p[i] = math.Float32frombits(0x7fc00000 | uint32(iteration+1))
		}
		if err := b.floats.CopyFromHost(p); err != nil {
			t.Fatal(err)
		}
	} else {
		p := make([]byte, b.bytes.Len())
		for i := range p {
			p[i] = byte(0xa5 ^ iteration)
		}
		if err := b.bytes.CopyFromHost(p); err != nil {
			t.Fatal(err)
		}
	}
}

func cudaDeterminismRead(t *testing.T, b cudaDeterminismBuffer) []byte {
	t.Helper()
	if b.floats.Len() == 0 {
		p := make([]byte, b.bytes.Len())
		if err := b.bytes.CopyToHost(p); err != nil {
			t.Fatal(err)
		}
		return p
	}
	values := cudaTestRead(t, b.floats)
	p := make([]byte, len(values)*4)
	for i, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s unwritten/nonfinite float[%d]=%08x", b.name, i, math.Float32bits(v))
		}
		binary.LittleEndian.PutUint32(p[i*4:], math.Float32bits(v))
	}
	return p
}

func cudaDeterminismDifference(got, want []byte, floats bool) error {
	if len(got) != len(want) {
		return fmt.Errorf("length %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] == want[i] {
			continue
		}
		if floats {
			j := i / 4
			g := binary.LittleEndian.Uint32(got[j*4:])
			w := binary.LittleEndian.Uint32(want[j*4:])
			return fmt.Errorf("float[%d] got %.9g (0x%08x), first %.9g (0x%08x)", j, math.Float32frombits(g), g, math.Float32frombits(w), w)
		}
		return fmt.Errorf("byte[%d] got 0x%02x, first 0x%02x", i, got[i], want[i])
	}
	return nil
}

func cudaDeterminismRun(t *testing.T, r *cudaFoundation, stages []cudaDeterminismStage, scratch ...cudaDeterminismBuffer) {
	t.Helper()
	baseline := make(map[string][]byte)
	for _, mode := range []string{"queued", "stage_sync"} {
		t.Run(mode, func(t *testing.T) {
			for iteration := range cudaDeterminismRepeats {
				for _, b := range scratch {
					cudaDeterminismPoison(t, b, iteration)
				}
				for _, stage := range stages {
					for _, b := range stage.outputs {
						cudaDeterminismPoison(t, b, iteration)
					}
				}
				for _, stage := range stages {
					if err := stage.encode(); err != nil {
						t.Fatalf("iteration %d stage %s: %v", iteration, stage.name, err)
					}
					if mode == "stage_sync" {
						if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
							t.Fatal(err)
						}
					}
				}
				if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
					t.Fatal(err)
				}
				for _, stage := range stages {
					for _, b := range stage.outputs {
						key := stage.name + "/" + b.name
						got := cudaDeterminismRead(t, b)
						want, ok := baseline[key]
						if !ok {
							baseline[key] = got
							t.Logf("baseline stage=%s bytes=%d sha256=%x", key, len(got), sha256.Sum256(got))
							continue
						}
						if err := cudaDeterminismDifference(got, want, b.floats.Len() != 0); err != nil {
							t.Fatalf("first mismatch: mode=%s iteration=%d stage=%s: %v", mode, iteration, key, err)
						}
					}
				}
			}
			t.Logf("%d exact repeats, %d stages; baseline shared across both modes", cudaDeterminismRepeats, len(stages))
		})
		if t.Failed() {
			return
		}
	}
}

func cudaDeterminismInput(n, seed int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32((i*37+i/17+seed*13)%257-128) / 128
	}
	return x
}

func TestCUDADeterminismQ80Scratch(t *testing.T) {
	r := cudaSynthFoundation(t)
	scratch := cudaExpertAllocBytes(t, r, 8192/32*36)
	runtime := &cudaRuntime{cudaFoundation: r}
	runtime.state.q80Input = scratch
	var stages []cudaDeterminismStage
	for step, cols := range []int{4096, 8192, 4096, 8192} {
		const rows = 129
		input := cudaSynthUpload(t, r, cudaDeterminismInput(cols, cols))
		raw := cudaSynthQ80(rand.New(rand.NewSource(int64(cols))), rows, cols, -8)
		weight := cudaSynthTensor(t, r, fmt.Sprintf("q80-%d", step), 8, []uint64{uint64(cols), rows}, raw)
		out := cudaTestAlloc(t, r, rows)
		snapshot := cudaExpertAllocBytes(t, r, cols/32*36)
		stages = append(stages, cudaDeterminismStage{
			name: fmt.Sprintf("step%d-cols%d", step, cols),
			encode: func() error {
				if err := runtime.encodeQ80Native(weight, input, out); err != nil {
					return err
				}
				return snapshot.CopyDeviceToDeviceAsync(scratch.Slice(0, snapshot.Len()), r.device.DefaultStream())
			},
			outputs: []cudaDeterminismBuffer{cudaDeterminismFloats("projection", out), cudaDeterminismBytes("quantized", snapshot)},
		})
	}
	cudaDeterminismRun(t, r, stages, cudaDeterminismBytes("shared-scratch", scratch))
}

func TestCUDADeterminismF16(t *testing.T) {
	r := cudaSynthFoundation(t)
	input := cudaSynthUpload(t, r, cudaDeterminismInput(4096, 7))
	var stages []cudaDeterminismStage
	for _, rows := range []int{256, 512, 1024} {
		a, _ := cudaSynthF16(rand.New(rand.NewSource(int64(rows))), rows, 4096, -5)
		b, _ := cudaSynthF16(rand.New(rand.NewSource(int64(rows+1))), rows, 4096, -5)
		ta := cudaSynthTensor(t, r, "f16-a", 1, []uint64{4096, uint64(rows)}, a)
		tb := cudaSynthTensor(t, r, "f16-b", 1, []uint64{4096, uint64(rows)}, b)
		oa, ob, od := cudaTestAlloc(t, r, rows), cudaTestAlloc(t, r, rows), cudaTestAlloc(t, r, rows)
		stages = append(stages, cudaDeterminismStage{
			name: fmt.Sprintf("pair-%d", rows), encode: func() error { return r.encodeF16PairOrdered(ta, tb, input, oa, ob) },
			outputs: []cudaDeterminismBuffer{cudaDeterminismFloats("a", oa), cudaDeterminismFloats("b", ob)},
		}, cudaDeterminismStage{
			name: fmt.Sprintf("default-%d", rows), encode: func() error { return r.encodeF16MatvecDefault(ta, input, od) },
			outputs: []cudaDeterminismBuffer{cudaDeterminismFloats("output", od)},
		})
	}
	cudaDeterminismRun(t, r, stages)
}

func TestCUDADeterminismQ81Experts(t *testing.T) {
	r := cudaSynthFoundation(t)
	const cols, hidden, rows, experts, slots = 4096, 2048, 129, 9, 6
	ids, err := cudamem.Alloc[int32](r.device, slots)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ids.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := ids.CopyFromHost([]int32{8, 2, 5, 0, 2, 7}); err != nil {
		t.Fatal(err)
	}
	weights := cudaSynthUpload(t, r, []float32{.25, .1875, .125, .0625, .15625, .21875})
	input := cudaSynthUpload(t, r, cudaDeterminismInput(cols, 17))
	q := cudaExpertAllocBytes(t, r, cols/32*36)
	sq := cudaExpertAllocBytes(t, r, slots*hidden/32*36)
	gate := cudaSynthTensor(t, r, "iq2-gate", 16, []uint64{cols, hidden, experts}, cudaExpertFixture(16, cols, hidden, experts, 17))
	up := cudaSynthTensor(t, r, "iq2-up", 16, []uint64{cols, hidden, experts}, cudaExpertFixture(16, cols, hidden, experts, 29))
	down := cudaSynthTensor(t, r, "q2-down", 10, []uint64{hidden, rows, experts}, cudaExpertFixture(10, hidden, rows, experts, 43))
	g, u, mid, d := cudaTestAlloc(t, r, slots*hidden), cudaTestAlloc(t, r, slots*hidden), cudaTestAlloc(t, r, slots*hidden), cudaTestAlloc(t, r, slots*rows)
	stages := []cudaDeterminismStage{
		{"quant-input", func() error { return r.encodeQuantQ81(input, q, cols) }, []cudaDeterminismBuffer{cudaDeterminismBytes("q81", q)}},
		{"iq2-pair", func() error { return r.encodeMatvecIQ2SelectedQ81(gate, up, ids, q, g, u) }, []cudaDeterminismBuffer{cudaDeterminismFloats("gate", g), cudaDeterminismFloats("up", u)}},
		{"swiglu", func() error { return r.encodeSwiGLUSlots(g, u, weights, mid, slots, 1) }, []cudaDeterminismBuffer{cudaDeterminismFloats("mid", mid)}},
		{"quant-experts", func() error { return r.encodeQuantQ81(mid, sq, hidden) }, []cudaDeterminismBuffer{cudaDeterminismBytes("q81", sq)}},
		{"q2-down", func() error { return r.encodeMatvecQ2KSelectedQ81(down, ids, sq, d) }, []cudaDeterminismBuffer{cudaDeterminismFloats("down", d)}},
	}
	cudaDeterminismRun(t, r, stages)
}

func TestCUDADeterminismComparator(t *testing.T) {
	first := []byte{0, 0, 128, 63}
	if err := cudaDeterminismDifference(first, append([]byte(nil), first...), true); err != nil {
		t.Fatal(err)
	}
	for _, floats := range []bool{false, true} {
		changed := append([]byte(nil), first...)
		changed[0] ^= 1
		if err := cudaDeterminismDifference(changed, first, floats); err == nil {
			t.Fatal("accepted one-bit mutation")
		}
	}
	if err := cudaDeterminismDifference(first[:3], first, false); err == nil {
		t.Fatal("accepted truncated capture")
	}
}
