//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func metalDSparkDenseFixture(kind uint32) (data []byte, input []float32, want []float64) {
	const cols, rows, count = 64, 3, 2
	width := 4
	if kind == 1 {
		width = 2
	}
	size := cols * rows * width
	if kind == 8 {
		size = cols * rows / 32 * 34
	}
	data = make([]byte, size)
	input = make([]float32, cols*count)
	for i := range input {
		input[i] = float32(i%11-5) * .25
	}
	want = make([]float64, rows*count)
	for row := range rows {
		for col := range cols {
			integer := (row*3+col)%5 - 2
			value := float32(integer) * .5
			switch kind {
			case 0:
				binary.LittleEndian.PutUint32(data[(row*cols+col)*4:], math.Float32bits(value))
			case 1:
				bits := map[int]uint16{-2: 0xbc00, -1: 0xb800, 0: 0, 1: 0x3800, 2: 0x3c00}[integer]
				binary.LittleEndian.PutUint16(data[(row*cols+col)*2:], bits)
			case 8:
				offset := (row*(cols/32) + col/32) * 34
				binary.LittleEndian.PutUint16(data[offset:], 0x3800)
				data[offset+2+col%32] = byte(int8(integer))
			}
			for b := range count {
				want[b*rows+row] += float64(value) * float64(input[b*cols+col])
			}
		}
	}
	return
}

func TestMetalDSparkDenseFixture(t *testing.T) {
	for _, kind := range []uint32{0, 1, 8} {
		data, input, want := metalDSparkDenseFixture(kind)
		if len(data) == 0 || len(input) != 128 || len(want) != 6 {
			t.Fatal("fixture extent")
		}
		for _, v := range want {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatal("oracle finite")
			}
		}
	}
	if strings.Count(metalDSparkDenseMSL, "output[index]=sum;") != 1 {
		t.Fatal("same-site mutant must be unique")
	}
}

// This opt-in primitive gate does not open a target/support model or qualify
// the full support transformer. Execute only after its source packet review.
func TestMetalDSparkDensePrimitiveGate(t *testing.T) {
	if os.Getenv("DS4_METAL_DSPARK_DENSE_GATE") != "1" {
		t.Skip("opt-in native support primitive gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	for _, kind := range []uint32{0, 1, 8} {
		data, input, want := metalDSparkDenseFixture(kind)
		metalDSparkFrozenFixture(t, string(rune('0'+kind))+"-dense", struct {
			Data  []byte
			Input []float32
			Want  []float64
		}{data, input, want})
	}
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	retained := false
	defer func() {
		if retained {
			return
		}
		if err := dev.Close(); err != nil {
			t.Errorf("device close: %v", err)
		}
	}()
	ops, err := newMetalPinnedOps(dev)
	defer func() {
		if ops != nil && !retained {
			ops.Close()
		}
	}()
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []uint32{0, 1, 8} {
		passed := t.Run(fmt.Sprint(kind), func(t *testing.T) {
			var live *metalBatch
			canRelease := func() bool {
				if retained {
					return false
				}
				if live == nil {
					return true
				}
				if !metalDSparkCanRelease(live.done, func() bool { return metalGroupedTerminal(live) }, func() { live.endEncoding(); live.ops.hc.complete(live) }, &retained) {
					t.Errorf("native support gate retains nonterminal owners")
					return false
				}
				return true
			}
			data, input, want := metalDSparkDenseFixture(kind)
			arena := metalmem.NewArena(dev)
			defer func() {
				if canRelease() {
					arena.Free()
				}
			}()
			storage := metalmem.UploadIn(arena, "support-weights", data)
			w := &metalDSparkWeights{device: dev, model: &dsparkModel{block: 1}, tensors: map[string]boundMetalTensor{"matrix": {name: "matrix", typeID: kind, dimensions: []uint64{64, 3}, data: storage}}}
			d, err := newMetalDSparkDense(dev, w)
			if d == nil && err != nil {
				var owner io.Closer
				if errors.As(err, &owner) {
					defer func() {
						if err := metalDSparkCloseOwner(owner, &retained); err != nil {
							t.Errorf("partial dense owner close: %v", err)
						}
					}()
				}
			}
			defer func() {
				if d == nil || !canRelease() {
					return
				}
				if err := metalDSparkCloseOwner(d, &retained); err != nil {
					t.Errorf("dense close: %v", err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			in := metalmem.UploadIn(arena, "support-input", input)
			out := metalmem.AllocIn[float32](arena, "support-output", 6)
			for _, mutant := range []bool{false, true} {
				original := d.pipeline
				if mutant {
					source := strings.Replace(metalDSparkDenseMSL, "output[index]=sum;", "output[index]=sum+1.0f;", 1)
					library, err := dev.CompileLibrary(source)
					libraryOwner := metalDSparkReleaseOnce(func() {
						if library != nil {
							objectivec.ObjectFromID(library.GetID()).Release()
						}
					})
					defer func() {
						if library != nil && canRelease() {
							libraryOwner()
						}
					}()
					if err != nil {
						t.Fatal(err)
					}
					pipeline, err := dev.NewComputePipeline(library, "dspark_dense")
					pipelineOwner := metalDSparkReleaseOnce(func() {
						if pipeline != nil {
							objectivec.ObjectFromID(pipeline.GetID()).Release()
						}
					})
					defer func() {
						if pipeline != nil && canRelease() {
							pipelineOwner()
						}
					}()
					if err != nil {
						t.Fatal(err)
					}
					d.pipeline = pipeline
				}
				batch, err := newMetalBatch(ops)
				if err != nil {
					t.Fatal(err)
				}
				live = batch
				if err := d.Encode(batch, "matrix", in, out, 2); err != nil {
					t.Fatal(err)
				}
				if err := batch.Wait(t.Context()); err != nil {
					t.Fatal(err)
				}
				changed := 0
				for i, v := range out.HostSlice() {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatalf("nonfinite primitive output %d of %d", i, out.Len())
					}
					if math.Abs(float64(v)-want[i]) > 1e-5 {
						changed++
					}
				}
				if (!mutant && changed != 0) || (mutant && changed != 6) {
					t.Fatalf("native primitive mutant=%v mismatches=%d/6", mutant, changed)
				}
				t.Logf("OPERATOR dense kind=%d mutant=%v compared=6 changed=%d", kind, mutant, changed)
				d.pipeline = original
			}
		})
		if !metalDSparkContinueArm(passed, retained) {
			return
		}
	}
}

func metalDSparkContinueArm(passed, retained bool) bool { return passed && !retained }

func metalDSparkCloseOwner(owner io.Closer, retained *bool) error {
	if err := owner.Close(); err != nil {
		*retained = true
		return err
	}
	return nil
}

func metalDSparkReleaseOnce(release func()) func() {
	done := false
	return func() {
		if !done {
			done = true
			release()
		}
	}
}

type metalDSparkFaultOwner struct {
	calls int
	err   error
}

func (o *metalDSparkFaultOwner) Close() error { o.calls++; return o.err }

func TestMetalDSparkPartialOwnerPolicy(t *testing.T) {
	for _, persistent := range []bool{false, true} {
		failure := errors.New("injected partial close")
		o := &metalDSparkFaultOwner{err: failure}
		wrapped := &backendCleanupOwner{cause: failure, backend: o}
		var partial io.Closer
		if !errors.As(wrapped, &partial) {
			t.Fatal("wrapped partial owner not recovered")
		}
		retained := false
		arena, device, next := 0, 0, 0
		if err := metalDSparkCloseOwner(partial, &retained); err == nil {
			t.Fatal("close failure hidden")
		}
		if !persistent {
			o.err = nil
		}
		_ = metalDSparkCloseOwner(partial, &retained)
		if !retained {
			arena++
			device++
		}
		if metalDSparkContinueArm(true, retained) {
			next++
		}
		if !retained || arena != 0 || device != 0 || next != 0 || o.calls != 2 {
			t.Fatalf("partial owner retention calls=%d retained=%v", o.calls, retained)
		}
	}
	for _, failAfter := range []int{0, 1, 2} {
		library, pipeline := 0, 0
		var releases []func()
		if failAfter >= 1 {
			releases = append(releases, metalDSparkReleaseOnce(func() { library++ }))
		}
		if failAfter >= 2 {
			releases = append(releases, metalDSparkReleaseOnce(func() { pipeline++ }))
		}
		for round := 0; round < 2; round++ {
			for i := len(releases) - 1; i >= 0; i-- {
				releases[i]()
			}
		}
		wantLibrary, wantPipeline := 0, 0
		if failAfter >= 1 {
			wantLibrary = 1
		}
		if failAfter >= 2 {
			wantPipeline = 1
		}
		if library != wantLibrary || pipeline != wantPipeline {
			t.Fatal("partial pipeline/library release not exactly once")
		}
	}
}

func metalDSparkCanRelease(submitted bool, terminal func() bool, cancel func(), retained *bool) bool {
	if *retained {
		return false
	}
	if !submitted {
		cancel()
		return true
	}
	if !terminal() {
		*retained = true
		return false
	}
	return true
}

func TestMetalDSparkArmLifetimePolicy(t *testing.T) {
	for _, tc := range []struct{ passed, retained, want bool }{{true, false, true}, {false, false, false}, {false, true, false}, {true, true, false}} {
		arms, releases := 0, 0
		if metalDSparkContinueArm(tc.passed, tc.retained) {
			arms++
		}
		if !tc.retained {
			releases++
		}
		if (arms == 1) != tc.want {
			t.Fatal("arm continuation policy")
		}
		if tc.retained && (arms != 0 || releases != 0) {
			t.Fatal("nonterminal graph released or reused")
		}
	}
	for _, tc := range []struct{ submitted, terminal bool }{{false, false}, {true, true}, {true, false}} {
		retained := false
		cancels, releases, next := 0, 0, 0
		if metalDSparkCanRelease(tc.submitted, func() bool { return tc.terminal }, func() { cancels++ }, &retained) {
			releases++
		}
		if metalDSparkContinueArm(true, retained) {
			next++
		}
		if tc.submitted && !tc.terminal {
			if !retained || releases != 0 || next != 0 || cancels != 0 {
				t.Fatal("nonterminal ownership policy")
			}
		} else if releases != 1 || retained {
			t.Fatal("terminal/unsubmitted ownership policy")
		}
		if !tc.submitted && cancels != 1 {
			t.Fatal("unsubmitted encoder not cancelled")
		}
	}
}
