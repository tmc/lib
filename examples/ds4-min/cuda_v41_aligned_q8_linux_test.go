//go:build linux

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDAV41AlignedQ8Route(t *testing.T) {
	for _, v := range []struct {
		name string
		k, m int
		want bool
	}{{"blk.0.attn_q_a.weight", 5120, 1280, true}, {"blk.0.attn_kv.weight", 5120, 512, true}, {"blk.0.attn_output_b.weight", 8192, 5120, true}, {"blk.0.attn_q_b.weight", 1280, 32768, false}, {"blk.0.attn_output_a.weight", 4096, 8192, false}} {
		d := dsv41.TensorDescriptor{Name: v.name, Encoding: 8, Bytes: int64(v.k / 32 * 34 * v.m), Dimensions: []uint64{uint64(v.k), uint64(v.m)}}
		if got := cudaV41AlignedQ8Eligible(d); got != v.want {
			t.Fatalf("%s aligned eligibility=%t want=%t", v.name, got, v.want)
		}
		d.Encoding = 12
		if cudaV41AlignedQ8Eligible(d) {
			t.Fatal("Q4_K admitted to Q8 route")
		}
	}
}

func TestCUDAV41AlignedQ8ValuesAndMutants(t *testing.T) {
	testCUDAV41AlignedQ8ValuesAndMutants(t, []v41AlignedShape{{"blk.0.attn_q_a.weight", 5120, 1280}, {"blk.0.attn_kv.weight", 5120, 512}, {"blk.0.attn_output_b.weight", 8192, 5120}})
}

type v41AlignedShape struct {
	name string
	k, m int
}

func testCUDAV41AlignedQ8ValuesAndMutants(t *testing.T, shapes []v41AlignedShape) {
	t.Helper()
	if os.Getenv("DS4_CUDA_V41_OPS_GATE") != "1" {
		t.Skip("opt-in V4.1 CUDA gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	capture := newV41AlignedCapture(t)
	for _, variant := range []struct{ name, old, new string }{
		{"baseline", "", ""},
		{"wrong_scale", "y[b].ds=__floats2half2_rn(d,sum);", "y[b].ds=__floats2half2_rn(2*d,sum);"},
		{"wrong_layout", "else aligned[scale_bytes+block*32+j-2]=raw[block*34+j];", "else aligned[scale_bytes+block*32+j-2]=raw[block*34+2+((j-2)^1)];"},
	} {
		t.Run(variant.name, func(t *testing.T) {
			source := cudaV41AlignedQ8Source
			if variant.old != "" {
				if strings.Count(source, variant.old) != 1 {
					t.Fatal("mutation site not unique")
				}
				source = strings.Replace(source, variant.old, variant.new, 1)
			}
			capture.save(t, "compiled-source", []byte(source))
			r := cudaSynthFoundation(t)
			_, literalQuantize, literalMatvec := v41AlignedLiteralModule(t, r)
			ops, err := newCUDAV41Ops(r)
			if ops != nil {
				t.Cleanup(func() {
					if e := ops.Close(); e != nil {
						t.Error(e)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			for _, shape := range shapes {
				t.Run(shape.name, func(t *testing.T) {
					blocks := shape.k / 32
					nblk := blocks * shape.m
					raw := make([]byte, nblk*34)
					for row := 0; row < shape.m; row++ {
						for b := 0; b < blocks; b++ {
							p := (row*blocks + b) * 34
							binary.LittleEndian.PutUint16(raw[p:], uint16(0x2400+((row+b)%3)*0x400))
							for j := 0; j < 32; j++ {
								raw[p+2+j] = byte(int8((row*3+b*5+j*7)%31 - 15))
							}
						}
					}
					capture.save(t, "input-weight-q8_0", raw)
					scaleBytes := (nblk*2 + 63) &^ 63
					wantAligned := make([]byte, scaleBytes+nblk*32)
					for b := 0; b < nblk; b++ {
						copy(wantAligned[b*2:b*2+2], raw[b*34:b*34+2])
						copy(wantAligned[scaleBytes+b*32:scaleBytes+(b+1)*32], raw[b*34+2:(b+1)*34])
					}
					capture.save(t, "reference-aligned", wantAligned)
					dRaw, e := cudamem.Alloc[byte](r.device, len(raw))
					if dRaw.Ptr() != 0 {
						t.Cleanup(func() {
							if e := dRaw.Free(); e != nil {
								t.Error(e)
							}
						})
					}
					if e != nil {
						t.Fatal(e)
					}
					if e := dRaw.CopyFromHost(raw); e != nil {
						t.Fatal(e)
					}
					d := dsv41.TensorDescriptor{Name: shape.name, Encoding: 8, Bytes: int64(len(raw)), Dimensions: []uint64{uint64(shape.k), uint64(shape.m)}}
					op, e := compileCUDAV41AlignedQ8(r, cudaV41Layer0Q8Weight{d, dRaw}, source)
					if op != nil {
						t.Cleanup(func() {
							if e := op.Close(); e != nil {
								t.Error(e)
							}
						})
					}
					if e != nil {
						t.Fatal(e)
					}
					// Repack is a one-time owned copy: raw can be freed before Eval.
					if e := dRaw.Free(); e != nil {
						t.Fatal(e)
					}
					aligned := make([]byte, op.aligned.Len())
					if e := op.aligned.CopyToHost(aligned); e != nil {
						t.Fatal(e)
					}
					layoutChanged := 0
					capture.save(t, "actual-aligned", aligned)
					for i := range aligned {
						if aligned[i] != wantAligned[i] {
							layoutChanged++
						}
					}
					if variant.name == "wrong_layout" {
						if layoutChanged == 0 {
							t.Fatal("layout mutant survived full artifact byte oracle")
						}
					} else if layoutChanged != 0 {
						t.Fatal("weight artifact byte mismatch")
					}
					for i := 0; i < int(op.scaleBytes); i++ {
						if aligned[i] != wantAligned[i] {
							t.Fatal("layout mutant altered adjacent scale/pad stream")
						}
					}
					x := make([]float32, shape.k)
					for b := 0; b < blocks; b++ {
						delta := float32(1) + float32((b%3)*2+1)/2048
						switch b % 17 {
						case 1:
							delta = 0
						case 2:
							delta = 1.0 / (1 << 25)
						case 3:
							delta = 3.0 / (1 << 25)
						}
						for j := 0; j < 32; j++ {
							x[b*32+j] = float32(float32(j%11-5)+.5) * delta
						}
						x[b*32] = 127 * delta
						x[b*32+31] = -127 * delta
						if b%17 == 4 {
							for j := 0; j < 32; j++ {
								x[b*32+j] = math.SmallestNonzeroFloat32 * float32(j%7-3)
							}
						}
					}
					packed, floatD := v41AlignedActivationReference(x)
					capture.save(t, "input-activation", x)
					capture.save(t, "reference-canonical", packed)
					capture.save(t, "reference-float-scales", floatD)
					want, dots := v41AlignedOutputReference(raw, packed, shape.k, shape.m)
					capture.save(t, "reference-output", want)
					capture.save(t, "reference-diagnostic-dots", dots)
					if binary.LittleEndian.Uint16(packed) == v41AlignedHalfBits(floatD[0]+1.0/2048) {
						t.Fatal("half-scale tie fixture degenerate")
					}
					dx, out := cudaSynthUpload(t, r, x), cudaSynthUpload(t, r, make([]float32, shape.m))
					refAligned, e := cudamem.Alloc[byte](r.device, len(wantAligned))
					if refAligned.Ptr() != 0 {
						t.Cleanup(func() {
							if e := refAligned.Free(); e != nil {
								t.Error(e)
							}
						})
					}
					if e != nil {
						t.Fatal(e)
					}
					if e := refAligned.CopyFromHost(wantAligned); e != nil {
						t.Fatal(e)
					}
					refPacked, e := cudamem.Alloc[byte](r.device, len(packed))
					if refPacked.Ptr() != 0 {
						t.Cleanup(func() {
							if e := refPacked.Free(); e != nil {
								t.Error(e)
							}
						})
					}
					if e != nil {
						t.Fatal(e)
					}
					refOut := cudaSynthUpload(t, r, make([]float32, shape.m))
					dDots, e := cudamem.Alloc[int32](r.device, nblk)
					if dDots.Ptr() != 0 {
						t.Cleanup(func() {
							if e := dDots.Free(); e != nil {
								t.Error(e)
							}
						})
					}
					if e != nil {
						t.Fatal(e)
					}
					t.Cleanup(func() {
						if e := op.Close(); e != nil {
							t.Error(e)
						}
					})
					var saved []uint32
					for repeat := 0; repeat < 2; repeat++ {
						if e := op.Eval(out, dx); e != nil {
							t.Fatal(e)
						}
						if e := r.device.WithContext(func() error {
							if e := cudalaunch.Kernel(literalQuantize).Grid(uint32((shape.k+255)/256), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(dx, refPacked, int64(shape.k), int64(shape.k), int64(shape.k), int64(shape.k), int64(shape.k), uint32(1), uint32(1)); e != nil {
								return e
							}
							return cudalaunch.Kernel(literalMatvec).Grid(uint32((shape.m+int(op.warps)-1)/int(op.warps)), 1, 1).Block(op.warps*32, 1, 1).On(r.stream()).Launch(refOut, driver.DevicePtr(uint64(refAligned.Ptr())+op.scaleBytes), refAligned, refPacked, int32(shape.m), int32(blocks))
						}); e != nil {
							t.Fatal(e)
						}
						if e := r.device.WithContext(func() error { return syncStream(r.stream()) }); e != nil {
							t.Fatal(e)
						}
						literalPacked := make([]byte, len(packed))
						if e := refPacked.CopyToHost(literalPacked); e != nil {
							t.Fatal(e)
						}
						capture.save(t, fmt.Sprintf("repeat-%d-literal-canonical", repeat), literalPacked)
						for i := range packed {
							if literalPacked[i] != packed[i] {
								t.Fatalf("pinned literal/CPU activation mismatch byte=%d; flag-sensitive STOP", i)
							}
						}
						gotPacked := make([]byte, len(packed))
						if e := op.activation.CopyToHost(gotPacked); e != nil {
							t.Fatal(e)
						}
						capture.save(t, fmt.Sprintf("repeat-%d-actual-canonical", repeat), gotPacked)
						packedChanged := 0
						for i := range packed {
							if gotPacked[i] != packed[i] {
								packedChanged++
							}
						}
						if variant.name == "wrong_scale" {
							if packedChanged == 0 {
								t.Fatal("half-scale mutant survived activation byte oracle")
							}
							for i := range packed {
								if i%36 >= 2 && gotPacked[i] != packed[i] {
									t.Fatal("scale mutant altered adjacent sum/codes")
								}
							}
						} else if packedChanged != 0 {
							t.Fatal("canonical activation code/half-d/half-s mismatch")
						}
						if e := r.device.WithContext(func() error {
							return cudalaunch.Kernel(op.dots).Grid(uint32((nblk+255)/256), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(dDots, op.aligned, op.activation, uint32(shape.m), uint32(blocks), op.scaleBytes)
						}); e != nil {
							t.Fatal(e)
						}
						if e := r.device.WithContext(func() error { return syncStream(r.stream()) }); e != nil {
							t.Fatal(e)
						}
						gotDots := make([]int32, nblk)
						if e := dDots.CopyToHost(gotDots); e != nil {
							t.Fatal(e)
						}
						capture.save(t, fmt.Sprintf("repeat-%d-actual-diagnostic-dots", repeat), gotDots)
						dotChanged := 0
						for i := range dots {
							if gotDots[i] != dots[i] {
								dotChanged++
							}
						}
						if variant.name == "wrong_layout" {
							if dotChanged == 0 {
								t.Fatal("layout mutant survived integer-dot oracle")
							}
						} else if dotChanged != 0 {
							t.Fatal("integer-dot mismatch")
						}
						got := make([]float32, shape.m)
						if e := out.CopyToHost(got); e != nil {
							t.Fatal(e)
						}
						literalOut := make([]float32, shape.m)
						if e := refOut.CopyToHost(literalOut); e != nil {
							t.Fatal(e)
						}
						capture.save(t, fmt.Sprintf("repeat-%d-actual-output", repeat), got)
						capture.save(t, fmt.Sprintf("repeat-%d-literal-output", repeat), literalOut)
						for i := range literalOut {
							if math.Float32bits(literalOut[i]) != math.Float32bits(want[i]) {
								t.Fatalf("pinned literal/CPU matvec mismatch row=%d; flag-sensitive STOP", i)
							}
							if variant.name == "baseline" && math.Float32bits(got[i]) != math.Float32bits(literalOut[i]) {
								t.Fatalf("port/pinned literal matvec mismatch row=%d", i)
							}
						}
						changed := 0
						bits := make([]uint32, len(got))
						for i := range got {
							bits[i] = math.Float32bits(got[i])
							if bits[i] != math.Float32bits(want[i]) {
								changed++
							}
						}
						t.Logf("repeat=%d layout/activation/dot/output differences=%d/%d/%d/%d", repeat, layoutChanged, packedChanged, dotChanged, changed)
						if variant.name == "baseline" {
							if changed != 0 {
								t.Fatal("aligned output mismatch")
							}
							if repeat == 0 {
								saved = bits
							} else {
								for i := range bits {
									if bits[i] != saved[i] {
										t.Fatal("repeat mismatch")
									}
								}
							}
						} else if changed == 0 {
							t.Fatal("device mutant survived output oracle")
						}
						if e := ops.Quantize(out, uint32(shape.m), 1, dsv41.BFloat16); e != nil {
							t.Fatal(e)
						}
						if e := r.device.WithContext(func() error { return syncStream(r.stream()) }); e != nil {
							t.Fatal(e)
						}
						if e := out.CopyToHost(got); e != nil {
							t.Fatal(e)
						}
						capture.save(t, fmt.Sprintf("repeat-%d-actual-bf16", repeat), got)
						wantBF16 := make([]float32, len(want))
						for i := range want {
							wantBF16[i] = dsv41.BF16(want[i])
						}
						capture.save(t, fmt.Sprintf("repeat-%d-reference-bf16", repeat), wantBF16)
						if variant.name == "baseline" {
							for i := range got {
								if math.Float32bits(got[i]) != math.Float32bits(dsv41.BF16(want[i])) {
									t.Fatal("BF16 boundary mismatch")
								}
							}
						}
					}
					if variant.name == "baseline" {
						f := op.matvec
						op.matvec = 0
						e := op.Eval(out, dx)
						op.matvec = f
						if e == nil || op.valid || len(op.pending) != 0 {
							t.Fatal("post-quantize launch error not drained/invalidated")
						}
						if e := op.Eval(out, dx); e == nil {
							t.Fatal("invalid owner reused")
						}
						if e := op.Reset(); e != nil {
							t.Fatal(e)
						}
						if e := op.Eval(out, dx); e != nil {
							t.Fatal(e)
						}
						replay := make([]float32, shape.m)
						if e := out.CopyToHost(replay); e != nil {
							t.Fatal(e)
						}
						capture.save(t, "reset-replay-output", replay)
						for i := range replay {
							if math.Float32bits(replay[i]) != math.Float32bits(want[i]) {
								t.Fatal("reset replay output mismatch")
							}
						}
					}
					if e := op.Close(); e != nil {
						t.Fatal(e)
					}
					if e := op.Eval(out, dx); e == nil {
						t.Fatal("closed Eval admitted")
					}
					if e := op.Close(); e != nil {
						t.Fatal(e)
					}
				})
			}
		})
	}
}

func v41AlignedActivationReference(x []float32) ([]byte, []float32) {
	y := make([]byte, len(x)/32*36)
	scales := make([]float32, len(x)/32)
	for b := range scales {
		var values [32]float32
		peak := float32(0)
		for j := range values {
			values[j] = x[b*32+j]
			// This module and the pinned C literal both use --use_fast_math,
			// hence float32 subnormal inputs to reductions flush to signed0.
			// Mixed-sign fixture rows make the final zero sum positive.
			if math.Abs(float64(values[j])) < 0x1p-126 {
				values[j] = float32(math.Copysign(0, float64(values[j])))
			}
			peak = max(peak, float32(math.Abs(float64(values[j]))))
		}
		for step := 16; step > 0; step /= 2 {
			old := values
			for lane := range values {
				values[lane] = float32(old[lane] + old[lane^step])
			}
		}
		d := float32(peak / 127)
		scales[b] = d
		binary.LittleEndian.PutUint16(y[b*36:], v41AlignedHalfBits(d))
		binary.LittleEndian.PutUint16(y[b*36+2:], v41AlignedHalfBits(values[0]))
		for j := 0; j < 32; j++ {
			q := int8(0)
			if peak != 0 {
				q = int8(math.Round(float64(float32(x[b*32+j] / d))))
			}
			y[b*36+4+j] = byte(q)
		}
	}
	return y, scales
}

// Float64 arithmetic independently rounds IEEE binary16, avoiding the device
// helper/header and the main package's half conversion implementations.
func v41AlignedHalfBits(v float32) uint16 {
	sign := uint16(0)
	if math.Signbit(float64(v)) {
		sign = 0x8000
	}
	x := math.Abs(float64(v))
	if math.IsNaN(x) {
		return sign | 0x7e00
	}
	if math.IsInf(x, 0) || x >= 65520 {
		return sign | 0x7c00
	}
	if x == 0 {
		return sign
	}
	if x < math.Ldexp(1, -14) {
		return sign | uint16(math.RoundToEven(math.Ldexp(x, 24)))
	}
	e := math.Ilogb(x)
	q := int(math.RoundToEven(math.Ldexp(x, 10-e)))
	if q == 2048 {
		e++
		q = 1024
	}
	return sign | uint16(e+15)<<10 | uint16(q-1024)
}
func v41AlignedHalfValue(h uint16) float32 {
	e, m := int((h>>10)&31), int(h&1023)
	var x float64
	if e == 0 {
		x = math.Ldexp(float64(m), -24)
	} else {
		x = math.Ldexp(float64(1024+m), e-25)
	}
	if h&0x8000 != 0 {
		x = -x
	}
	return float32(x)
}

func v41AlignedOutputReference(raw, packed []byte, k, m int) ([]float32, []int32) {
	blocks := k / 32
	out, dots := make([]float32, m), make([]int32, m*blocks)
	for row := range out {
		var lane [32]float32
		for l := range lane {
			for b := l; b < blocks; b += 32 {
				p := (row*blocks + b) * 34
				dot := int32(0)
				for j := 0; j < 32; j++ {
					dot += int32(int8(raw[p+2+j])) * int32(int8(packed[b*36+4+j]))
				}
				dots[row*blocks+b] = dot
				product := float32(v41AlignedHalfValue(binary.LittleEndian.Uint16(raw[p:])) * v41AlignedHalfValue(binary.LittleEndian.Uint16(packed[b*36:])))
				// Fixture half scales are powers-of-two and canonical d is near1;
				// product*dot+acc has <53 significant bits, so this double FMA then
				// float32 conversion is an exact float32 FMA oracle for these inputs.
				lane[l] = float32(math.FMA(float64(product), float64(dot), float64(lane[l])))
			}
		}
		for step := 16; step > 0; step /= 2 {
			for l := 0; l < step; l++ {
				lane[l] += lane[l+step]
			}
		}
		out[row] = lane[0]
	}
	return out, dots
}
