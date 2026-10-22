//go:build linux

package main

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/examples/ds4-min/dsv4cpu"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestCUDAV41RawAttentionValuesAndMutants(t *testing.T) {
	if os.Getenv("DS4_CUDA_V41_OPS_GATE") != "1" {
		t.Skip("opt-in V4.1 CUDA gate")
	}
	if pid, err := strconv.Atoi(os.Getenv("GPU_LOCK_HELD")); err != nil || pid <= 0 {
		t.Fatal("gpu-lock required")
	}
	capture := newV41RawWorkspaceCapture(t)
	for _, variant := range []struct{ name, old, new string }{
		{"baseline", "", ""},
		{"window", "if(n>128)n=128;", "if(n>127)n=127;"},
		{"sink", "float denominator=float(exp(double(__fsub_rn(sinks[head],peak))));", "float denominator=0;"},
		{"scratch", "scores[row]=__fmul_rn(dot,scale);", "if(row!=0)scores[row]=__fmul_rn(dot,scale);"},
	} {
		if !t.Run(variant.name, func(t *testing.T) {
			r, resources := newV41RawTestResources(t)
			source := cudaV41RawAttentionSource
			if variant.old != "" {
				if strings.Count(source, variant.old) != 1 {
					t.Fatal("mutant site not unique")
				}
				source = strings.Replace(source, variant.old, variant.new, 1)
			}
			capture.saveBytes(t, "source", []byte(source))
			op, err := compileCUDAV41RawAttentionCaptured(r, 130, source, func(p *nvrtcx.CompiledModule) error { return capture.compiled(t.Name(), r, p) })
			if op != nil {
				resources.ownerClose = op.Close
			}
			if err != nil {
				resources.retainError(err)
				t.Fatal(err)
			}
			testV41RawScoreABI(t, op)
			q, kv, sinks := make([]float32, 64*512), make([]float32, 512), make([]float32, 64)
			for h := range sinks {
				sinks[h] = float32(h%7-3) / 4
			}
			dq, dk, ds, out := v41RawUpload(t, r, resources, q), v41RawUpload(t, r, resources, kv), v41RawUpload(t, r, resources, sinks), v41RawUpload(t, r, resources, q)
			selected := map[int]bool{0: true, 1: true, 64: true, 127: true, 128: true, 129: true}
			violations := 0
			var saved [][]uint32
			for repeat := 0; repeat < 2; repeat++ {
				if err := op.Reset(); err != nil {
					t.Fatal(err)
				}
				ring := make([]float32, 128*512)
				boundary := 0
				for pos := 0; pos < 130; pos++ {
					for i := range q {
						q[i] = dsv41.BF16(float32(math.Sin(float64(i*3+pos)*.031)) * .02)
					}
					for i := range kv {
						kv[i] = dsv41.BF16(float32(math.Cos(float64(i*7+pos*23)*.017))*.7 + float32(pos)/32)
					}
					wantQ, wantKV := append([]float32(nil), q...), append([]float32(nil), kv...)
					v41RawReferenceRope(wantQ, pos, false)
					v41RawReferenceRope(wantKV, pos, false)
					if err := dsv41.Quantize(wantKV, 512, dsv41.FP8E8M0); err != nil {
						t.Fatal(err)
					}
					copy(ring[(pos%128)*512:], wantKV)
					if err := dq.CopyFromHost(q); err != nil {
						t.Fatal(err)
					}
					if err := dk.CopyFromHost(kv); err != nil {
						t.Fatal(err)
					}
					poisonV41RawScores(t, op, repeat)
					if err := op.Append(out, dq, dk, ds, uint32(pos)); err != nil {
						t.Fatal(err)
					}
					if !op.valid || op.position != uint32(pos+1) {
						t.Fatal("frontier not committed")
					}
					if !selected[pos] {
						continue
					}
					n := min(pos+1, 128)
					rows := make([]float32, n*512)
					for i := 0; i < n; i++ {
						copy(rows[i*512:], ring[((pos+1-n+i)%128)*512:((pos+1-n+i)%128+1)*512])
					}
					want := make([]float32, len(q))
					if err := dsv4cpu.Attend(want, wantQ, rows, sinks, 512); err != nil {
						t.Fatal(err)
					}
					headsReference := append([]float32(nil), want...)
					if err := dsv41.Quantize(want, 512, dsv41.BFloat16); err != nil {
						t.Fatal(err)
					}
					v41RawReferenceRope(want, pos, true)
					got := make([]float32, len(q))
					if err := out.CopyToHost(got); err != nil {
						t.Fatal(err)
					}
					prefix := fmt.Sprintf("r%d-p%03d-", repeat, pos)
					capture.save(t, prefix+"q-input", q)
					capture.save(t, prefix+"kv-input", kv)
					capture.save(t, prefix+"sinks", sinks)
					capture.save(t, prefix+"reference", want)
					capture.save(t, prefix+"output", got)
					capture.save(t, prefix+"reference-heads", headsReference)
					checkV41RawScores(t, capture, op, prefix, wantQ, ring, sinks, pos, variant.name)
					captureV41RawReplay(t, capture, resources, op, out, ds, prefix, uint32(pos))
					bits := make([]uint32, len(got))
					local := 0
					for i := range got {
						bits[i] = math.Float32bits(got[i])
						if bits[i] != math.Float32bits(want[i]) {
							violations++
							local++
						}
					}
					if variant.name == "window" && pos < 127 && local != 0 {
						t.Fatal("window mutant altered adjacent pre-wrap case")
					}
					cache := make([]float32, len(ring))
					if err := op.ring.CopyToHost(cache); err != nil {
						t.Fatal(err)
					}
					capture.save(t, prefix+"ring", cache)
					for i := range cache {
						if math.Float32bits(cache[i]) != math.Float32bits(ring[i]) {
							t.Fatalf("raw cache pos=%d index=%d mismatch", pos, i)
						}
					}
					if variant.name == "baseline" || variant.name == "window" {
						if repeat == 0 {
							saved = append(saved, bits)
						} else {
							for i := range bits {
								if bits[i] != saved[boundary][i] {
									t.Fatal("baseline repeat differs")
								}
							}
						}
					}
					boundary++
					for _, v := range []struct {
						dev  cudamem.Slice[float32]
						want []float32
					}{{dq, q}, {dk, kv}, {ds, sinks}} {
						read := make([]float32, len(v.want))
						if err := v.dev.CopyToHost(read); err != nil {
							t.Fatal(err)
						}
						for i := range read {
							if math.Float32bits(read[i]) != math.Float32bits(v.want[i]) {
								t.Fatal("borrowed input mutated")
							}
						}
					}
				}
			}
			t.Logf("head value violations=%d; six boundary positions, all32768 values, two repeats", violations)
			if variant.name == "baseline" {
				if violations != 0 {
					t.Fatal("scalar device attention mismatch")
				}
				testV41RawAdmission(t, op, dq, dk, ds, out)
			} else if violations == 0 {
				t.Fatal("raw attention mutant survived")
			}
			// Adjacent negative for both mutants: at pos0 the ring has one row,
			// and sink=-1000 contributes exactly zero after underflow.
			if err := op.Reset(); err != nil {
				t.Fatal(err)
			}
			for i := range sinks {
				sinks[i] = -1000
			}
			if err := ds.CopyFromHost(sinks); err != nil {
				t.Fatal(err)
			}
			if err := op.Append(out, dq, dk, ds, 0); err != nil {
				t.Fatal(err)
			}
			wantKV := append([]float32(nil), kv...)
			v41RawReferenceRope(wantKV, 0, false)
			if err := dsv41.Quantize(wantKV, 512, dsv41.FP8E8M0); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(q))
			if err := out.CopyToHost(got); err != nil {
				t.Fatal(err)
			}
			capture.save(t, "zero-sink-output", got)
			for h := 0; h < 64; h++ {
				for d := 0; d < 512; d++ {
					if math.Float32bits(got[h*512+d]) != math.Float32bits(wantKV[d]) {
						t.Fatal("adjacent one-row zero-sink negative differs")
					}
				}
			}
			if err := op.Close(); err != nil {
				resources.err = err
				t.Fatal(err)
			}
			if err := op.Append(out, dq, dk, ds, 1); err == nil {
				t.Fatal("closed request admitted")
			}
			if err := op.Reset(); err == nil {
				t.Fatal("closed reset admitted")
			}
			if err := op.Close(); err != nil {
				resources.err = err
				t.Fatal(err)
			}
		}) {
			t.Fatal("raw variant failed; remaining variants not run")
		}
	}
}

func v41RawReferenceRope(x []float32, position int, inverse bool) {
	for lane := 0; lane < 32; lane++ {
		f := float32(1) / float32(math.Pow(10000, float64(float32(lane)/32)))
		theta := float32(float32(position) * f)
		c, s := float32(math.Cos(float64(theta))), float32(math.Sin(float64(theta)))
		if inverse {
			s = -s
		}
		for h := 0; h < len(x)/512; h++ {
			i := h*512 + 448 + lane*2
			a, b := x[i], x[i+1]
			x[i] = dsv41.BF16(float32(a*c) - float32(b*s))
			x[i+1] = dsv41.BF16(float32(a*s) + float32(b*c))
		}
	}
}

func testV41RawAdmission(t *testing.T, op *cudaV41RawAttention, q, kv, sinks, out cudamem.Slice[float32]) {
	t.Helper()
	if err := op.Append(out, q, kv, sinks, 130); err == nil {
		t.Fatal("context bound ignored")
	}
	if !op.valid || op.position != 130 {
		t.Fatal("context admission changed frontier")
	}
	if err := op.Reset(); err != nil {
		t.Fatal(err)
	}
	testV41RawWorkspaceAdmission(t, op, q, kv, sinks, out)
	if err := op.Append(out, q, kv, sinks, 1); err == nil {
		t.Fatal("frontier gap accepted")
	}
	if err := op.Append(q, q, kv, sinks, 0); err == nil {
		t.Fatal("overlap accepted")
	}
	var empty cudamem.Slice[float32]
	if err := op.Append(out, empty, kv, sinks, 0); err == nil {
		t.Fatal("missing query accepted")
	}
	if !op.valid || op.position != 0 {
		t.Fatal("admission changed frontier")
	}
	function := op.attend
	op.attend = 0
	err := op.Append(out, q, kv, sinks, 0)
	op.attend = function
	if err == nil || op.valid || op.position != 0 {
		t.Fatal("post-commit launch error not invalidated")
	}
	if len(op.pending) != 0 {
		t.Fatal("successful drain retained pending borrows")
	}
	if err := op.Append(out, q, kv, sinks, 0); err == nil {
		t.Fatal("invalid request reused")
	}
	if err := op.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := op.Append(out, q, kv, sinks, 0); err != nil {
		t.Fatal(err)
	}
	hq, hkv, hs := make([]float32, q.Len()), make([]float32, kv.Len()), make([]float32, sinks.Len())
	for _, v := range []struct {
		dev  cudamem.Slice[float32]
		host []float32
	}{{q, hq}, {kv, hkv}, {sinks, hs}} {
		if err := v.dev.CopyToHost(v.host); err != nil {
			t.Fatal(err)
		}
	}
	v41RawReferenceRope(hq, 0, false)
	v41RawReferenceRope(hkv, 0, false)
	if err := dsv41.Quantize(hkv, 512, dsv41.FP8E8M0); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, q.Len())
	if err := dsv4cpu.Attend(want, hq, hkv, hs, 512); err != nil {
		t.Fatal(err)
	}
	if err := dsv41.Quantize(want, 512, dsv41.BFloat16); err != nil {
		t.Fatal(err)
	}
	v41RawReferenceRope(want, 0, true)
	got := make([]float32, out.Len())
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("error-reset replay output differs at %d", i)
		}
	}
}
