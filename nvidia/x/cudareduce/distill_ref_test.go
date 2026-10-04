package cudareduce

import (
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

type distillFixture struct {
	Name       string
	Config     SparseTeacherConfig
	Z, W       []float32
	IDs, Masks []int32
}

func distillFixtures() []distillFixture {
	var fixtures []distillFixture
	for _, shape := range []struct {
		name       string
		rows, v, k int
		mode       SparseTeacherObjective
		t          float32
	}{
		{"normalized_irregular", 4, 129, 16, NormalizedSupport, 2.5},
		{"normalized_outside", 2, 7, 2, NormalizedSupport, 0.7},
		{"normalized_singleton", 1, 1, 1, NormalizedSupport, 1},
		{"normalized_full", 2, 8, 8, NormalizedSupport, 1},
		{"normalized_zero_weight", 2, 17, 3, NormalizedSupport, 3},
		{"tail_irregular", 4, 129, 16, PreservedTail, 1},
		{"tail_underflow", 2, 7, 2, PreservedTail, 1},
		{"tail_zero_other", 2, 17, 2, PreservedTail, 1},
		{"tail_full", 2, 8, 8, PreservedTail, 1},
		{"all_masked", 3, 33, 4, PreservedTail, 1},
	} {
		c := SparseTeacherConfig{shape.rows, shape.v, shape.k, shape.mode, shape.t}
		e, _ := c.Extents()
		f := distillFixture{Name: shape.name, Config: c, Z: make([]float32, e.Logits), W: make([]float32, e.Targets), IDs: make([]int32, e.Targets), Masks: make([]int32, e.Rows)}
		for i := range f.Z {
			f.Z[i] = float32((i*31+7)%47-23) * 0.17
		}
		for row := 0; row < c.Rows; row++ {
			f.Masks[row] = 1
			for i := 0; i < c.Retained; i++ {
				f.IDs[row*c.Retained+i] = int32((i * 7) % c.Vocabulary)
				if c.Vocabulary%7 == 0 {
					f.IDs[row*c.Retained+i] = int32(i)
				}
				if shape.mode == PreservedTail {
					f.W[row*c.Retained+i] = 0.5 / float32(c.Retained)
				} else {
					f.W[row*c.Retained+i] = float32(i + 1)
				}
			}
		}
		if shape.name == "tail_full" {
			for i := range f.W {
				f.W[i] = 0.125
			}
		}
		if shape.name == "normalized_zero_weight" {
			f.W[0] = 0
		}
		if shape.name == "tail_zero_other" {
			for row := 0; row < c.Rows; row++ {
				f.W[row*c.Retained] = 0.75
				f.W[row*c.Retained+1] = 0.25
			}
		}
		if shape.name == "tail_underflow" {
			for row := 0; row < c.Rows; row++ {
				for j := 0; j < c.Vocabulary; j++ {
					f.Z[row*c.Vocabulary+j] = -1000 + float32(j)
				}
				f.Z[row*c.Vocabulary] = 1000
			}
		}
		if shape.name == "normalized_outside" {
			f.Z[c.Vocabulary-1] = 20
		}
		if shape.name == "all_masked" {
			for i := range f.Masks {
				f.Masks[i] = 0
			}
		} else if c.Rows == 4 {
			f.Masks[1] = 0
		}
		fixtures = append(fixtures, f)
	}
	return fixtures
}

// The reference builds a dense FP64 teacher distribution or a k+1 categorical
// distribution. It normalizes materialized exponentials and differentiates
// the bucket mapping independently of the CUDA recurrence/reductions.
func distillReference(f distillFixture) (losses, grad []float64, total float64) {
	c := f.Config
	losses = make([]float64, c.Rows)
	grad = make([]float64, len(f.Z))
	for row := 0; row < c.Rows; row++ {
		if f.Masks[row] == 0 {
			continue
		}
		z := f.Z[row*c.Vocabulary : (row+1)*c.Vocabulary]
		p := make([]float64, c.Vocabulary)
		marked := make([]bool, c.Vocabulary)
		mass := 0.0
		for i := 0; i < c.Retained; i++ {
			id := int(f.IDs[row*c.Retained+i])
			p[id] = float64(f.W[row*c.Retained+i])
			marked[id] = true
			mass += p[id]
		}
		maxZ := math.Inf(-1)
		for _, x := range z {
			maxZ = math.Max(maxZ, float64(x)/float64(c.Temperature))
		}
		ex := make([]float64, len(z))
		den := 0.0
		for j, x := range z {
			ex[j] = math.Exp(float64(x)/float64(c.Temperature) - maxZ)
			den += ex[j]
		}
		logQ := make([]float64, len(z))
		for j, x := range z {
			logQ[j] = float64(x)/float64(c.Temperature) - maxZ - math.Log(den)
		}
		if c.Objective == NormalizedSupport {
			for j := range p {
				p[j] /= mass
				if p[j] > 0 {
					losses[row] += p[j] * (math.Log(p[j]) - logQ[j])
				}
				grad[row*c.Vocabulary+j] = (ex[j]/den - p[j]) / float64(c.Temperature)
			}
		} else {
			// Independent omitted categorical normalization survives full-vocab q
			// underflow, including when the largest retained token dominates by 2000.
			omitted := make([]int, 0)
			omax := math.Inf(-1)
			for j, x := range z {
				if !marked[j] {
					omitted = append(omitted, j)
					omax = math.Max(omax, float64(x))
				}
			}
			oe := make([]float64, len(omitted))
			od := 0.0
			for i, j := range omitted {
				oe[i] = math.Exp(float64(z[j]) - omax)
				od += oe[i]
			}
			for j := range p {
				if marked[j] && p[j] > 0 {
					losses[row] += p[j] * (math.Log(p[j]) - logQ[j])
				}
				grad[row*c.Vocabulary+j] = ex[j]/den - p[j]
			}
			tail := 1 - mass
			if tail > 0 {
				logBucket := omax - maxZ + math.Log(od) - math.Log(den)
				losses[row] += tail * (math.Log(tail) - logBucket)
				for i, j := range omitted {
					grad[row*c.Vocabulary+j] -= tail * oe[i] / od
				}
			}
		}
		total += losses[row]
	}
	return
}
func distillCompare(f distillFixture, target string, actual []float32, want []float64) error {
	expected := make([]float32, len(want))
	for i, x := range want {
		expected[i] = float32(x)
	}
	return cudatest.CompareCUDAOutputFP32(actual, expected, cudatest.CUDAOutputPolicy{Target: target, DType: "fp32", MathMode: "fp32 fmad=false ftz=false; coarse-tail only T=1", SourceDigest: distillDigest([]byte(SparseTeacherSource)), InputDigest: distillInputDigest(f), Elements: len(want), Absolute: 3e-5, Relative: 4e-5, NaN: "reject", Infinity: "reject", Zero: "equal"})
}
func distillInputDigest(f distillFixture) string {
	data := fmt.Appendf(nil, "v1/%d/%d/%d/%d/%08x/", f.Config.Rows, f.Config.Vocabulary, f.Config.Retained, f.Config.Objective, math.Float32bits(f.Config.Temperature))
	for _, s := range [][]float32{f.Z, f.W} {
		data = fmt.Appendf(data, "%d:", len(s))
		for _, x := range s {
			data = fmt.Appendf(data, "%08x", math.Float32bits(x))
		}
	}
	for _, s := range [][]int32{f.IDs, f.Masks} {
		data = fmt.Appendf(data, "%d:", len(s))
		for _, x := range s {
			data = fmt.Appendf(data, "%08x", uint32(x))
		}
	}
	return distillDigest(data)
}
func distillCheckReference(t *testing.T, f distillFixture, result distillHostResult) {
	t.Helper()
	loss, grad, total := distillReference(f)
	if err := distillCompare(f, "host genuine CUDA control", result.Losses, loss); err != nil {
		t.Fatal(err)
	}
	if err := distillCompare(f, "host genuine CUDA control", result.Gradients, grad); err != nil {
		t.Fatal(err)
	}
	if err := distillCompare(f, "host genuine CUDA control", []float32{result.Total}, []float64{total}); err != nil {
		t.Fatal(err)
	}
}
