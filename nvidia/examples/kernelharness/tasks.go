package main

import (
	"fmt"
	"math"
	"sort"
)

// KernelABI is the fixed signature every candidate must define. It is fixed so
// the harness can pack launch arguments without parsing the candidate, and so a
// candidate that does not match it is rejected at the launch, as a classified
// launch_rejected or missing_entry rather than as a wrong answer.
const KernelABI = `extern "C" __global__ void ENTRY(float *out, const float *a, const float *b, float alpha, int n)`

// Task states one problem. Everything in a Task is written here, from the
// problem's definition. No field is derived from a candidate, and no candidate
// can read one: the oracle is not the producer.
type Task struct {
	Name    string
	Entry   string
	Summary string
	// N is the element count. Reference and Compare both run over exactly N
	// elements; a candidate that writes fewer leaves the untouched tail at its
	// initialized value, which the comparison sees.
	N int
	// Alpha is the scalar parameter passed to the kernel.
	Alpha float32
	// Grid and Block are chosen by the harness, not by the candidate, so two
	// candidates are timed under the same launch geometry.
	Grid, Block uint32
	Tolerance   Tolerance
	// Reference computes the expected output on the CPU. It accumulates in
	// float64 and narrows once, so it does not share the candidate's rounding.
	Reference func(a, b []float32, alpha float32) []float32
	// ReferenceDoc names where the reference comes from, and is copied into
	// every report so the reader never has to take the independence on trust.
	ReferenceDoc string
	// Baseline is a readable, untuned CUDA implementation of the same task,
	// used only as the performance comparison arm. It is never used to compute
	// an expected value: it is judged by the same CPU reference the candidate is.
	Baseline string
	// BytesPerLaunch is the minimum traffic the task requires: two input
	// vectors read and one output vector written. It is a property of the
	// problem, not a measurement of what a candidate actually moved.
	BytesPerLaunch int64
}

// genInputs builds one trial's host inputs. Every correctness trial uses a
// different seed, so a kernel cannot pass by being right on one draw. The
// generator is a plain integer recurrence rather than math/rand so a seed
// reproduces its data exactly across Go versions and platforms.
func genInputs(seed int64, n int) (a, b []float32) {
	a, b = make([]float32, n), make([]float32, n)
	s := uint64(seed)*6364136223846793005 + 1442695040888963407
	next := func() float32 {
		s = s*6364136223846793005 + 1442695040888963407
		// Take the high bits; the low bits of an LCG have short periods.
		return float32(int64(s>>40)%2001-1000) / 512
	}
	for i := range a {
		a[i] = next()
		b[i] = next()
	}
	return a, b
}

func axpbyReference(a, b []float32, alpha float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = float32(float64(alpha)*float64(a[i]) + float64(b[i]))
	}
	return out
}

func reluAxpbyReference(a, b []float32, alpha float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		v := float64(alpha)*float64(a[i]) + float64(b[i])
		if v < 0 {
			v = 0
		}
		out[i] = float32(v)
	}
	return out
}

func rsqrtScaleReference(a, b []float32, alpha float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		out[i] = float32(x / math.Sqrt(y*y+float64(alpha)))
	}
	return out
}

const axpbyBaseline = `
extern "C" __global__ void axpby(float *out, const float *a, const float *b, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) out[i] = alpha * a[i] + b[i];
}
`

const reluAxpbyBaseline = `
extern "C" __global__ void relu_axpby(float *out, const float *a, const float *b, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) {
    float v = alpha * a[i] + b[i];
    out[i] = v > 0.0f ? v : 0.0f;
  }
}
`

const rsqrtScaleBaseline = `
extern "C" __global__ void rsqrt_scale(float *out, const float *a, const float *b, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) out[i] = a[i] / sqrtf(b[i] * b[i] + alpha);
}
`

// tasks is the built-in task set. The tolerances are loose enough to admit a
// different accumulation order and tight enough to reject a wrong formula:
// each was checked against the deliberately wrong kernels in testdata/.
var tasks = map[string]*Task{
	"axpby": {
		Name: "axpby", Entry: "axpby", Summary: "out[i] = alpha*a[i] + b[i]",
		N: 1 << 20, Alpha: 2.5, Grid: 1024, Block: 256,
		Tolerance:      Tolerance{Abs: 1e-6, Rel: 1e-6},
		Reference:      axpbyReference,
		ReferenceDoc:   "axpbyReference in examples/kernelharness/tasks.go: float64 accumulation on the CPU, narrowed once",
		Baseline:       axpbyBaseline,
		BytesPerLaunch: 3 * 4 * (1 << 20),
	},
	"relu_axpby": {
		Name: "relu_axpby", Entry: "relu_axpby", Summary: "out[i] = max(alpha*a[i] + b[i], 0)",
		N: 1 << 20, Alpha: 1.75, Grid: 1024, Block: 256,
		Tolerance:      Tolerance{Abs: 1e-6, Rel: 1e-6},
		Reference:      reluAxpbyReference,
		ReferenceDoc:   "reluAxpbyReference in examples/kernelharness/tasks.go: float64 accumulation on the CPU, narrowed once",
		Baseline:       reluAxpbyBaseline,
		BytesPerLaunch: 3 * 4 * (1 << 20),
	},
	"rsqrt_scale": {
		Name: "rsqrt_scale", Entry: "rsqrt_scale", Summary: "out[i] = a[i] / sqrt(b[i]*b[i] + alpha)",
		N: 1 << 20, Alpha: 1e-3, Grid: 1024, Block: 256,
		// A reciprocal square root is where a candidate is most likely to reach
		// for a fast approximation. 2e-3 relative admits rsqrtf and the
		// -use_fast_math form; it does not admit a wrong expression.
		Tolerance:      Tolerance{Abs: 1e-5, Rel: 2e-3},
		Reference:      rsqrtScaleReference,
		ReferenceDoc:   "rsqrtScaleReference in examples/kernelharness/tasks.go: float64 math.Sqrt on the CPU, narrowed once",
		Baseline:       rsqrtScaleBaseline,
		BytesPerLaunch: 3 * 4 * (1 << 20),
	},
}

// TaskNames lists the built-in tasks in a stable order.
func TaskNames() []string {
	names := make([]string, 0, len(tasks))
	for name := range tasks {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// LookupTask returns the named built-in task.
func LookupTask(name string) (*Task, error) {
	t, ok := tasks[name]
	if !ok {
		return nil, fmt.Errorf("unknown task %q; known tasks: %v", name, TaskNames())
	}
	return t, nil
}
