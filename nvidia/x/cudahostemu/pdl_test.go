package cudahostemu

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"
)

func pdlF32Bytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func pdlBytesF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func TestPDLChainPositiveAndNegative(t *testing.T) {
	if _, err := Compiler(); errors.Is(err, ErrNoCompiler) {
		t.Skip(err)
	}

	const n = 64
	inVals := make([]float32, n)
	for i := range inVals {
		inVals[i] = float32(i + 1)
	}

	tests := []struct {
		name          string
		src           string
		wantViolation bool
		wantNaN       bool
	}{
		{
			name: "positive producer triggers and consumer waits before reading",
			src: `
__global__ void producer(const float* in, float* buf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    buf[idx] = in[idx] * 2.0f + 1.0f;
    __syncthreads();
    eg_pdl_trigger(1);
}

__global__ void consumer(const float* buf, float* out) {
    eg_pdl_wait(1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = buf[idx] + 10.0f;
}
`,
			wantViolation: false,
			wantNaN:       false,
		},
		{
			name: "negative consumer reads buffer without calling eg_pdl_wait",
			src: `
__global__ void producer(const float* in, float* buf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    buf[idx] = in[idx] * 2.0f + 1.0f;
    __syncthreads();
    eg_pdl_trigger(1);
}

__global__ void consumer(const float* buf, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = buf[idx] + 10.0f;
}
`,
			wantViolation: true,
			wantNaN:       true,
		},
		{
			name: "negative consumer reads buffer before calling eg_pdl_wait",
			src: `
__global__ void producer(const float* in, float* buf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    buf[idx] = in[idx] * 2.0f + 1.0f;
    __syncthreads();
    eg_pdl_trigger(1);
}

__global__ void consumer(const float* buf, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float premature = buf[idx];
    eg_pdl_wait(1);
    out[idx] = premature + 10.0f;
}
`,
			wantViolation: true,
			wantNaN:       true,
		},
		{
			name: "negative producer never calls eg_pdl_trigger",
			src: `
__global__ void producer(const float* in, float* buf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    buf[idx] = in[idx] * 2.0f + 1.0f;
}

__global__ void consumer(const float* buf, float* out) {
    eg_pdl_wait(1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = buf[idx] + 10.0f;
}
`,
			wantViolation: true,
			wantNaN:       true,
		},
	}

	stages := []PDLStage{
		{
			Kernel:  "producer",
			Grid:    Dim3{X: 2},
			Block:   Dim3{X: 32},
			StageID: 1,
			Buffer:  "buf",
			Args:    []string{"in", "buf"},
		},
		{
			Kernel:  "consumer",
			Grid:    Dim3{X: 2},
			Block:   Dim3{X: 32},
			StageID: 1,
			Args:    []string{"buf", "out"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inBuf := pdlF32Bytes(inVals)
			midBuf := make([]byte, 4*n)
			outBuf := Poison(4 * n)

			err := RunPDLChain(context.Background(), tt.src, stages,
				BufferArg("in", "float", inBuf),
				BufferArg("buf", "float", midBuf),
				BufferArg("out", "float", outBuf),
			)
			if tt.wantViolation {
				if err == nil {
					t.Fatalf("RunPDLChain succeeded, want PDL ordering violation error")
				}
				var vErr *PDLViolationError
				if !errors.As(err, &vErr) || len(vErr.Violations) == 0 {
					t.Fatalf("RunPDLChain error = %v, want *PDLViolationError", err)
				}
			} else if err != nil {
				t.Fatalf("RunPDLChain unexpected error: %v", err)
			}

			got := pdlBytesF32(outBuf)
			if tt.wantNaN {
				for i, v := range got {
					if !math.IsNaN(float64(v)) {
						t.Fatalf("out[%d] = %g, want poisoned NaN", i, v)
					}
				}
				return
			}

			for i, v := range got {
				want := inVals[i]*2.0 + 1.0 + 10.0
				if v != want {
					t.Fatalf("out[%d] = %g, want %g", i, v, want)
				}
			}
		})
	}
}

func TestPDLHeaderDirectPrimitives(t *testing.T) {
	if _, err := Compiler(); errors.Is(err, ErrNoCompiler) {
		t.Skip(err)
	}
	hdr := PDLHeader()
	for _, want := range []string{
		"eg_emu_pdl_register_buffer",
		"eg_emu_pdl_launch_dependents",
		"eg_emu_pdl_wait",
		"eg_emu_pdl_violation_count",
	} {
		if !strings.Contains(hdr, want) {
			t.Fatalf("PDLHeader() missing %q", want)
		}
	}

	src := hdr + `
__global__ void prod_k(float* buf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    buf[i] = (float)(i + 1) * 3.0f;
    eg_emu_pdl_launch_dependents(1);
}

__global__ void cons_k(const float* buf, float* out, int* violations) {
    eg_emu_pdl_wait(1);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = buf[i] - 1.0f;
    if (i == 0) {
        violations[0] = eg_emu_pdl_violation_count();
    }
}
`
	const n = 32
	launch := strings.Join([]string{
		"eg_emu_pdl_register_buffer(buf, 32 * sizeof(float), 1)",
		"eg_emu_launch(dim3(1), dim3(32), 0, prod_k, buf)",
		"eg_emu_launch(dim3(1), dim3(32), 0, cons_k, buf, out, violations)",
	}, ";\n    ")

	res, err := Run(context.Background(), t.TempDir(), Program{
		Source: src,
		Launch: launch,
		Buffers: []Buffer{
			{Name: "buf", CType: "float", Data: make([]byte, 4*n)},
			{Name: "out", CType: "float", Data: Poison(4 * n)},
			{Name: "violations", CType: "int", Data: Poison(4)},
		},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if v := binary.LittleEndian.Uint32(res["violations"]); v != 0 {
		t.Fatalf("violation_count = %d, want 0", v)
	}
	for i, got := range pdlBytesF32(res["out"]) {
		want := float32(i+1)*3.0 - 1.0
		if got != want {
			t.Fatalf("out[%d] = %g, want %g", i, got, want)
		}
	}
}
