package cudahostemu

import (
	"context"
	"encoding/binary"
	"errors"
	"slices"
	"strings"
	"testing"
)

func condI32Bytes(v []int32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return b
}

func condBytesI32(b []byte) []int32 {
	v := make([]int32, len(b)/4)
	for i := range v {
		v[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func TestConditionalWhileLoop(t *testing.T) {
	if _, err := Compiler(); errors.Is(err, ErrNoCompiler) {
		t.Skip(err)
	}

	hdr := CondGraphHeader()
	for _, sym := range []string{
		"cudaGraphConditionalHandle",
		"cudaGraphSetConditional",
		"eg_condgraph_set",
		"eg_condgraph_get",
		"eg_condgraph_update_count",
	} {
		if !strings.Contains(hdr, sym) {
			t.Fatalf("CondGraphHeader() missing %q", sym)
		}
	}

	const loopSrc = `
__global__ void step_accumulate(int32_t* state, int32_t* history) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int32_t step = state[0];
        int32_t acc = state[1] + (step + 1) * 3;
        history[step] = acc;
        state[0] = step + 1;
        state[1] = acc;
    }
}

__global__ void step_controller(
    const int32_t* state,
    int32_t target_step,
    cudaGraphConditionalHandle cond_handle) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int keep = (state[0] < target_step) ? 1u : 0u;
        cudaGraphSetConditional(cond_handle, keep);
    }
}
`

	body := []CondGraphStep{
		{
			Kernel: "step_accumulate",
			Grid:   Dim3{X: 1},
			Block:  Dim3{X: 32},
			Args:   []string{"state", "history"},
		},
		{
			Kernel: "step_controller",
			Grid:   Dim3{X: 1},
			Block:  Dim3{X: 32},
			Args:   []string{"state", "target_step", "cond_handle"},
		},
	}

	t.Run("multi_iteration_convergence", func(t *testing.T) {
		stateBuf := condI32Bytes([]int32{0, 0})
		histBuf := condI32Bytes(make([]int32, 8))

		res, err := RunConditionalWhile(
			context.Background(),
			loopSrc,
			CondWhileConfig{
				Handle:                1,
				InitialCondition:      1,
				MaxIterations:         16,
				RequireExplicitUpdate: true,
				DetectDataRace:        true,
			},
			body,
			BufferArg("state", "int32_t", stateBuf),
			BufferArg("history", "int32_t", histBuf),
			Arg{Name: "target_step", Expr: "5"},
			CondHandleArg("cond_handle", 1),
		)
		if err != nil {
			t.Fatalf("RunConditionalWhile: %v", err)
		}
		if res.IterationsExecuted != 5 {
			t.Fatalf("IterationsExecuted = %d, want 5", res.IterationsExecuted)
		}
		if res.FinalCondition != 0 {
			t.Fatalf("FinalCondition = %d, want 0", res.FinalCondition)
		}
		if res.HandleUpdateCount != 5 {
			t.Fatalf("HandleUpdateCount = %d, want 5", res.HandleUpdateCount)
		}
		wantTrace := []uint32{1, 1, 1, 1, 0}
		if !slices.Equal(res.IterationTrace, wantTrace) {
			t.Fatalf("IterationTrace = %v, want %v", res.IterationTrace, wantTrace)
		}
		gotState := condBytesI32(stateBuf)
		if !slices.Equal(gotState, []int32{5, 45}) {
			t.Fatalf("state = %v, want [5 45]", gotState)
		}
		gotHist := condBytesI32(histBuf)[:5]
		wantHist := []int32{3, 9, 18, 30, 45}
		if !slices.Equal(gotHist, wantHist) {
			t.Fatalf("history = %v, want %v", gotHist, wantHist)
		}
	})

	t.Run("zero_iteration_skip_when_initial_condition_zero", func(t *testing.T) {
		stateBuf := condI32Bytes([]int32{0, 77})
		histBuf := condI32Bytes(make([]int32, 4))

		res, err := RunConditionalWhile(
			context.Background(),
			loopSrc,
			CondWhileConfig{
				Handle:                1,
				InitialCondition:      0,
				MaxIterations:         8,
				RequireExplicitUpdate: true,
			},
			body,
			BufferArg("state", "int32_t", stateBuf),
			BufferArg("history", "int32_t", histBuf),
			Arg{Name: "target_step", Expr: "4"},
			CondHandleArg("cond_handle", 1),
		)
		if err != nil {
			t.Fatalf("RunConditionalWhile: %v", err)
		}
		if res.IterationsExecuted != 0 {
			t.Fatalf("IterationsExecuted = %d, want 0", res.IterationsExecuted)
		}
		if res.FinalCondition != 0 {
			t.Fatalf("FinalCondition = %d, want 0", res.FinalCondition)
		}
		if res.HandleUpdateCount != 0 {
			t.Fatalf("HandleUpdateCount = %d, want 0", res.HandleUpdateCount)
		}
		if got := condBytesI32(stateBuf); !slices.Equal(got, []int32{0, 77}) {
			t.Fatalf("state modified on skipped loop: got %v, want [0 77]", got)
		}
	})

	t.Run("runaway_loop_exceeds_max_iterations", func(t *testing.T) {
		stateBuf := condI32Bytes([]int32{0, 0})
		histBuf := condI32Bytes(make([]int32, 16))

		res, err := RunConditionalWhile(
			context.Background(),
			loopSrc,
			CondWhileConfig{
				Handle:                1,
				InitialCondition:      1,
				MaxIterations:         4,
				RequireExplicitUpdate: true,
			},
			body,
			BufferArg("state", "int32_t", stateBuf),
			BufferArg("history", "int32_t", histBuf),
			Arg{Name: "target_step", Expr: "100"},
			CondHandleArg("cond_handle", 1),
		)
		if !errors.Is(err, ErrCondGraphRunawayLoop) {
			t.Fatalf("err = %v, want ErrCondGraphRunawayLoop", err)
		}
		if res == nil || res.IterationsExecuted != 4 || res.FinalCondition != 1 {
			t.Fatalf("res = %+v, want IterationsExecuted=4 FinalCondition=1", res)
		}
	})

	t.Run("missing_explicit_handle_update_trips_runaway_guard", func(t *testing.T) {
		const noUpdateSrc = `
__global__ void bad_body(int32_t* count, cudaGraphConditionalHandle cond_handle) {
    (void)cond_handle;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        count[0] += 1;
    }
}
`
		countBuf := condI32Bytes([]int32{0})
		res, err := RunConditionalWhile(
			context.Background(),
			noUpdateSrc,
			CondWhileConfig{
				Handle:                1,
				InitialCondition:      1,
				MaxIterations:         8,
				RequireExplicitUpdate: true,
			},
			[]CondGraphStep{{Kernel: "bad_body", Grid: Dim3{X: 1}, Block: Dim3{X: 32}}},
			BufferArg("count", "int32_t", countBuf),
		)
		if !errors.Is(err, ErrCondGraphRunawayLoop) {
			t.Fatalf("err = %v, want ErrCondGraphRunawayLoop", err)
		}
		if res == nil || res.IterationsExecuted != 1 || res.HandleUpdateCount != 0 {
			t.Fatalf("res = %+v, want IterationsExecuted=1 HandleUpdateCount=0", res)
		}
	})

	t.Run("detect_data_race_on_multi_thread_set_conditional", func(t *testing.T) {
		const raceSrc = `
__global__ void racy_controller(int32_t* out, cudaGraphConditionalHandle cond_handle) {
    // Bug: all threads in the CTA write to the conditional handle without leader election.
    out[threadIdx.x] = (int32_t)threadIdx.x;
    cudaGraphSetConditional(cond_handle, 0u);
}
`
		outBuf := condI32Bytes(make([]int32, 32))
		_, err := RunConditionalWhile(
			context.Background(),
			raceSrc,
			CondWhileConfig{
				Handle:           1,
				InitialCondition: 1,
				MaxIterations:    4,
				DetectDataRace:   true,
			},
			[]CondGraphStep{{Kernel: "racy_controller", Grid: Dim3{X: 1}, Block: Dim3{X: 32}}},
			BufferArg("out", "int32_t", outBuf),
		)
		if !errors.Is(err, ErrCondGraphDataRace) {
			t.Fatalf("err = %v, want ErrCondGraphDataRace", err)
		}
	})
}
