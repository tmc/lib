package cudahostemu

import (
	"context"
	"encoding/binary"
	"errors"
	"strings"
	"testing"
)

func TestCLCHeader(t *testing.T) {
	hdr := CLCHeader()
	for _, want := range []string{
		"eg_emu_clc_try_cancel_async",
		"eg_emu_clc_wait",
		"eg_emu_clc_query_is_canceled",
		"eg_emu_clc_query_first_ctaid_x",
		"eg_emu_clc_error_count",
	} {
		if !strings.Contains(hdr, want) {
			t.Errorf("CLCHeader() missing %q", want)
		}
	}
}

func TestCLCWorkStealing(t *testing.T) {
	const (
		activeCTAs = 4
		totalTiles = 19
	)

	tests := []struct {
		name string
		src  string
	}{
		{
			name: "eg_emu_clc_wait",
			src: `
__global__ void clc_steal(uint32_t* visited, uint32_t* values, int total_tiles) {
    __shared__ uint4 resp;
    __shared__ uint64_t mbar;
    __shared__ int next_tile;
    if (threadIdx.x == 0) {
        eg_emu_mbar_init(&mbar, 1);
    }
    __syncthreads();

    int phase = 0;
    for (int tile = (int)blockIdx.x; tile < total_tiles; ) {
        if (threadIdx.x == 0) {
            atomicAdd(&visited[tile], 1u);
            values[tile] = (uint32_t)(tile * 100 + 7);
            eg_emu_clc_try_cancel_async(&resp, &mbar);
            eg_emu_clc_wait(&mbar, phase);
            phase ^= 1;
            if (eg_emu_clc_query_is_canceled(resp)) {
                next_tile = (int)eg_emu_clc_query_first_ctaid_x(resp);
            } else {
                next_tile = total_tiles;
            }
        }
        __syncthreads();
        tile = next_tile;
    }
}`,
		},
		{
			name: "eg_emu_mbar_wait",
			src: `
__global__ void clc_steal_mbar(uint32_t* visited, uint32_t* values, int total_tiles) {
    __shared__ uint4 resp;
    __shared__ uint64_t mbar;
    __shared__ int next_tile;
    if (threadIdx.x == 0) {
        eg_emu_mbar_init(&mbar, 1);
    }
    __syncthreads();

    int phase = 0;
    for (int tile = (int)blockIdx.x; tile < total_tiles; ) {
        if (threadIdx.x == 0) {
            atomicAdd(&visited[tile], 1u);
            values[tile] = (uint32_t)(tile * 100 + 7);
            eg_emu_mbar_arrive_expect_tx(&mbar, 16);
            eg_emu_clc_try_cancel_async(&resp, &mbar);
        }
        eg_emu_mbar_wait(&mbar, phase);
        phase ^= 1;
        if (threadIdx.x == 0) {
            if (eg_emu_clc_query_is_canceled(resp)) {
                next_tile = (int)eg_emu_clc_query_first_ctaid_x(resp);
            } else {
                next_tile = total_tiles;
            }
        }
        __syncthreads();
        tile = next_tile;
    }
}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			visited := make([]byte, 4*totalTiles)
			values := Poison(4 * totalTiles)

			err := RunWithCLC(
				context.Background(),
				tt.src,
				activeCTAs,
				totalTiles,
				Dim3{X: 32},
				0,
				BufferArg("visited", "uint32_t", visited),
				BufferArg("values", "uint32_t", values),
				IntArg(totalTiles),
			)
			if errors.Is(err, ErrNoCompiler) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatalf("RunWithCLC: %v", err)
			}

			for tile := 0; tile < totalTiles; tile++ {
				gotCount := binary.LittleEndian.Uint32(visited[4*tile : 4*(tile+1)])
				if gotCount != 1 {
					t.Errorf("tile %d visited %d times, want 1", tile, gotCount)
				}
				gotVal := binary.LittleEndian.Uint32(values[4*tile : 4*(tile+1)])
				wantVal := uint32(tile*100 + 7)
				if gotVal != wantVal {
					t.Errorf("tile %d value = %d, want %d", tile, gotVal, wantVal)
				}
			}
		})
	}
}

func TestCLCHazardBeforeMbarWait(t *testing.T) {
	tests := []struct {
		name    string
		src     string
		wantSub string
	}{
		{
			name: "query_is_canceled before mbarrier wait",
			src: `
__global__ void clc_bad_is_canceled(uint32_t* out) {
    __shared__ uint4 resp;
    __shared__ uint64_t mbar;
    if (threadIdx.x == 0) {
        eg_emu_mbar_init(&mbar, 1);
        eg_emu_clc_try_cancel_async(&resp, &mbar);
        // Missing eg_emu_clc_wait / eg_emu_mbar_wait before query!
        if (eg_emu_clc_query_is_canceled(resp)) {
            out[0] = 1u;
        }
    }
}`,
			wantSub: "eg_emu_clc_query_is_canceled",
		},
		{
			name: "query_first_ctaid_x before mbarrier wait",
			src: `
__global__ void clc_bad_first_ctaid(uint32_t* out) {
    __shared__ uint4 resp;
    __shared__ uint64_t mbar;
    if (threadIdx.x == 0) {
        eg_emu_mbar_init(&mbar, 1);
        eg_emu_clc_try_cancel_async(&resp, &mbar);
        // Missing eg_emu_clc_wait / eg_emu_mbar_wait before query!
        out[0] = eg_emu_clc_query_first_ctaid_x(resp);
    }
}`,
			wantSub: "eg_emu_clc_query_first_ctaid_x",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := make([]byte, 16)
			err := RunWithCLC(
				context.Background(),
				tt.src,
				4,
				19,
				Dim3{X: 32},
				0,
				BufferArg("out", "uint32_t", out),
			)
			if errors.Is(err, ErrNoCompiler) {
				t.Skip(err)
			}
			if err == nil {
				t.Fatalf("RunWithCLC succeeded; want CLC hazard error containing %q", tt.wantSub)
			}
			if !strings.Contains(err.Error(), "clc hazard") || !strings.Contains(err.Error(), tt.wantSub) {
				t.Fatalf("RunWithCLC error = %v; want error containing \"clc hazard\" and %q", err, tt.wantSub)
			}
		})
	}
}
