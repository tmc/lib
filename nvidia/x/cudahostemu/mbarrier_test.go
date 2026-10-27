package cudahostemu

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"
)

// ringSrc is a warp-specialized ring: thread 0 copies STEPS stages of 128
// floats into a STAGES-deep ring with cp.async.bulk, and warps 1 and 2 each
// copy every stage to out after waiting on full[stage], then release it on
// empty[stage]. MUT selects a mutation of the protocol.
const ringSrc = `
#ifndef MUT
#define MUT 0
#endif
constexpr int STAGES = 2, STEPS = 6, ELEMS = 128;

__global__ void ring(const float* in, float* out) {
    float* ring = (float*)eg_emu_dynamic_smem();
    __shared__ uint64_t full[STAGES], empty[STAGES];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    if (tid == 0)
        for (int s = 0; s < STAGES; ++s) {
            eg_emu_mbar_init(&full[s], 1);
            eg_emu_mbar_init(&empty[s], 2);
        }
    __syncthreads();
    if (warp == 0) {
        for (int k = 0; tid == 0 && k < STEPS; ++k) {
            const int s = k % STAGES, n = k / STAGES;
#if MUT != 4  // 4: refill a stage without waiting for its release
            if (n > 0) eg_emu_mbar_wait(&empty[s], (n - 1) & 1);
#endif
#if MUT == 5  // 5: expect 16 bytes too few
            eg_emu_mbar_arrive_expect_tx(&full[s], ELEMS * 4 - 16);
#elif MUT == 6  // 6: expect 16 bytes too many
            eg_emu_mbar_arrive_expect_tx(&full[s], ELEMS * 4 + 16);
#else
            eg_emu_mbar_arrive_expect_tx(&full[s], ELEMS * 4);
#endif
            eg_emu_cp_async_bulk(ring + s * ELEMS, in + k * ELEMS, ELEMS * 2, &full[s]);
            eg_emu_cp_async_bulk(ring + s * ELEMS + ELEMS / 2, in + k * ELEMS + ELEMS / 2, ELEMS * 2, &full[s]);
        }
        return;
    }
    for (int k = 0; k < STEPS; ++k) {
        const int s = k % STAGES, n = k / STAGES;
#if MUT == 2  // 2: wait for the wrong phase parity
        eg_emu_mbar_wait(&full[s], (n & 1) ^ 1);
#elif MUT != 1  // 1: no wait
        eg_emu_mbar_wait(&full[s], n & 1);
#endif
        for (int i = lane; i < ELEMS; i += 32) out[((warp - 1) * STEPS + k) * ELEMS + i] = ring[s * ELEMS + i];
        __syncwarp();
#if MUT != 3  // 3: never release the stage
        if (lane == 0) eg_emu_mbar_arrive(&empty[s]);
#endif
    }
}
`

func ringInput() []byte {
	in := make([]byte, 6*128*4)
	for i := range 6 * 128 {
		binary.LittleEndian.PutUint32(in[4*i:], math.Float32bits(float32(i+1)))
	}
	return in
}

// runRing runs ringSrc with mutation mut and reports whether its output is
// correct, or the run's error.
func runRing(t *testing.T, mut int, flags ...string) (bool, error) {
	t.Helper()
	res, err := Run(context.Background(), t.TempDir(), Program{
		Source:  ringSrc,
		Flags:   append([]string{fmt.Sprintf("-DMUT=%d", mut)}, flags...),
		Launch:  "eg_emu_launch(dim3(1), dim3(96), 2 * 128 * 4, ring, in, out)",
		Buffers: []Buffer{{"in", "float", ringInput()}, {"out", "float", Poison(2 * 6 * 128 * 4)}},
	})
	if err != nil {
		return false, err
	}
	got := f32s(res["out"])
	for i, v := range got {
		if v != float32(i%(6*128)+1) {
			return false, nil
		}
	}
	return true, nil
}

func TestMBarrierRing(t *testing.T) {
	if _, err := Compiler(); err != nil {
		t.Skip(err)
	}
	for _, flags := range [][]string{nil, {"-DEG_EMU_SYNC_MBAR"}} {
		ok, err := runRing(t, 0, flags...)
		if err != nil || !ok {
			t.Errorf("flags %v: correct ring: ok %v, err %v", flags, ok, err)
		}
	}
}

// TestMBarrierMutations checks that each broken ring protocol fails under
// the adversarial model, by wrong output or a loud error, on every run.
func TestMBarrierMutations(t *testing.T) {
	if _, err := Compiler(); err != nil {
		t.Skip(err)
	}
	tests := []struct {
		name string
		mut  int
	}{
		{"no full wait", 1},
		{"wrong parity", 2},
		{"no empty arrive", 3},
		{"no empty wait", 4},
		{"expect_tx short", 5},
		{"expect_tx long", 6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for range 3 {
				ok, err := runRing(t, tt.mut)
				if ok {
					t.Fatal("mutant produced correct output")
				}
				t.Logf("adversarial: %s", outcome(ok, err))
			}
			ok, err := runRing(t, tt.mut, "-DEG_EMU_SYNC_MBAR")
			t.Logf("EG_EMU_SYNC_MBAR: %s", outcome(ok, err))
		})
	}
}

func TestMBarrierParityAndTestWait(t *testing.T) {
	const src = `
__global__ void k(const float* in, float* out, int* flags) {
    float* s = (float*)eg_emu_dynamic_smem();
    __shared__ uint64_t bar;
    const int tid = threadIdx.x;
    if (tid == 0) {
        eg_emu_mbar_init(&bar, 1);
        flags[0] = eg_emu_mbar_test_wait(&bar, 0);  // phase 0 incomplete
        flags[1] = eg_emu_mbar_test_wait(&bar, 1);  // the phase before 0 counts as complete
        eg_emu_mbar_arrive_expect_tx(&bar, 64);
        eg_emu_cp_async_bulk(s, in, 32, &bar);
    }
    __syncthreads();
    out[tid] = s[tid];          // no wait
    eg_emu_mbar_wait(&bar, 1);  // the wrong parity: returns at once
    out[16 + tid] = s[tid];
    __syncthreads();
    if (tid == 0) {
        eg_emu_cp_async_bulk(s + 8, in + 8, 32, &bar);  // completes phase 0
        flags[2] = eg_emu_mbar_test_wait(&bar, 0);
    }
    __syncthreads();
    out[32 + tid] = s[tid];
}
`
	in := make([]byte, 64)
	for i := range 16 {
		binary.LittleEndian.PutUint32(in[4*i:], math.Float32bits(float32(i+1)))
	}
	res := run(t, Program{
		Source:  src,
		Launch:  "eg_emu_launch(dim3(1), dim3(16), 64, k, in, out, flags)",
		Buffers: []Buffer{{"in", "float", in}, {"out", "float", Poison(48 * 4)}, {"flags", "int", Poison(12)}},
	})
	if f := u32s(res["flags"]); f[0] != 0 || f[1] != 1 || f[2] != 1 {
		t.Errorf("test_wait results %v, want [0 1 1]", f)
	}
	got := f32s(res["out"])
	for i := range 16 {
		if !math.IsNaN(float64(got[i])) || !math.IsNaN(float64(got[16+i])) {
			t.Errorf("s[%d] read %v without a wait and %v after a wrong-parity wait, want NaN", i, got[i], got[16+i])
		}
		if want := float32(i + 1); got[32+i] != want {
			t.Errorf("s[%d] = %v after test_wait, want %v", i, got[32+i], want)
		}
	}
}

func TestTensorMap2D(t *testing.T) {
	// A 10 x 70 BF16 tensor; boxes of 8 rows x 64 columns at (0, 0) and,
	// partly outside it, at (64, 8).
	const src = `
__global__ void k(const uint16_t* x, uint16_t* out) {
    unsigned char* s = eg_emu_dynamic_smem();
    __shared__ uint64_t bar;
    const eg_emu_tensor_map m = eg_emu_make_tensor_map_2d(x, 2, 70, 10, 70 * 2 + 4, 64, 8, true);
    if (threadIdx.x == 0) {
        eg_emu_mbar_init(&bar, 1);
        eg_emu_mbar_arrive_expect_tx(&bar, 2 * 1024);
        eg_emu_cp_async_bulk_tensor_2d(s, &m, 0, 0, &bar);
        eg_emu_cp_async_bulk_tensor_2d(s + 1024, &m, 64, 8, &bar);
    }
    __syncthreads();
    eg_emu_mbar_wait(&bar, 0);
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) out[i] = ((uint16_t*)s)[i];
}
`
	const rows, cols, pitch = 10, 70, 72 // pitch in elements: 70*2+4 bytes
	x := make([]byte, rows*pitch*2)
	for r := range rows {
		for c := range cols {
			binary.LittleEndian.PutUint16(x[2*(r*pitch+c):], uint16(1000*r+c+1))
		}
	}
	res := run(t, Program{
		Source:  src,
		Launch:  "eg_emu_launch(dim3(1), dim3(32), 2048, k, x, out)",
		Buffers: []Buffer{{"x", "uint16_t", x}, {"out", "uint16_t", Poison(2048)}},
	})
	got := res["out"]
	for b := range 2 {
		for r := range 8 {
			for c := range 64 {
				gr, gc := r+8*b, c+64*b
				var want uint16
				if gr < rows && gc < cols {
					want = uint16(1000*gr + gc + 1)
				}
				off := r*128 + c*2
				off ^= ((off >> 7) & 7) << 4
				if v := binary.LittleEndian.Uint16(got[1024*b+off:]); v != want {
					t.Fatalf("box %d (%d, %d) = %d, want %d", b, r, c, v, want)
				}
			}
		}
	}
}

func TestMBarrierErrors(t *testing.T) {
	if _, err := Compiler(); err != nil {
		t.Skip(err)
	}
	tests := []struct {
		name string
		body string
		want string
	}{
		{"parity 2", "eg_emu_mbar_init(&b, 1); eg_emu_mbar_wait(&b, 2);", "not 0 or 1"},
		{"uninitialized wait", "eg_emu_mbar_wait(&b, 0);", "uninitialized mbarrier"},
		{"uninitialized arrive", "eg_emu_mbar_arrive(&b);", "uninitialized mbarrier"},
		{"uninitialized copy", "eg_emu_cp_async_bulk(eg_emu_dynamic_smem(), x, 16, &b);", "uninitialized mbarrier"},
		{"zero count", "eg_emu_mbar_init(&b, 0);", "bad barrier address or count"},
		{"extra arrival", "eg_emu_mbar_init(&b, 1); eg_emu_mbar_arrive_expect_tx(&b, 16); eg_emu_mbar_arrive(&b);", "more than 1 arrivals"},
		{"copy size", "eg_emu_mbar_init(&b, 1); eg_emu_cp_async_bulk(eg_emu_dynamic_smem(), x, 8, &b);", "multiple of 16"},
		{"copy alignment", "eg_emu_mbar_init(&b, 1); eg_emu_cp_async_bulk(eg_emu_dynamic_smem() + 4, x, 16, &b);", "multiple of 16"},
		{"waits forever", "eg_emu_mbar_init(&b, 2); eg_emu_mbar_arrive(&b); eg_emu_mbar_wait(&b, 0);", "deadlock"},
		{"bytes outstanding", "eg_emu_mbar_init(&b, 1); eg_emu_mbar_arrive_expect_tx(&b, 32); eg_emu_cp_async_bulk(eg_emu_dynamic_smem(), x, 16, &b);", "16 transaction bytes outstanding"},
		{"copy in flight", "eg_emu_mbar_init(&b, 2); eg_emu_cp_async_bulk(eg_emu_dynamic_smem(), x, 16, &b); eg_emu_mbar_arrive_expect_tx(&b, 16);", "incomplete"},
		{"swizzled box misaligned", "eg_emu_mbar_init(&b, 1); auto m = eg_emu_make_tensor_map_2d(x, 2, 64, 8, 128, 64, 8, true); eg_emu_cp_async_bulk_tensor_2d(eg_emu_dynamic_smem() + 128, &m, 0, 0, &b);", "misaligned"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := fmt.Sprintf("__global__ void k(const uint32_t* x) { __shared__ uint64_t b; if (threadIdx.x == 0) { %s } }", tt.body)
			_, err := Run(context.Background(), t.TempDir(), Program{
				Source:  src,
				Launch:  "eg_emu_launch(dim3(1), dim3(32), 2048, k, x)",
				Buffers: []Buffer{{"x", "uint32_t", make([]byte, 1024)}},
			})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %v, want %q", err, tt.want)
			}
		})
	}
}

func outcome(ok bool, err error) string {
	switch {
	case err != nil:
		msg := err.Error()
		if i := strings.Index(msg, "cudahostemu: "); i >= 0 {
			msg = msg[i:]
		}
		if i := strings.IndexByte(msg, '\n'); i >= 0 {
			msg = msg[:i]
		}
		return "error: " + msg
	case ok:
		return "correct output"
	}
	return "wrong output"
}
