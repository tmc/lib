package cudahostemu

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func run(t *testing.T, p Program) map[string][]byte {
	t.Helper()
	out, err := Run(context.Background(), t.TempDir(), p)
	if errors.Is(err, ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func zeros(n int) []byte { return make([]byte, n) }

func u32s(b []byte) []uint32 {
	out := make([]uint32, len(b)/4)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return out
}

func f32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func TestWarpCollectives(t *testing.T) {
	const src = `
__global__ void k(int* out) {
    int v = threadIdx.x;
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(~0u, v, m);
    out[threadIdx.x * 4 + 0] = v;
    out[threadIdx.x * 4 + 1] = (int)__ballot_sync(~0u, threadIdx.x & 1);
    out[threadIdx.x * 4 + 2] = __shfl_up_sync(~0u, (int)threadIdx.x, 1);
    out[threadIdx.x * 4 + 3] = __shfl_down_sync(~0u, (int)threadIdx.x, 1);
}`
	got := u32s(run(t, Program{Source: src, Launch: "eg_emu_launch(dim3(1), dim3(64), 0, k, out)",
		Buffers: []Buffer{{"out", "int", zeros(64 * 16)}}})["out"])
	for i := 0; i < 64; i++ {
		lane, sum := i%32, uint32(496)
		if i >= 32 {
			sum = 1520
		}
		up, down := i-1, i+1
		if lane == 0 {
			up = i
		}
		if lane == 31 {
			down = i
		}
		want := []uint32{sum, 0xaaaaaaaa, uint32(up), uint32(down)}
		for j, w := range want {
			if got[4*i+j] != w {
				t.Errorf("thread %d value %d = %#x, want %#x", i, j, got[4*i+j], w)
			}
		}
	}
}

// TestPartialWarpsAndEarlyExit checks that a block need not be a multiple
// of 32 threads and that threads returning before a barrier do not hold it.
func TestPartialWarpsAndEarlyExit(t *testing.T) {
	const src = `
__global__ void k(unsigned* out) {
    if (threadIdx.x >= 33) return;
    __syncthreads();
    __syncwarp();
    out[threadIdx.x] = __shfl_sync(~0u, threadIdx.x + 100, 0);
}`
	got := u32s(run(t, Program{Source: src, Launch: "eg_emu_launch(dim3(1), dim3(40), 0, k, out)",
		Buffers: []Buffer{{"out", "unsigned", zeros(40 * 4)}}})["out"])
	for i, v := range got {
		want := uint32(100)
		switch {
		case i == 32:
			want = 132
		case i > 32:
			want = 0
		}
		if v != want {
			t.Errorf("out[%d] = %d, want %d", i, v, want)
		}
	}
}

// TestSharedMemoryStartsPoisoned checks that each block starts with static
// and dynamic shared memory holding 0xff bytes, whatever an earlier block
// wrote, and zeros under EG_EMU_ZERO_SMEM.
func TestSharedMemoryStartsPoisoned(t *testing.T) {
	const src = `
__global__ void k(uint32_t* out) {
    __shared__ uint32_t s[4];
    uint32_t* d = (uint32_t*)eg_emu_dynamic_smem();
    if (threadIdx.x == 0) {
        out[blockIdx.x * 2] = s[blockIdx.x];
        out[blockIdx.x * 2 + 1] = d[blockIdx.x];
        s[blockIdx.x + 1] = 7;
        d[blockIdx.x + 1] = 7;
    }
}`
	for _, tt := range []struct {
		flags []string
		env   string // CUDAHOSTEMU_FLAGS
		want  uint32
	}{
		{nil, "", 0xffffffff},
		{[]string{"-DEG_EMU_ZERO_SMEM"}, "", 0},
		{nil, "-DEG_EMU_ZERO_SMEM", 0},
	} {
		t.Setenv("CUDAHOSTEMU_FLAGS", tt.env)
		got := u32s(run(t, Program{Source: src, Flags: tt.flags, Launch: "eg_emu_launch(dim3(3), dim3(32), 64, k, out)",
			Buffers: []Buffer{{"out", "uint32_t", zeros(24)}}})["out"])
		for i, v := range got {
			if v != tt.want {
				t.Errorf("%v %q: out[%d] = %#x, want %#x", tt.flags, tt.env, i, v, tt.want)
			}
		}
	}
}

// TestCpAsyncLandsAtWait checks that cp.async data is visible only after
// the wait that retires its group.
func TestCpAsyncLandsAtWait(t *testing.T) {
	const src = `
__global__ void k(const float* in, float* out) {
    float* s = (float*)eg_emu_dynamic_smem();
    eg_emu_cp_async(s + 4 * threadIdx.x, in + 4 * threadIdx.x, 16);
    eg_emu_cp_async_commit();
    out[threadIdx.x * 2] = s[4 * threadIdx.x];
    eg_emu_cp_async_wait(0);
    out[threadIdx.x * 2 + 1] = s[4 * threadIdx.x];
}`
	in := make([]byte, 32*16)
	for i := 0; i < 32*4; i++ {
		binary.LittleEndian.PutUint32(in[4*i:], math.Float32bits(float32(i)))
	}
	got := f32s(run(t, Program{Source: src, Launch: "eg_emu_launch(dim3(1), dim3(32), 512, k, in, out)",
		Buffers: []Buffer{{"in", "float", in}, {"out", "float", Poison(32 * 8)}}})["out"])
	for i := 0; i < 32; i++ {
		if !math.IsNaN(float64(got[2*i])) || got[2*i+1] != float32(4*i) {
			t.Errorf("thread %d read %v before wait and %v after, want NaN and %d", i, got[2*i], got[2*i+1], 4*i)
		}
	}
}

func bf16(f float32) uint16 {
	u := math.Float32bits(f)
	u += 0x7fff + (u>>16)&1
	return uint16(u >> 16)
}

func fromBF16(h uint16) float64 { return float64(math.Float32frombits(uint32(h) << 16)) }

// TestMMABF16 builds m16n8k16 fragments as the PTX ISA lays them out and
// checks the product against a float64 matrix multiply.
func TestMMABF16(t *testing.T) {
	const src = `
__global__ void k(const uint16_t* A, const uint16_t* B, float* D) {
    const int lane = threadIdx.x, g = lane / 4, t = lane % 4;
    auto pk = [](uint16_t lo, uint16_t hi) { return (uint32_t)lo | ((uint32_t)hi << 16); };
    uint32_t a[4], b[2];
    for (int i = 0; i < 4; ++i) {
        int row = g + 8 * (i % 2), col = 2 * t + 8 * (i / 2);
        a[i] = pk(A[row * 16 + col], A[row * 16 + col + 1]);
    }
    for (int i = 0; i < 2; ++i) {
        int k = 2 * t + 8 * i;
        b[i] = pk(B[k * 8 + g], B[(k + 1) * 8 + g]);
    }
    float c[4] = {0, 0, 0, 0}, d[4];
    eg_emu_mma_bf16_16816(d, a, b, c);
    for (int i = 0; i < 4; ++i) D[(g + 8 * (i / 2)) * 8 + 2 * t + (i & 1)] = d[i];
}`
	r := rand.New(rand.NewPCG(1, 2))
	a, b := make([]uint16, 16*16), make([]uint16, 16*8)
	enc := func(h []uint16) []byte {
		out := make([]byte, 2*len(h))
		for i, v := range h {
			binary.LittleEndian.PutUint16(out[2*i:], v)
		}
		return out
	}
	for i := range a {
		a[i] = bf16(float32(r.NormFloat64()))
	}
	for i := range b {
		b[i] = bf16(float32(r.NormFloat64()))
	}
	got := f32s(run(t, Program{Source: src, Launch: "eg_emu_launch(dim3(1), dim3(32), 0, k, A, B, D)",
		Buffers: []Buffer{{"A", "uint16_t", enc(a)}, {"B", "uint16_t", enc(b)}, {"D", "float", zeros(16 * 8 * 4)}}})["D"])
	for i := 0; i < 16; i++ {
		for j := 0; j < 8; j++ {
			var want float64
			for k := 0; k < 16; k++ {
				want += fromBF16(a[i*16+k]) * fromBF16(b[k*8+j])
			}
			if d := math.Abs(float64(got[i*8+j]) - want); d > 1e-4*(1+math.Abs(want)) {
				t.Errorf("D[%d][%d] = %v, want %v", i, j, got[i*8+j], want)
			}
		}
	}
}

// TestCluster checks per-block shared memory and map_shared_rank across
// the blocks of a cluster, for keyed and dynamic shared memory.
func TestCluster(t *testing.T) {
	const src = `
__global__ void k(int* out) {
    MOE8_SHARED(int, s, [2]);
    int* d = (int*)eg_emu_dynamic_smem();
    auto cl = cooperative_groups::this_cluster();
    if (threadIdx.x == 0) {
        s[1] = 10 * blockIdx.x;
        d[3] = 100 * blockIdx.x;
    }
    cl.sync();
    const int next = (cl.block_rank() + 1) % cl.num_blocks();
    if (threadIdx.x == 0) {
        out[blockIdx.x * 3] = *cl.map_shared_rank(&s[1], next);
        out[blockIdx.x * 3 + 1] = cl.map_shared_rank(d, next)[3];
        out[blockIdx.x * 3 + 2] = s[0];
    }
    cl.sync();
}`
	got := u32s(run(t, Program{Source: src, Launch: "eg_emu_launch_cluster(dim3(8), dim3(64), 16, 4, k, out)",
		Buffers: []Buffer{{"out", "int", zeros(8 * 12)}}})["out"])
	for bx := 0; bx < 8; bx++ {
		next := bx/4*4 + (bx%4+1)%4
		want := []uint32{uint32(10 * next), uint32(100 * next), 0}
		for j, w := range want {
			if got[bx*3+j] != w {
				t.Errorf("block %d value %d = %d, want %d", bx, j, got[bx*3+j], w)
			}
		}
	}
}

func TestRuntime(t *testing.T) {
	const src = `
__global__ void fill(float* p) { p[threadIdx.x] = threadIdx.x + 0.5f; }
__global__ void mark(float* p) { p[threadIdx.x] = -1; }

void host(int* res) {
    float* d;
    res[0] = cudaMalloc(&d, 8 * sizeof(float));
    float before[8], after[8];
    cudaMemcpy(before, d, sizeof before, cudaMemcpyDeviceToHost);
    res[1] = std::isnan(before[7]);
    CUDAHOSTEMU_LAUNCH(fill, dim3(1), dim3(4), 0, nullptr, d);
    res[2] = cudaGetLastError();
    cudaMemcpy(after, d, sizeof after, cudaMemcpyDeviceToHost);
    res[3] = after[3] == 3.5f && std::isnan(after[4]);
    // Three blocks do not divide into clusters of two: nothing may run.
    eg_emu_launch_cluster(dim3(3), dim3(4), 0, 2, mark, d);
    res[4] = cudaGetLastError();
    res[5] = cudaGetLastError() + (d[0] != 0.5f);
    res[6] = cudaMemset(d, 0, 8 * sizeof(float)) == cudaSuccess && d[7] == 0.0f;
    cudaFree(d);
}`
	got := u32s(run(t, Program{Source: src, Launch: "host(res)", Buffers: []Buffer{{"res", "int", zeros(7 * 4)}}})["res"])
	want := []uint32{0, 1, 0, 1, 9, 0, 1}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("res[%d] = %d, want %d (all: %v)", i, got[i], w, got)
		}
	}
}

// TestDeviceQueries checks that launchers sizing grids from the device see
// the emulated device.
func TestDeviceQueries(t *testing.T) {
	const src = `
__global__ void k() {}
void host(int* res) {
    eg_emu_sms = 7;
    eg_emu_blocks_per_sm = 3;
    int dev = -1, sms = 0, per = 0;
    res[0] = cudaGetDevice(&dev) + dev;
    res[1] = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) + sms;
    res[2] = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 128, 0) + per;
    eg_emu_launch(dim3(sms * per, 2), dim3(32), 0, k);
    res[3] = eg_emu_last_grid.x * 10 + eg_emu_last_grid.y;
    run_cluster_grid(dim3(4, 5), dim3(32), 2, [] {});
    res[4] = eg_emu_last_grid.x * 10 + eg_emu_last_grid.y;
}`
	got := u32s(run(t, Program{Source: src, Launch: "host(res)", Buffers: []Buffer{{"res", "int", Poison(20)}}})["res"])
	want := []uint32{0, 7, 3, 212, 45}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("res = %v, want %v", got, want)
			break
		}
	}
}

func TestSGLangMock(t *testing.T) {
	const src = `
__global__ void scale(float* p) { p[blockIdx.x * 32 + threadIdx.x] *= 2; }

void host(int* res, float* data) {
    using namespace sglang::host;
    tvm::ffi::TensorView v{data, {2, 32}, {32, 1}, {kDLFloat, 32, 1}, {kDLCUDA, 0}};
    SymbolicSize rows{"rows"};
    SymbolicDevice dev;
    dev.set_options<kDLCUDA>();
    TensorMatcher({rows, 32}).with_dtype<float>().with_device(dev).verify(v);
    res[0] = (int)rows.unwrap();
    try {
        TensorMatcher({rows, 32}).with_dtype<__nv_bfloat16>().verify(v);
    } catch (const std::runtime_error& e) {
        res[1] = std::string(e.what()).find("dtype") != std::string::npos;
    }
    LaunchKernel(dim3(2), dim3(32), dev.unwrap())(scale, data);
}`
	data := make([]byte, 64*4)
	for i := 0; i < 64; i++ {
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(float32(i)))
	}
	out := run(t, Program{Source: src, Mock: true, Launch: "host(res, data)",
		Buffers: []Buffer{{"res", "int", zeros(8)}, {"data", "float", data}}})
	if res := u32s(out["res"]); res[0] != 2 || res[1] != 1 {
		t.Errorf("res = %v, want [2 1]", res)
	}
	for i, v := range f32s(out["data"]) {
		if v != float32(2*i) {
			t.Fatalf("data[%d] = %v, want %d", i, v, 2*i)
		}
	}
}

// TestE4M3MatchesMoe8ops checks that the merged e4m3 conversion agrees with
// the one moe8ops's shim.h carried, at every rounding tie and boundary and
// on a sample of all floats.
func TestE4M3MatchesMoe8ops(t *testing.T) {
	const src = `
static uint8_t old_e4m3(float f) {
    uint32_t bits; std::memcpy(&bits, &f, 4);
    uint8_t sign = (uint8_t)((bits >> 24) & 0x80u);
    uint32_t a = bits & 0x7fffffffu;
    if (a > 0x7f800000u) return sign | 0x7f;
    float af = std::fabs(f);
    if (af >= 448.0f) return sign | 0x7e;
    if (af < 0.015625f) {
        int m = (int)std::nearbyint(af * 512.0f);
        return sign | (uint8_t)(m >= 8 ? 8 : m);
    }
    int e = (int)((a >> 23) & 0xff) - 127 + 7;
    uint32_t m = (a & 0x7fffffu) >> 20, rem = a & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) { if (++m == 8) { m = 0; ++e; } }
    if (e > 15 || (e == 15 && m == 7)) return sign | 0x7e;
    return sign | (uint8_t)((e << 3) | m);
}

void host(uint32_t* res) {
    auto check = [&](uint32_t u) {
        ++res[0];
        if (e4m3(__uint_as_float(u)) != old_e4m3(__uint_as_float(u))) ++res[1];
    };
    for (uint64_t u = 0; u < (1ull << 32); u += 97) check((uint32_t)u);
    const uint32_t lows[] = {0, 1, 0x7ffff, 0x80000, 0x80001, 0xfffff};
    for (uint32_t hi = 0; hi < (1u << 12); ++hi)
        for (uint32_t lo : lows) check(hi << 20 | lo);
}`
	res := u32s(run(t, Program{Source: src, Launch: "host(res)", Buffers: []Buffer{{"res", "uint32_t", zeros(8)}}})["res"])
	if res[0] < 1<<25 || res[1] != 0 {
		t.Errorf("checked %d values, %d differ", res[0], res[1])
	}
}

func TestRunErrors(t *testing.T) {
	_, err := Run(context.Background(), t.TempDir(), Program{Source: "int f() { return nope; }", Launch: "f()"})
	if errors.Is(err, ErrNoCompiler) {
		t.Skip(err)
	}
	if err == nil || !strings.Contains(err.Error(), "source:1") {
		t.Errorf("compile error = %v, want it located in source:1", err)
	}
	_, err = Run(context.Background(), t.TempDir(), Program{Source: `void f() { run_grid(dim3(1), dim3(4096), [] {}); }`, Launch: "f()"})
	if err == nil || !strings.Contains(err.Error(), "run_grid: invalid launch configuration") {
		t.Errorf("run error = %v", err)
	}
}

func TestPoison(t *testing.T) {
	p := Poison(4)
	if len(p) != 4 {
		t.Fatalf("len(Poison(4)) = %d", len(p))
	}
	if f := f32s(p)[0]; !math.IsNaN(float64(f)) {
		t.Errorf("Poison as float32 = %v, want NaN", f)
	}
	if b := fromBF16(binary.LittleEndian.Uint16(p)); !math.IsNaN(b) {
		t.Errorf("Poison as bf16 = %v, want NaN", b)
	}
}

// wgmmaSrc is a warpgroup computing D = A1·B1 + A2·B2 (m64n8, two k16
// steps) from shared memory, then writing D. STEP runs between issue and
// the final wait.
const wgmmaSrc = `
__global__ void k(const uint16_t* A, const uint16_t* B, float* D, int mode) {
    unsigned char* s = eg_emu_dynamic_smem();
    // K-major, no swizzle, LBO 128 (8x8 core matrices), SBO 256: A step j
    // at 4096*j, B step j at 2048 + 4096*j.
    auto put = [&](size_t base, const uint16_t* m, int rows, int ld, int k0) {
        for (int r = threadIdx.x; r < rows; r += blockDim.x)
            for (int k = 0; k < 16; ++k)
                std::memcpy(s + base + (r / 8) * 256 + (k / 8) * 128 + (r % 8) * 16 + (k % 8) * 2, &m[r * ld + k0 + k], 2);
    };
    for (int j = 0; j < 2; ++j) {
        put(4096 * j, A, 64, 32, 16 * j);
        put(2048 + 4096 * j, B, 8, 32, 16 * j);
    }
    __syncthreads();
    auto desc = [&](size_t off) { return (uint64_t)(off >> 4) | (uint64_t)(128 >> 4) << 16 | (uint64_t)(256 >> 4) << 32; };
    float d[4] = {0, 0, 0, 0};
    eg_emu_wgmma_fence();
    eg_emu_wgmma_bf16(d, desc(0), desc(2048), 8);
    if (mode == 2) eg_emu_wgmma_commit();
    eg_emu_wgmma_bf16(d, desc(4096), desc(6144), 8);
    if (mode != 5) eg_emu_wgmma_commit();
    const int wig = threadIdx.x / 32, lane = threadIdx.x % 32;
    auto store = [&](float* out) {
        for (int i = 0; i < 4; ++i) out[(wig * 16 + lane / 4 + 8 * (i / 2)) * 8 + 2 * (lane % 4) + (i & 1)] = d[i];
    };
    if (mode == 1) store(D + 512);                   // read before the wait
    if (mode == 2) { eg_emu_wgmma_wait(1); store(D + 512); }  // first group only: chain still pending
    if (mode == 5) { eg_emu_wgmma_wait(0); store(D + 512); eg_emu_wgmma_commit(); }  // wait retires no uncommitted op
    if (mode == 3) { __syncthreads(); std::memset(s + 6144, 0, 256); __syncthreads(); }  // B2 overwritten in flight
    if (mode != 4) eg_emu_wgmma_wait(0);
    store(D);
}`

func wgmmaInputs() (a, b []uint16, want []float64) {
	r := rand.New(rand.NewPCG(3, 4))
	a, b = make([]uint16, 64*32), make([]uint16, 8*32)
	for i := range a {
		a[i] = bf16(float32(r.NormFloat64()))
	}
	for i := range b {
		b[i] = bf16(float32(r.NormFloat64()))
	}
	want = make([]float64, 64*8)
	for m := 0; m < 64; m++ {
		for n := 0; n < 8; n++ {
			for k := 0; k < 32; k++ {
				want[m*8+n] += fromBF16(a[m*32+k]) * fromBF16(b[n*32+k])
			}
		}
	}
	return a, b, want
}

func runWGMMA(t *testing.T, mode int, flags ...string) ([]float32, error) {
	t.Helper()
	a, b, _ := wgmmaInputs()
	enc := func(h []uint16) []byte {
		out := make([]byte, 2*len(h))
		for i, v := range h {
			binary.LittleEndian.PutUint16(out[2*i:], v)
		}
		return out
	}
	out, err := Run(context.Background(), t.TempDir(), Program{Source: wgmmaSrc, Flags: flags,
		Launch:  fmt.Sprintf("eg_emu_launch(dim3(1), dim3(128), 8192, k, A, B, D, %d)", mode),
		Buffers: []Buffer{{"A", "uint16_t", enc(a)}, {"B", "uint16_t", enc(b)}, {"D", "float", Poison(2 * 512 * 4)}}})
	if errors.Is(err, ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		return nil, err
	}
	return f32s(out["D"]), nil
}

func near(got float32, want float64) bool {
	return math.Abs(float64(got)-want) <= 1e-4*(1+math.Abs(want))
}

// TestWGMMAAsync checks the async wgmma model: a correct fence, commit,
// wait sequence gives the product; the accumulator reads NaN until the
// wait that retires its last op; operands are read at the wait.
func TestWGMMAAsync(t *testing.T) {
	_, _, want := wgmmaInputs()
	for _, tt := range []struct {
		name  string
		mode  int
		flags []string
		early string // what the early read (D+512) must see: "", "nan" or "done"
		final string // "done" or "wrong"
	}{
		{"protocol", 0, nil, "", "done"},
		{"read before wait", 1, nil, "nan", "done"},
		{"chain across groups", 2, nil, "nan", "done"},
		{"uncommitted ops stay pending", 5, nil, "nan", "done"},
		{"operand overwritten in flight", 3, nil, "", "wrong"},
		{"sync control", 1, []string{"-DEG_EMU_SYNC_WGMMA"}, "done", "done"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			d, err := runWGMMA(t, tt.mode, tt.flags...)
			if err != nil {
				t.Fatal(err)
			}
			wrong := 0
			for i, w := range want {
				if !near(d[i], w) {
					wrong++
				}
				switch e := d[512+i]; tt.early {
				case "nan":
					if !math.IsNaN(float64(e)) {
						t.Fatalf("early read D[%d] = %v, want NaN", i, e)
					}
				case "done":
					if !near(e, w) {
						t.Fatalf("early read D[%d] = %v, want %v", i, e, w)
					}
				}
			}
			switch {
			case tt.final == "done" && wrong > 0:
				t.Errorf("%d of 512 results wrong", wrong)
			case tt.final == "wrong" && wrong == 0:
				t.Errorf("all results right; an operand overwritten before the wait must change them")
			}
		})
	}
}

func TestWGMMAMissingWait(t *testing.T) {
	_, err := runWGMMA(t, 4)
	if err == nil || !strings.Contains(err.Error(), "wgmma ops pending (no wgmma.wait_group)") {
		t.Errorf("run error = %v, want a pending-wgmma error", err)
	}
}
