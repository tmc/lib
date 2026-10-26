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

// TestClusterRegisteredKernelAndEGShared checks that a kernel registered with
// eg_emu_register_cluster_kernel is automatically launched as a cluster by
// eg_emu_launch, and that EG_EMU_SHARED, eg_emu_cluster_sync, and
// eg_emu_map_shared_rank work across cluster ranks.
func TestClusterRegisteredKernelAndEGShared(t *testing.T) {
	const src = `
__global__ void ck(uint32_t* out) {
    EG_EMU_SHARED(uint32_t, s, [2]);
    if (threadIdx.x == 0) {
        out[blockIdx.x * 2 + 0] = s[0];  // starts poisoned (0xffffffff)
        s[1] = 100 + blockIdx.x;
    }
    eg_emu_cluster_sync();
    if (threadIdx.x == 0) {
        int partner = (blockIdx.x % 8) ^ 1;
        out[blockIdx.x * 2 + 1] = *eg_emu_map_shared_rank(&s[1], partner);
    }
    eg_emu_cluster_sync();
}
static const bool reg_ck = eg_emu_register_cluster_kernel(ck);
`
	got := u32s(run(t, Program{
		Source:  src,
		Launch:  "eg_emu_launch(dim3(8, 2), dim3(32), 0, ck, out)",
		Buffers: []Buffer{{"out", "uint32_t", zeros(16 * 8)}},
	})["out"])
	for by := 0; by < 2; by++ {
		for bx := 0; bx < 8; bx++ {
			idx := by*8 + bx
			// Wait: in ck above, blockIdx.x is 0..7 for both by=0 and by=1; let's check bx=0..7:
			_ = idx
		}
	}
	for bx := 0; bx < 8; bx++ {
		if got[bx*2+0] != 0xffffffff {
			t.Errorf("block %d initial EG_EMU_SHARED = %#x, want 0xffffffff", bx, got[bx*2+0])
		}
		wantPartner := uint32(100 + (bx ^ 1))
		if got[bx*2+1] != wantPartner {
			t.Errorf("block %d partner value = %d, want %d", bx, got[bx*2+1], wantPartner)
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

// TestNamedBarriers checks that named barriers synchronize subsets of a CTA
// (such as producer/consumer warpgroups and consumer-internal barriers) without
// deadlocking non-participating warps, and fail loudly on invalid counts or
// incomplete barriers.
func TestNamedBarriers(t *testing.T) {
	const src = `
__global__ void prod_cons(uint32_t* out) {
    uint32_t* s = (uint32_t*)eg_emu_dynamic_smem();
    const int tid = threadIdx.x;
    if (tid < 64) {
        // Producer warpgroup (64 threads): write stage 0, signal barrier 1 (128 threads),
        // wait for consumer ack on barrier 2 (128 threads), then write stage 1.
        s[tid] = tid + 100;
        eg_emu_bar_sync(1, 128);
        eg_emu_bar_sync(2, 128);
        s[tid] = tid + 200;
        eg_emu_bar_sync(1, 128);
    } else if (tid < 192) {
        // Consumer warpgroups (128 threads total, 64 active in stage handoff + 64 in internal reduce).
        if (tid < 128) {
            eg_emu_bar_sync(1, 128);
            s[64 + (tid - 64)] = s[tid - 64] * 2;
        }
        // All 128 consumer threads (64..191) synchronize on barrier 3 without producer (0..63) or idle (192..255).
        bar_sync(3, 128);
        if (tid >= 128) {
            out[tid - 128] = s[64 + (tid - 128)];
        }
        bar_sync(3, 128);
        if (tid < 128) {
            eg_emu_bar_sync(2, 128);
            eg_emu_bar_sync(1, 128);
            out[64 + (tid - 64)] = s[tid - 64] * 3;
        }
    } else {
        // Threads 192..255 do independent work and exit without touching barriers 1, 2, or 3.
        out[128 + (tid - 192)] = tid + 1000;
    }
}`
	got := u32s(run(t, Program{
		Source:  src,
		Launch:  "eg_emu_launch(dim3(1), dim3(256), 1024, prod_cons, out)",
		Buffers: []Buffer{{"out", "uint32_t", Poison(192 * 4)}},
	})["out"])
	for i := 0; i < 64; i++ {
		if want := uint32((i + 100) * 2); got[i] != want {
			t.Fatalf("stage 0 out[%d] = %d, want %d", i, got[i], want)
		}
		if want := uint32((i + 200) * 3); got[64+i] != want {
			t.Fatalf("stage 1 out[%d] = %d, want %d", 64+i, got[64+i], want)
		}
		if want := uint32(192 + i + 1000); got[128+i] != want {
			t.Fatalf("independent out[%d] = %d, want %d", 128+i, got[128+i], want)
		}
	}

	for _, tt := range []struct {
		name string
		body string
		want string
	}{
		{
			name: "non-multiple of 32 count",
			body: "eg_emu_bar_sync(1, 48);",
			want: "invalid barrier id or thread count",
		},
		{
			name: "count exceeds block size",
			body: "eg_emu_bar_sync(1, 128);",
			want: "invalid barrier id or thread count",
		},
		{
			name: "invalid barrier id",
			body: "eg_emu_bar_sync(16, 32);",
			want: "invalid barrier id or thread count",
		},
		{
			name: "count mismatch",
			body: "eg_emu_bar_sync(2, threadIdx.x < 16 ? 32 : 64);",
			want: "count mismatch",
		},
		{
			name: "incomplete barrier on exit",
			body: "if (threadIdx.x < 32) eg_emu_bar_sync(4, 64);",
			want: "named barrier 4 incomplete",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			errSrc := fmt.Sprintf("__global__ void k() { %s }", tt.body)
			_, err := Run(context.Background(), t.TempDir(), Program{
				Source: errSrc,
				Launch: "eg_emu_launch(dim3(1), dim3(64), 0, k)",
			})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %v, want substring %q", err, tt.want)
			}
		})
	}
}

// TestSubWarpMaskShuffles checks that shuffles and warp votes over disjoint
// 8-lane sub-warp masks (0xff << 8k) execute without deadlock even when groups
// diverge across branches.
func TestSubWarpMaskShuffles(t *testing.T) {
	const src = `
__global__ void subwarp(uint32_t* out) {
    const int lane = threadIdx.x & 31;
    const int grp = lane >> 3;
    const unsigned mask = 0xffu << (8 * grp);
    uint32_t v = lane + 1;
    if (grp & 1) {
        // Only odd groups (lanes 8..15 and 24..31) enter this branch.
        for (int d = 4; d >= 1; d >>= 1) {
            v += __shfl_xor_sync(mask, v, d);
        }
        uint32_t bcast = __shfl_sync(mask, (uint32_t)(lane * 10), 3, 8);
        uint32_t down = __shfl_down_sync(mask, (uint32_t)lane, 2, 8);
        uint32_t up = __shfl_up_sync(mask, (uint32_t)lane, 2, 8);
        unsigned bal = __ballot_sync(mask, (lane & 1) != 0);
        int any = __any_sync(mask, (lane & 7) == 5);
        int all1 = __all_sync(mask, v > 0);
        int all0 = __all_sync(mask, (lane & 1) == 0);
        out[lane * 8 + 0] = v;
        out[lane * 8 + 1] = bcast;
        out[lane * 8 + 2] = down;
        out[lane * 8 + 3] = up;
        out[lane * 8 + 4] = bal;
        out[lane * 8 + 5] = (uint32_t)any;
        out[lane * 8 + 6] = (uint32_t)all1;
        out[lane * 8 + 7] = (uint32_t)all0;
    } else {
        // Even groups (lanes 0..7 and 16..23) do a different reduction count.
        v = __shfl_xor_sync(mask, v, 1);
        __syncwarp(mask);
        unsigned bal = __ballot_sync(mask, (lane & 2) == 0);
        int any = __any_sync(mask, (lane & 7) == 9);
        out[lane * 8 + 0] = v;
        out[lane * 8 + 4] = bal;
        out[lane * 8 + 5] = (uint32_t)any;
    }
}`
	got := u32s(run(t, Program{
		Source:  src,
		Launch:  "eg_emu_launch(dim3(1), dim3(32), 0, subwarp, out)",
		Buffers: []Buffer{{"out", "uint32_t", zeros(32 * 8 * 4)}},
	})["out"])
	for lane := 0; lane < 32; lane++ {
		grp := lane >> 3
		base := 8 * grp
		sub := lane & 7
		if grp&1 == 1 {
			wantSum := uint32(0)
			for i := 0; i < 8; i++ {
				wantSum += uint32(base + i + 1)
			}
			wantBcast := uint32((base + 3) * 10)
			wantDown := uint32(lane)
			if sub+2 < 8 {
				wantDown = uint32(lane + 2)
			}
			wantUp := uint32(lane)
			if sub-2 >= 0 {
				wantUp = uint32(lane - 2)
			}
			wantBal := uint32(0xaa) << (8 * grp)
			row := got[lane*8 : (lane+1)*8]
			wantRow := []uint32{wantSum, wantBcast, wantDown, wantUp, wantBal, 1, 1, 0}
			for j, w := range wantRow {
				if row[j] != w {
					t.Errorf("lane %d field %d = %#x, want %#x (row=%v)", lane, j, row[j], w, row)
				}
			}
		} else {
			wantV := uint32((lane ^ 1) + 1)
			wantBal := uint32(0x33) << (8 * grp)
			if got[lane*8+0] != wantV || got[lane*8+4] != wantBal || got[lane*8+5] != 0 {
				t.Errorf("even group lane %d = (%d, %#x, %d), want (%d, %#x, 0)",
					lane, got[lane*8+0], got[lane*8+4], got[lane*8+5], wantV, wantBal)
			}
		}
	}
}

// TestIntrinsicsAndQualifiers checks __byte_perm, cache-hinted loads/stores,
// __grid_constant__, uint2, and cvt_e4m3x2_to_f16x2.
func TestIntrinsicsAndQualifiers(t *testing.T) {
	const src = `
__global__ void kern(const __grid_constant__ uint2 cfg, const uint32_t* in, uint32_t* out, uint32_t* fp16_out) {
    int i = threadIdx.x;
    uint32_t a = __ldcg(&in[i]);
    uint32_t b = __ldca(&in[i + 32]);
    uint32_t s = __ldcs(&in[i + 64]);
    uint32_t p = __byte_perm(a ^ cfg.x, b ^ cfg.y, s);
    if (i & 1) __stwb(&out[i], p);
    else __stcg(&out[i], p);
    for (int k = i; k < 128; k += 32) {
        uint32_t pair = (uint32_t)(2 * k) | ((uint32_t)(2 * k + 1) << 8);
        fp16_out[k] = cvt_e4m3x2_to_f16x2(pair);
    }
}`
	r := rand.New(rand.NewPCG(11, 22))
	inWords := make([]uint32, 96)
	inBytes := make([]byte, 96*4)
	for i := range inWords {
		inWords[i] = r.Uint32()
		binary.LittleEndian.PutUint32(inBytes[4*i:], inWords[i])
	}
	cfgX, cfgY := uint32(0x13579bdf), uint32(0x2468ace0)
	res := run(t, Program{
		Source: src,
		Launch: fmt.Sprintf("eg_emu_launch(dim3(1), dim3(32), 0, kern, make_uint2(%#x, %#x), in, out, fp16_out)", cfgX, cfgY),
		Buffers: []Buffer{
			{"in", "uint32_t", inBytes},
			{"out", "uint32_t", Poison(32 * 4)},
			{"fp16_out", "uint32_t", Poison(128 * 4)},
		},
	})
	gotPerm := u32s(res["out"])
	for i := 0; i < 32; i++ {
		a, b, s := inWords[i]^cfgX, inWords[i+32]^cfgY, inWords[i+64]
		var bytes8 [8]uint8
		for j := 0; j < 4; j++ {
			bytes8[j] = uint8(a >> (8 * j))
			bytes8[4+j] = uint8(b >> (8 * j))
		}
		var want uint32
		for j := 0; j < 4; j++ {
			want |= uint32(bytes8[(s>>(4*j))&7]) << (8 * j)
		}
		if gotPerm[i] != want {
			t.Errorf("__byte_perm[%d] = %#08x, want %#08x", i, gotPerm[i], want)
		}
	}

	e4m3ToF16 := func(b uint8) uint16 {
		sign := uint16(b&0x80) << 8
		e := int((b >> 3) & 0x0f)
		m := uint16(b & 0x07)
		if e == 15 && m == 7 {
			return sign | 0x7e00
		}
		if e == 0 {
			if m == 0 {
				return sign
			}
			// E4M3 subnormal: (-1)^s * 2^-6 * (m/8) = (-1)^s * 2^-9 * m
			// Normalize into F16 (bias 15):
			f := math.Ldexp(float64(m), -9)
			for exp := -9; exp < 0; exp++ {
				_ = f
			}
			// Since F16 bias is 15, 2^-9 has F16 exponent 15-9 = 6:
			switch m {
			case 1: // 1.0 * 2^-9 -> exp = 6, mant = 0
				return sign | (6 << 10)
			case 2: // 1.0 * 2^-8 -> exp = 7, mant = 0
				return sign | (7 << 10)
			case 3: // 1.5 * 2^-8 -> exp = 7, mant = 0x200
				return sign | (7 << 10) | 0x200
			default: // m in 4..7: 1.(m-4)/4 * 2^-7 -> exp = 8, mant = (m-4)<<8
				return sign | (8 << 10) | ((m - 4) << 8)
			}
		}
		// Normal E4M3: exponent e-7, F16 exponent (e-7)+15 = e+8, mantissa m<<7.
		return sign | (uint16(e+8) << 10) | (m << 7)
	}
	gotFP16 := u32s(res["fp16_out"])
	for k := 0; k < 128; k++ {
		lo := e4m3ToF16(uint8(2 * k))
		hi := e4m3ToF16(uint8(2*k + 1))
		want := uint32(lo) | (uint32(hi) << 16)
		if gotFP16[k] != want {
			t.Errorf("cvt_e4m3x2_to_f16x2(%#02x, %#02x) = %#08x, want %#08x", 2*k, 2*k+1, gotFP16[k], want)
		}
	}
}

func fromE4M3(b uint8) float64 {
	e := int((b >> 3) & 15)
	m := int(b & 7)
	var v float64
	switch {
	case e == 15 && m == 7:
		return math.NaN()
	case e == 0:
		v = math.Ldexp(float64(m), -9)
	default:
		v = math.Ldexp(1.0+float64(m)/8.0, e-7)
	}
	if b&0x80 != 0 {
		return -v
	}
	return v
}

func toF16(f float32) uint16 {
	u := math.Float32bits(f)
	s := (u >> 16) & 0x8000
	a := u & 0x7fffffff
	if a > 0x7f800000 {
		return uint16(s | 0x7e00)
	}
	if a >= 0x477ff000 {
		return uint16(s | 0x7c00)
	}
	if a < 0x38800000 {
		return uint16(s | uint32(math.RoundToEven(float64(math.Float32frombits(a))*16777216.0)))
	}
	r := (((a >> 23) - 112) << 10) | ((a >> 13) & 0x3ff)
	rem := a & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && (r&1) != 0) {
		r++
	}
	return uint16(s | r)
}

func fromF16(h uint16) float64 {
	e := int((h >> 10) & 31)
	m := int(h & 0x3ff)
	var v float64
	switch e {
	case 0:
		v = math.Ldexp(float64(m), -24)
	case 31:
		if m != 0 {
			v = math.NaN()
		} else {
			v = math.Inf(1)
		}
	default:
		v = math.Ldexp(float64(m|0x400), e-25)
	}
	if h&0x8000 != 0 {
		return -v
	}
	return v
}

// TestWGMMASwizzle128BAndE4M3 checks 128B-swizzled E4M3 K-major (m64n64k32)
// and F16 MN-major (m64n64k16) WGMMA in both register-A (RS) and shared-A (SS)
// forms, including async NaN-poison hazards and EG_EMU_SYNC_WGMMA=1 survival.
func TestWGMMASwizzle128BAndE4M3(t *testing.T) {
	const src = `
static inline uint32_t sw128(uint32_t a) { return a ^ (((a >> 7) & 7u) << 4); }

__global__ void wgmma_e4m3_k(const uint8_t* A, const uint8_t* B, float* D, int mode) {
    unsigned char* smem = eg_emu_dynamic_smem();
    // Place A (64x32 K-major, 128B swizzled, sbo=1024) at smem+0,
    // and B (64x32 K-major, 128B swizzled, sbo=1024) at smem+8192.
    for (int idx = threadIdx.x; idx < 64 * 32; idx += blockDim.x) {
        int r = idx / 32, k = idx % 32;
        smem[sw128((r / 8) * 1024 + (r % 8) * 128 + k)] = A[r * 32 + k];
        smem[sw128(8192 + (r / 8) * 1024 + (r % 8) * 128 + k)] = B[r * 32 + k];
    }
    __syncthreads();

    const int wig = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    uint32_t a_frag[4];
    for (int i = 0; i < 4; ++i) {
        int row = wig * 16 + g + 8 * (i & 1);
        int k0 = 4 * c + 16 * (i >> 1);
        uint32_t w = 0;
        for (int b = 0; b < 4; ++b) w |= (uint32_t)A[row * 32 + k0 + b] << (8 * b);
        a_frag[i] = w;
    }

    uint64_t da = eg_emu_wgmma_desc_128b(smem, 1024);
    uint64_t db = eg_emu_wgmma_desc_128b(smem + 8192, 1024);
    float d[32];
    for (int i = 0; i < 32; ++i) d[i] = 1234.0f;  // scale_d=0 must overwrite this

    eg_emu_wgmma_fence();
    if (mode == 0) {
        // Register-A + 128B-swizzled shared-B, scale_d=0
        eg_emu_wgmma_e4m3(d, a_frag, db, 64, 0);
    } else {
        // 128B-swizzled shared-A + 128B-swizzled shared-B, scale_d=0
        eg_emu_wgmma_e4m3(d, da, db, 64, 0);
    }
    eg_emu_wgmma_commit();

    auto store = [&](float* out) {
        for (int nt = 0; nt < 8; ++nt)
            for (int i = 0; i < 4; ++i) {
                int row = wig * 16 + g + 8 * (i / 2), col = nt * 8 + 2 * c + (i & 1);
                out[row * 64 + col] = d[nt * 4 + i];
            }
    };
    if (mode == 2) store(D + 4096);  // hazard: read before wait_group
    if (mode != 3) eg_emu_wgmma_wait(0);
    store(D);
}

__global__ void wgmma_f16_mn_k(const uint16_t* A, const uint16_t* B, float* D, int mode) {
    unsigned char* smem = eg_emu_dynamic_smem();
    // Place B (16x64 MN-major, 128B swizzled, sbo=1024) at smem+0:
    // element (k, col) at sw128((k/8)*1024 + (k%8)*128 + 2*col).
    for (int idx = threadIdx.x; idx < 16 * 64; idx += blockDim.x) {
        int k = idx / 64, col = idx % 64;
        std::memcpy(smem + sw128((k / 8) * 1024 + (k % 8) * 128 + 2 * col), &B[k * 64 + col], 2);
    }
    __syncthreads();

    const int wig = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    uint32_t a_frag[4];
    for (int i = 0; i < 4; ++i) {
        int row = wig * 16 + g + 8 * (i & 1);
        int k0 = 2 * c + 8 * (i >> 1);
        a_frag[i] = (uint32_t)A[row * 16 + k0] | ((uint32_t)A[row * 16 + k0 + 1] << 16);
    }

    uint64_t db = eg_emu_wgmma_desc_128b(smem, 1024);
    float d[32] = {};
    eg_emu_wgmma_fence();
    eg_emu_wgmma_f16(d, a_frag, db, 64, /*trans_b=*/1, /*scale_d=*/1);
    eg_emu_wgmma_commit();

    auto store = [&](float* out) {
        for (int nt = 0; nt < 8; ++nt)
            for (int i = 0; i < 4; ++i) {
                int row = wig * 16 + g + 8 * (i / 2), col = nt * 8 + 2 * c + (i & 1);
                out[row * 64 + col] = d[nt * 4 + i];
            }
    };
    if (mode == 1) store(D + 4096);  // read before wait
    eg_emu_wgmma_wait(0);
    store(D);
}`

	r := rand.New(rand.NewPCG(7, 8))
	aFP8 := make([]byte, 64*32)
	bFP8 := make([]byte, 64*32)
	for i := range aFP8 {
		// Random non-NaN E4M3 bytes (avoid 0x7f and 0xff).
		v := uint8(r.UintN(254))
		if v&0x7f == 0x7f {
			v &= 0x7e
		}
		// Keep exponents moderate to avoid float32 accumulation rounding divergence.
		v = (v & 0x87) | (uint8(r.UintN(8)+4) << 3)
		aFP8[i] = v
	}
	for i := range bFP8 {
		v := uint8(r.UintN(254))
		v = (v & 0x87) | (uint8(r.UintN(8)+4) << 3)
		bFP8[i] = v
	}
	wantE4M3 := make([]float64, 64*64)
	for m := 0; m < 64; m++ {
		for n := 0; n < 64; n++ {
			var sum float64
			for k := 0; k < 32; k++ {
				sum += fromE4M3(aFP8[m*32+k]) * fromE4M3(bFP8[n*32+k])
			}
			wantE4M3[m*64+n] = sum
		}
	}

	for _, tt := range []struct {
		name  string
		mode  int
		flags []string
		early string
	}{
		{"e4m3 RS 128B swizzle", 0, nil, ""},
		{"e4m3 SS 128B swizzle", 1, nil, ""},
		{"e4m3 hazard read before wait", 2, nil, "nan"},
		{"e4m3 hazard survives under EG_EMU_SYNC_WGMMA", 2, []string{"-DEG_EMU_SYNC_WGMMA=1"}, "done"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			out := run(t, Program{
				Source:  src,
				Flags:   tt.flags,
				Launch:  fmt.Sprintf("eg_emu_launch(dim3(1), dim3(128), 16384, wgmma_e4m3_k, A, B, D, %d)", tt.mode),
				Buffers: []Buffer{{"A", "uint8_t", aFP8}, {"B", "uint8_t", bFP8}, {"D", "float", Poison(2 * 4096 * 4)}},
			})
			d := f32s(out["D"])
			for i, w := range wantE4M3 {
				if !near(d[i], w) {
					t.Fatalf("D[%d] = %v, want %v", i, d[i], w)
				}
				switch e := d[4096+i]; tt.early {
				case "nan":
					if !math.IsNaN(float64(e)) {
						t.Fatalf("early D[%d] = %v, want NaN", i, e)
					}
				case "done":
					if !near(e, w) {
						t.Fatalf("early D[%d] = %v, want %v", i, e, w)
					}
				}
			}
		})
	}

	// Missing wait_group on E4M3 WGMMA must fail at thread exit.
	_, err := Run(context.Background(), t.TempDir(), Program{
		Source:  src,
		Launch:  "eg_emu_launch(dim3(1), dim3(128), 16384, wgmma_e4m3_k, A, B, D, 3)",
		Buffers: []Buffer{{"A", "uint8_t", aFP8}, {"B", "uint8_t", bFP8}, {"D", "float", Poison(2 * 4096 * 4)}},
	})
	if err == nil || !strings.Contains(err.Error(), "wgmma ops pending") {
		t.Errorf("missing wait err = %v, want pending wgmma error", err)
	}

	// Now test F16 MN-major 128B swizzle (both async hazard and sync survival).
	aF16 := make([]uint16, 64*16)
	bF16 := make([]uint16, 16*64)
	encU16 := func(h []uint16) []byte {
		b := make([]byte, 2*len(h))
		for i, v := range h {
			binary.LittleEndian.PutUint16(b[2*i:], v)
		}
		return b
	}
	for i := range aF16 {
		aF16[i] = toF16(float32(r.NormFloat64()))
	}
	for i := range bF16 {
		bF16[i] = toF16(float32(r.NormFloat64()))
	}
	wantF16 := make([]float64, 64*64)
	for m := 0; m < 64; m++ {
		for n := 0; n < 64; n++ {
			var sum float64
			for k := 0; k < 16; k++ {
				sum += fromF16(aF16[m*16+k]) * fromF16(bF16[k*64+n])
			}
			wantF16[m*64+n] = sum
		}
	}
	for _, tt := range []struct {
		name  string
		flags []string
		early string
	}{
		{"f16 MN-major 128B swizzle hazard", nil, "nan"},
		{"f16 MN-major 128B swizzle sync", []string{"-DEG_EMU_SYNC_WGMMA=1"}, "done"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			out := run(t, Program{
				Source:  src,
				Flags:   tt.flags,
				Launch:  "eg_emu_launch(dim3(1), dim3(128), 16384, wgmma_f16_mn_k, A, B, D, 1)",
				Buffers: []Buffer{{"A", "uint16_t", encU16(aF16)}, {"B", "uint16_t", encU16(bF16)}, {"D", "float", Poison(2 * 4096 * 4)}},
			})
			d := f32s(out["D"])
			for i, w := range wantF16 {
				if !near(d[i], w) {
					t.Fatalf("D[%d] = %v, want %v", i, d[i], w)
				}
				if tt.early == "nan" && !math.IsNaN(float64(d[4096+i])) {
					t.Fatalf("early D[%d] = %v, want NaN", i, d[4096+i])
				}
				if tt.early == "done" && !near(d[4096+i], w) {
					t.Fatalf("early D[%d] = %v, want %v", i, d[4096+i], w)
				}
			}
		})
	}
}

func TestEpochReadyCounters(t *testing.T) {
	const src = `
__global__ void stage_pair(const float* in, float* mid, float* out, unsigned long long* ready, unsigned int epoch) {
    const int b = blockIdx.x, tid = threadIdx.x;
    if (b < 4) {
        mid[b * 32 + tid] = in[b * 32 + tid] + (float)epoch;
        __syncthreads();
        if (tid == 0) {
            eg_emu_ready_stage(&ready[0], epoch, 4, mid, 4 * 32 * sizeof(float));
            eg_emu_ready_signal(&ready[0], epoch, 1);
        }
        return;
    }
    if (tid == 0) {
#if !defined(EG_NO_READY_WAIT)
        eg_emu_ready_wait(&ready[0], epoch, 4);
#endif
    }
    __syncthreads();
    out[(b - 4) * 32 + tid] = mid[(b - 4) * 32 + tid] * 2.0f;
}
`
	in := make([]byte, 128*4)
	for i := range 128 {
		binary.LittleEndian.PutUint32(in[4*i:], math.Float32bits(float32(i+1)))
	}
	res := run(t, Program{
		Source: src,
		Launch: `eg_emu_launch(dim3(8), dim3(32), 0, stage_pair, in, mid, out, ready, 1u);
    eg_emu_launch(dim3(8), dim3(32), 0, stage_pair, in, mid, out, ready, 2u);
    eg_emu_launch(dim3(8), dim3(32), 0, stage_pair, in, mid, out, ready, 3u)`,
		Buffers: []Buffer{
			{"in", "float", in},
			{"mid", "float", Poison(128 * 4)},
			{"out", "float", Poison(128 * 4)},
			{"ready", "unsigned long long", zeros(8)},
		},
	})
	got := f32s(res["out"])
	for i, v := range got {
		if want := float32(i+1+3) * 2; v != want {
			t.Fatalf("out[%d] = %g, want %g", i, v, want)
		}
	}
	if r := binary.LittleEndian.Uint64(res["ready"]); r != (3<<32)|4 {
		t.Fatalf("ready[0] = %#x, want %#x", r, uint64((3<<32)|4))
	}

	// Without eg_emu_ready_wait, mid stays poisoned with NaN.
	resNoWait := run(t, Program{
		Source: src,
		Flags:  []string{"-DEG_NO_READY_WAIT"},
		Launch: `eg_emu_launch(dim3(8), dim3(32), 0, stage_pair, in, mid, out, ready, 1u)`,
		Buffers: []Buffer{
			{"in", "float", in},
			{"mid", "float", Poison(128 * 4)},
			{"out", "float", Poison(128 * 4)},
			{"ready", "unsigned long long", zeros(8)},
		},
	})
	for i, v := range f32s(resNoWait["out"]) {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("out[%d] without wait = %g, want NaN", i, v)
		}
	}
}
