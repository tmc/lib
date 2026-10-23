package hopper

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// An emuBuf is one kernel buffer for runEmu. Every buffer is read back after
// the launch; out-only buffers start zeroed with len(data) bytes.
type emuBuf struct {
	name  string
	ctype string
	data  []byte
}

// runEmu compiles prelude+src as host C++ against testdata/emu.h, runs
// launch (a C++ statement that may name the buffers), and returns every
// buffer's final contents.
func runEmu(t *testing.T, src, launch string, bufs ...emuBuf) map[string][]byte {
	t.Helper()
	cxx, err := exec.LookPath("clang++")
	if err != nil {
		cxx, err = exec.LookPath("g++")
	}
	if err != nil {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}
	emu, err := filepath.Abs("testdata/emu.h")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	var main strings.Builder
	main.WriteString(Prelude)
	main.WriteString(src)
	main.WriteString("\nint main() {\n")
	for _, b := range bufs {
		if err := os.WriteFile(filepath.Join(dir, b.name+".bin"), b.data, 0o644); err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(&main, "    auto eg_b_%s = eg_emu_read(%q);\n    auto* %s = reinterpret_cast<%s*>(eg_b_%s.data());\n",
			b.name, filepath.Join(dir, b.name+".bin"), b.name, b.ctype, b.name)
	}
	fmt.Fprintf(&main, "    %s;\n", launch)
	for _, b := range bufs {
		fmt.Fprintf(&main, "    eg_emu_write(%q, %s, %d);\n", filepath.Join(dir, b.name+".out"), b.name, len(b.data))
	}
	main.WriteString("    return 0;\n}\n")
	cc := filepath.Join(dir, "main.cc")
	if err := os.WriteFile(cc, []byte(main.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "main")
	cmd := exec.Command(cxx, "-std=c++20", "-O1", "-w", "-include", emu, "-o", bin, cc)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile emulated kernel: %v\n%s\n--- source ---\n%s", err, out, numbered(main.String()))
	}
	if out, err := exec.Command(bin).CombinedOutput(); err != nil {
		t.Fatalf("run emulated kernel: %v\n%s", err, out)
	}
	res := make(map[string][]byte)
	for _, b := range bufs {
		data, err := os.ReadFile(filepath.Join(dir, b.name+".out"))
		if err != nil {
			t.Fatal(err)
		}
		res[b.name] = data
	}
	return res
}

func numbered(s string) string {
	var b strings.Builder
	for i, line := range strings.Split(s, "\n") {
		fmt.Fprintf(&b, "%4d %s\n", i+1, line)
	}
	return b.String()
}

func f32Bytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

func bytesF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func u16Bytes(v []uint16) []byte {
	b := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[2*i:], x)
	}
	return b
}

func bytesU16(b []byte) []uint16 {
	v := make([]uint16, len(b)/2)
	for i := range v {
		v[i] = binary.LittleEndian.Uint16(b[2*i:])
	}
	return v
}

func i32Bytes(v []int32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return b
}

func i64Bytes(v []int64) []byte {
	b := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(b[8*i:], uint64(x))
	}
	return b
}

// TestPreludeConversions checks the device conversions against the Go codecs
// for every E4M3 and E2M1 code and a spread of BF16 inputs.
func TestPreludeConversions(t *testing.T) {
	const kernel = `
__global__ void conv(const float* in, uint16_t* bf, uint8_t* e4, float* e4dec, uint16_t* e2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        bf[i] = eg_f32_to_bf16(in[i]);
        uint16_t pair = eg_f32x2_to_e4m3x2(in[i], -in[i]);
        e4[2 * i] = pair & 0xff;
        e4[2 * i + 1] = pair >> 8;
    }
    if (i < 256) e4dec[i] = eg_e4m3_to_f32(i);
    if (i < 64) {
        // codes 4i .. 4i+3 (mod 16) in one 16-bit word
        uint32_t q = 0;
        for (int k = 0; k < 4; ++k) q |= ((4 * i + k) & 15u) << (4 * k);
        uint32_t out[2];
        eg_e2m1x4_to_bf16x4(q, out);
        e2[4 * i] = out[0] & 0xffff; e2[4 * i + 1] = out[0] >> 16;
        e2[4 * i + 2] = out[1] & 0xffff; e2[4 * i + 3] = out[1] >> 16;
    }
}
`
	var in []float32
	for b := 0; b < 0x7f; b++ {
		v := FromE4M3Bits(uint8(b))
		in = append(in, v, math.Nextafter32(v, 1000), v*1.0625)
	}
	in = append(in, 1e6, 1+1.0/256, 1+3.0/256, -0.3333, 3e-39)
	n := len(in)
	res := runEmu(t, kernel, fmt.Sprintf("eg_emu_launch(dim3((%d+127)/128), dim3(128), 0, conv, in, bf, e4, e4dec, e2, %d)", n, n),
		emuBuf{"in", "float", f32Bytes(in)},
		emuBuf{"bf", "uint16_t", make([]byte, 2*n)},
		emuBuf{"e4", "uint8_t", make([]byte, 2*n)},
		emuBuf{"e4dec", "float", make([]byte, 4*256)},
		emuBuf{"e2", "uint16_t", make([]byte, 2*256)},
	)
	bf := bytesU16(res["bf"])
	for i, f := range in {
		if want := BF16Bits(f); bf[i] != want {
			t.Errorf("bf16(%g) = %#x, want %#x", f, bf[i], want)
		}
		if got, want := res["e4"][2*i], E4M3Bits(f); got != want {
			t.Errorf("e4m3(%g) = %#x, want %#x", f, got, want)
		}
		if got, want := res["e4"][2*i+1], E4M3Bits(-f); got != want {
			t.Errorf("e4m3(%g) = %#x, want %#x", -f, got, want)
		}
	}
	for i, got := range bytesF32(res["e4dec"]) {
		want := FromE4M3Bits(uint8(i))
		if got != want && !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
			t.Errorf("decode e4m3 %#x = %g, want %g", i, got, want)
		}
	}
	for i, got := range bytesU16(res["e2"]) {
		if want := FromE2M1(uint8(i % 16)); FromBF16Bits(got) != want {
			t.Errorf("decode e2m1 %d = %g, want %g", i%16, FromBF16Bits(got), want)
		}
	}
}

// TestPreludeMMA multiplies one 16x16 by 16x8 BF16 tile through ldmatrix and
// mma.sync and compares with a direct product.
func TestPreludeMMA(t *testing.T) {
	const kernel = `
__global__ void mma(const uint16_t* a, const uint16_t* bt, float* d) {
    alignas(16) __shared__ uint16_t sa[16 * 16];
    alignas(16) __shared__ uint16_t sb[8 * 16];
    int lane = threadIdx.x;
    for (int i = lane; i < 256; i += 32) sa[i] = a[i];
    for (int i = lane; i < 128; i += 32) sb[i] = bt[i];
    __syncthreads();
    uint32_t fa[4], fb[4];
    // A row-major 16x16: matrices (rows 0-7, k 0-7), (rows 8-15, k 0-7), (rows 0-7, k 8-15), (rows 8-15, k 8-15).
    eg_ldmatrix_x4(fa, &sa[(lane % 16) * 16 + (lane / 16) * 8]);
    // B stored as [n][k] (k contiguous): matrices (n 0-7, k 0-7), (n 0-7, k 8-15).
    eg_ldmatrix_x4(fb, &sb[(lane % 8) * 16 + ((lane / 8) % 2) * 8]);
    float c[4] = {0, 0, 0, 0}, out[4];
    eg_mma_bf16(out, fa, fb, c);
    int g = lane / 4, t = lane % 4;
    d[g * 8 + 2 * t] = out[0];
    d[g * 8 + 2 * t + 1] = out[1];
    d[(g + 8) * 8 + 2 * t] = out[2];
    d[(g + 8) * 8 + 2 * t + 1] = out[3];
}
`
	a := make([]uint16, 256)
	bt := make([]uint16, 128)
	for i := range a {
		a[i] = BF16Bits(float32(i%7) - 3)
	}
	for i := range bt {
		bt[i] = BF16Bits(float32(i%5)*0.5 - 1)
	}
	res := runEmu(t, kernel, "eg_emu_launch(dim3(1), dim3(32), 0, mma, a, bt, d)",
		emuBuf{"a", "uint16_t", u16Bytes(a)},
		emuBuf{"bt", "uint16_t", u16Bytes(bt)},
		emuBuf{"d", "float", make([]byte, 4*128)},
	)
	d := bytesF32(res["d"])
	for m := 0; m < 16; m++ {
		for n := 0; n < 8; n++ {
			var want float32
			for k := 0; k < 16; k++ {
				want += FromBF16Bits(a[m*16+k]) * FromBF16Bits(bt[n*16+k])
			}
			if d[m*8+n] != want {
				t.Fatalf("d[%d][%d] = %g, want %g", m, n, d[m*8+n], want)
			}
		}
	}
}

// TestPreludeMMAE4M3 multiplies one 16x32 by 32x8 E4M3 tile through
// ldmatrix (on the b16 view of the bytes) and eg_mma_e4m3, as
// PagedAttention's QKFP8 path addresses them.
func TestPreludeMMAE4M3(t *testing.T) {
	const kernel = `
__global__ void mma8(const uint8_t* a, const uint8_t* bt, float* d) {
    alignas(16) __shared__ uint16_t sa[16 * 16];
    alignas(16) __shared__ uint16_t sb[8 * 16];
    int lane = threadIdx.x;
    for (int i = lane; i < 256; i += 32) sa[i] = a[2 * i] | (uint16_t)(a[2 * i + 1] << 8);
    for (int i = lane; i < 128; i += 32) sb[i] = bt[2 * i] | (uint16_t)(bt[2 * i + 1] << 8);
    __syncthreads();
    uint32_t fa[4], fb[4];
    eg_ldmatrix_x4(fa, &sa[(lane % 16) * 16 + (lane / 16) * 8]);
    // B stored as [n][k] bytes: matrices (n 0-7, k 0-15), (n 0-7, k 16-31).
    eg_ldmatrix_x4(fb, &sb[(lane % 8) * 16 + ((lane / 8) % 2) * 8]);
    float c[4] = {0, 0, 0, 0}, out[4];
    eg_mma_e4m3(out, fa, fb, c);
    int g = lane / 4, t = lane % 4;
    d[g * 8 + 2 * t] = out[0];
    d[g * 8 + 2 * t + 1] = out[1];
    d[(g + 8) * 8 + 2 * t] = out[2];
    d[(g + 8) * 8 + 2 * t + 1] = out[3];
}
`
	a := make([]uint8, 16*32)
	bt := make([]uint8, 8*32)
	for i := range a {
		a[i] = E4M3Bits(float32(i%7) - 3.5)
	}
	for i := range bt {
		bt[i] = E4M3Bits(float32(i%11)*0.25 - 1)
	}
	res := runEmu(t, kernel, "eg_emu_launch(dim3(1), dim3(32), 0, mma8, a, bt, d)",
		emuBuf{"a", "uint8_t", a},
		emuBuf{"bt", "uint8_t", bt},
		emuBuf{"d", "float", make([]byte, 4*128)},
	)
	d := bytesF32(res["d"])
	for m := 0; m < 16; m++ {
		for n := 0; n < 8; n++ {
			var want float64
			for k := 0; k < 32; k++ {
				want += float64(FromE4M3Bits(a[m*32+k])) * float64(FromE4M3Bits(bt[n*32+k]))
			}
			if float64(d[m*8+n]) != want {
				t.Fatalf("d[%d][%d] = %g, want %g", m, n, d[m*8+n], want)
			}
		}
	}
}
