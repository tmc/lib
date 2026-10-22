package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/lm/quant"
)

func TestCUDAExpertAddressRoutes(t *testing.T) {
	ptrs := [][3]uint64{{9000, 3000, 8000}, {1000, 7000, 5000}, {9000, 3000, 8000}}
	resolve := func(slot int) ([3]uint64, [3]uint64, error) { return ptrs[slot], [3]uint64{198, 198, 252}, nil }
	a, err := cudaCollectExpertAddresses(3, 256, 256, resolve)
	// width and hidden must describe exact component row extents, not just
	// a conveniently larger allocation.
	if err == nil {
		t.Fatal("wrong row extent accepted")
	}
	resolve = func(slot int) ([3]uint64, [3]uint64, error) { return ptrs[slot], [3]uint64{16896, 16896, 21504}, nil }
	a, err = cudaCollectExpertAddresses(3, 256, 256, resolve)
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range ptrs {
		if a.gate[i] != p[0] || a.up[i] != p[1] || a.down[i] != p[2] {
			t.Fatal("route order or duplicate changed")
		}
	}
	for _, tc := range []struct {
		name string
		ptr  [3]uint64
		size [3]uint64
		err  error
	}{
		{"stale", ptrs[0], [3]uint64{16896, 16896, 21504}, errors.New("stale view")},
		{"null", [3]uint64{0, 3000, 8000}, [3]uint64{16896, 16896, 21504}, nil},
		{"overflow", [3]uint64{math.MaxUint64, 3000, 8000}, [3]uint64{16896, 16896, 21504}, nil},
		{"short", ptrs[0], [3]uint64{16895, 16896, 21504}, nil},
		{"whole_slab", ptrs[0], [3]uint64{50688, 50688, 64512}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := cudaCollectExpertAddresses(3, 256, 256, func(slot int) ([3]uint64, [3]uint64, error) {
				if slot == 1 {
					return tc.ptr, tc.size, tc.err
				}
				return ptrs[slot], [3]uint64{16896, 16896, 21504}, nil
			})
			if err == nil || len(got.gate) != 0 {
				t.Fatal("invalid view leaked partial table")
			}
		})
	}
}

func TestCUDAExpertAddressCleanup(t *testing.T) {
	finished := false
	finishes := 0
	freed := [6]bool{}
	calls := [6]int{}
	finishFailure := true
	freeFailure := true
	finish := func() error {
		finishes++
		if finishFailure {
			return errors.New("consumer incomplete")
		}
		return nil
	}
	var free [6]func() error
	for i := range free {
		free[i] = func() error {
			calls[i]++
			if i == 2 && freeFailure {
				return errors.New("free uncertain")
			}
			freed[i] = true
			return nil
		}
	}
	if err := cudaCloseExpertAddresses(&finished, finish, free); err == nil || finished || calls != [6]int{} {
		t.Fatal("consumer failure released tables")
	}
	finishFailure = false
	if err := cudaCloseExpertAddresses(&finished, finish, free); err == nil || !finished || !freed[0] || !freed[1] || freed[2] || freed[3] {
		t.Fatal("free failure did not retain remaining tables")
	}
	freeFailure = false
	if err := cudaCloseExpertAddresses(&finished, finish, free); err != nil {
		t.Fatal(err)
	}
	if finishes != 2 {
		t.Fatal("cleanup retried released selection")
	}
	for _, done := range freed {
		if !done {
			t.Fatal("remaining table orphaned")
		}
	}
}

func TestCUDAExpertAddressExtent(t *testing.T) {
	for _, tc := range []struct {
		cols, rows, slots int
		block             uint64
		want              uint64
	}{
		{256, 3, 4, 66, 198}, {768, 7, 6, 84, 1764},
		{0, 3, 4, 66, 0}, {257, 3, 4, 66, 0}, {256, 0, 4, 66, 0},
		{256, 3, 0, 66, 0}, {256, math.MaxInt32, 2, 66, 0},
		{256, 2, 1, math.MaxUint64, 0},
	} {
		got, err := cudaExpertAddressExtent(tc.cols, tc.rows, tc.slots, tc.block)
		if (err == nil) != (tc.want != 0) || got != tc.want {
			t.Fatalf("%+v: got %d, %v", tc, got, err)
		}
	}
}

// This executes the actual source body with CPU threads and barriers. It tests
// arithmetic and route indexing, not CUDA compilation, GPU memory or streams.
func TestCUDAExpertAddressSourceOracle(t *testing.T) {
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("clang++ unavailable")
	}
	base, err := os.ReadFile("cuda_experts.cu")
	if err != nil {
		t.Fatal(err)
	}
	base = bytes.Split(base, []byte("extern \"C\" __global__ void ds4_iq2_xxs_pair_selected_f32"))[0]
	const cols, rows, experts = 512, 3, 4
	ids := []int{3, 1, 3, 0}
	g := addressOracleFixture(66, cols, rows, experts, 1)
	u := addressOracleFixture(66, cols, rows, experts, 5)
	d := addressOracleFixture(84, cols, rows, experts, 9)
	x, sx := make([]float32, cols), make([]float32, cols*len(ids))
	for i := range x {
		x[i] = float32(i%29-14) / 16
	}
	for i := range sx {
		sx[i] = float32((i*7+i/cols)%31-15) / 16
	}
	var arrays strings.Builder
	for _, a := range []struct {
		name string
		raw  []byte
	}{{"g", g}, {"u", u}, {"d", d}} {
		fmt.Fprintf(&arrays, "unsigned char %s[] = {", a.name)
		for _, b := range a.raw {
			fmt.Fprintf(&arrays, "%d,", b)
		}
		arrays.WriteString("};\n")
	}
	for _, a := range []struct {
		name string
		raw  []float32
	}{{"x", x}, {"sx", sx}} {
		fmt.Fprintf(&arrays, "float %s[] = {", a.name)
		for _, f := range a.raw {
			fmt.Fprintf(&arrays, "%gf,", float64(f))
		}
		arrays.WriteString("};\n")
	}
	// Decimal integer float literals require a decimal point in C++.
	arrayText := strings.ReplaceAll(arrays.String(), "0f,", "0.0f,")
	for _, n := range []string{"1", "-1"} {
		arrayText = strings.ReplaceAll(arrayText, n+"f,", n+".0f,")
	}
	preamble := `#include <barrier>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#define __device__
#define __global__
#define __shared__ static
struct Dim { unsigned x; };
thread_local Dim threadIdx;
Dim blockIdx, blockDim{256};
std::barrier sync_threads(256);
void __syncthreads() { sync_threads.arrive_and_wait(); }
float __int_as_float(unsigned x) { float f; std::memcpy(&f,&x,4); return f; }
float ds4_half_to_float(unsigned short h) {
  float f = std::ldexp(float(h & 1023), -24);
  unsigned e = (h >> 10) & 31;
  if (e) f = std::ldexp(1.0f + float(h & 1023)/1024, int(e)-15);
  return (h & 32768) ? -f : f;
}
template<class F> void run(F f) {
  std::vector<std::thread> ts;
  for(unsigned t=0;t<256;t++) ts.emplace_back([=]{threadIdx.x=t; f();});
  for(auto &t:ts) t.join();
}
`
	body := `int main() {
  unsigned ids[4]={3,1,3,0};
  unsigned long long gp[4],up[4],dp[4],gb[4],ub[4],db[4];
  for(unsigned s=0;s<4;s++) {
    gp[s]=(unsigned long long)(g+ids[s]*396); up[s]=(unsigned long long)(u+ids[s]*396);
    dp[s]=(unsigned long long)(d+ids[s]*504); gb[s]=ub[s]=396; db[s]=504;
  }
  float go[12],uo[12],out[12];
  for(unsigned b=0;b<12;b++) { blockIdx.x=b;
    run([&]{ds4_iq2_xxs_pair_address_f32(gp,up,gb,ub,x,go,uo,512,3,4,512,12);});
    run([&]{ds4_q2_k_address_f32(dp,db,sx,out,512,3,4,2048,12);});
  }
  for(auto p:{go,uo,out}) for(unsigned i=0;i<12;i++) std::printf("%.9g\n",p[i]);
  gb[0]--; blockIdx.x=0;
  run([&]{ds4_iq2_xxs_pair_address_f32(gp,up,gb,ub,x,go,uo,512,3,4,512,12);});
  if(!std::isnan(go[0]) || !std::isnan(uo[0])) return 3;
  db[0]--; run([&]{ds4_q2_k_address_f32(dp,db,sx,out,512,3,4,2048,12);});
  if(!std::isnan(out[0])) return 4;
}
`
	dir := t.TempDir()
	src := filepath.Join(dir, "source.cc")
	bin := filepath.Join(dir, "oracle")
	if err := os.WriteFile(src, []byte(preamble+string(base)+cudaExpertAddressSource+arrayText+body), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(compiler, "-std=c++20", "-O1", "-pthread", src, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
	output, err := exec.Command(bin).CombinedOutput()
	if err != nil {
		t.Fatalf("source oracle: %v\n%s", err, output)
	}
	fields := strings.Fields(string(output))
	if len(fields) != 36 {
		t.Fatalf("got %d outputs", len(fields))
	}
	index := 0
	for _, a := range []struct {
		raw     []byte
		input   []float32
		block   int
		slotted bool
	}{{g, x, 66, false}, {u, x, 66, false}, {d, sx, 84, true}} {
		for slot, id := range ids {
			for row := 0; row < rows; row++ {
				off := (id*rows + row) * (cols / 256) * a.block
				dequant := quant.DequantizeIQ2_XXS
				if a.block == 84 {
					dequant = quant.DequantizeQ2_K
				}
				w, err := dequant(a.raw[off:off+cols/256*a.block], cols)
				if err != nil {
					t.Fatal(err)
				}
				input := a.input
				if a.slotted {
					input = input[slot*cols : (slot+1)*cols]
				}
				var want float64
				for i, v := range w {
					want += float64(v) * float64(input[i])
				}
				got, err := strconv.ParseFloat(fields[index], 64)
				if err != nil {
					t.Fatal(err)
				}
				index++
				if math.IsNaN(got) || math.Abs(got-want) > 1e-4+1e-4*math.Abs(want) {
					t.Fatalf("output %d: %g want %g", index-1, got, want)
				}
			}
		}
	}
}

func addressOracleFixture(block, cols, rows, experts, seed int) []byte {
	b := make([]byte, block*(cols/256)*rows*experts)
	for off := 0; off < len(b); off += block {
		for i := 0; i < block; i++ {
			b[off+i] = byte((off/block*17 + i*13 + seed*11) % 256)
		}
		if block == 66 {
			binary.LittleEndian.PutUint16(b[off:], 0x2400)
		} else {
			binary.LittleEndian.PutUint16(b[off+80:], 0x2400)
			binary.LittleEndian.PutUint16(b[off+82:], 0x2000)
		}
	}
	return b
}
