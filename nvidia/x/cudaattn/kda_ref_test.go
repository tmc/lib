package cudaattn

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

// These tolerances precede candidate evaluation. Fixtures use unit q/key,
// |value|<=0.5 and |initial|<=0.02, at most 33 updates per sequence.
const kdaAbsolute = 3e-5
const kdaRelative = 3e-5

type kdaFixture struct {
	c                                 KDAConfig
	offsets                           []int32
	q, key, value, decay, beta, state []float32
}

var kdaHostBuild struct {
	sync.Mutex
	binaries map[string]string
	dirs     []string
}

func TestMain(m *testing.M) {
	code := m.Run()
	for _, dir := range kdaHostBuild.dirs {
		os.RemoveAll(dir)
	}
	os.Exit(code)
}

func kdaInput(length, sequences, heads int) kdaFixture {
	f := kdaFixture{c: KDAConfig{128, heads, sequences, length * sequences, 0.7}}
	e, _ := f.c.Extents()
	f.offsets = make([]int32, sequences+1)
	for i := range f.offsets {
		f.offsets[i] = int32(i * length)
	}
	f.q = make([]float32, e.Vectors)
	f.key = make([]float32, e.Vectors)
	f.value = make([]float32, e.Vectors)
	f.decay = make([]float32, e.Vectors)
	f.beta = make([]float32, e.Betas)
	f.state = make([]float32, e.State)
	r := rand.New(rand.NewSource(76))
	for _, a := range [][]float32{f.q, f.key} {
		for i := 0; i < len(a); i += 128 {
			norm := 0.0
			for j := 0; j < 128; j++ {
				a[i+j] = float32(r.Float64()*2 - 1)
				norm += float64(a[i+j]) * float64(a[i+j])
			}
			for j := 0; j < 128; j++ {
				a[i+j] = float32(float64(a[i+j]) / math.Sqrt(norm))
			}
		}
	}
	for i := range f.value {
		f.value[i] = float32(r.Float64() - 0.5)
		f.decay[i] = float32(0.8 + 0.2*r.Float64())
	}
	for i := range f.beta {
		f.beta[i] = float32(0.15 + 0.8*r.Float64())
	}
	for i := range f.state {
		f.state[i] = float32((r.Float64()*2 - 1) * 0.02)
	}
	return f
}

// kdaOracle independently computes full FP64 matrices, with decay, projection,
// outer-product correction and readout in separate passes. It never calls the
// CUDA source or a production recurrence helper and never mutates its input.
func kdaOracle(f kdaFixture) ([]float64, []float64) {
	state := make([]float64, len(f.state))
	for i, x := range f.state {
		state[i] = float64(x)
	}
	out := make([]float64, len(f.q))
	for s := 0; s < f.c.Sequences; s++ {
		for h := 0; h < f.c.Heads; h++ {
			base := (s*f.c.Heads + h) * 128 * 128
			for t := int(f.offsets[s]); t < int(f.offsets[s+1]); t++ {
				token := (t*f.c.Heads + h) * 128
				sd := make([]float64, 128*128)
				u := make([]float64, 128)
				for v := 0; v < 128; v++ {
					for k := 0; k < 128; k++ {
						sd[v*128+k] = state[base+v*128+k] * float64(f.decay[token+k])
					}
				}
				for v := 0; v < 128; v++ {
					u[v] = float64(f.value[token+v])
					projection := 0.0
					for k := 0; k < 128; k++ {
						projection += sd[v*128+k] * float64(f.key[token+k])
					}
					u[v] -= projection
				}
				for v := 0; v < 128; v++ {
					for k := 0; k < 128; k++ {
						state[base+v*128+k] = sd[v*128+k] + float64(f.beta[t*f.c.Heads+h])*u[v]*float64(f.key[token+k])
					}
				}
				for v := 0; v < 128; v++ {
					dot := 0.0
					for k := 0; k < 128; k++ {
						dot += state[base+v*128+k] * float64(f.q[token+k])
					}
					out[token+v] = float64(f.c.Scale) * dot
				}
			}
		}
	}
	return state, out
}

func kdaInputBytes(f kdaFixture) []byte {
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, []int32{int32(f.c.Tokens), int32(f.c.Heads), int32(f.c.Sequences)})
	binary.Write(&b, binary.LittleEndian, f.c.Scale)
	for _, a := range [][]float32{f.q, f.key, f.value, f.decay, f.beta, f.state} {
		binary.Write(&b, binary.LittleEndian, a)
	}
	binary.Write(&b, binary.LittleEndian, f.offsets)
	return b.Bytes()
}

func kdaCompare(got []float32, want []float64, f kdaFixture, source, target string) error {
	if len(got) != len(want) {
		return fmt.Errorf("kda result extent mismatch")
	}
	rounded := make([]float32, len(want))
	for i, x := range want {
		rounded[i] = float32(x)
	}
	sd := sha256.Sum256([]byte(source))
	id := sha256.Sum256(kdaInputBytes(f))
	p := cudatest.CUDAOutputPolicy{Target: target, DType: "fp32", MathMode: KDAFP32, SourceDigest: hex.EncodeToString(sd[:]), InputDigest: hex.EncodeToString(id[:]), Elements: len(want), Absolute: kdaAbsolute, Relative: kdaRelative, NaN: "reject", Infinity: "reject", Zero: "equal"}
	if err := cudatest.CompareCUDAOutputFP32(got, rounded, p); err != nil {
		return err
	}
	for i, x := range got {
		if !kdaFinite(x) || math.IsNaN(want[i]) || math.IsInf(want[i], 0) || math.Abs(float64(x)-want[i]) > kdaAbsolute+kdaRelative*math.Max(math.Abs(float64(x)), math.Abs(want[i])) {
			return fmt.Errorf("kda FP64 oracle mismatch at %d: %g vs %.17g", i, x, want[i])
		}
	}
	return nil
}

// Host compilation executes the exact CUDA recurrence body with separately
// rounded FP32 intrinsic shims. It verifies the source arithmetic, not CUDA
// compilation, device execution, launch ownership or hardware rounding.
func kdaHostKernel(t *testing.T, source string) func(kdaFixture) ([]float32, []float32, int32) {
	t.Helper()
	kdaHostBuild.Lock()
	defer kdaHostBuild.Unlock()
	digest := kdaDigestHost(source)
	binaryPath := kdaHostBuild.binaries[digest]
	if binaryPath == "" {
		if _, err := exec.LookPath("c++"); err != nil {
			t.Skip("host CUDA source arithmetic shim requires a C++ compiler; not verified")
		}
		dir, err := os.MkdirTemp("", "kda-host-source-")
		if err != nil {
			t.Fatal(err)
		}
		kdaHostBuild.dirs = append(kdaHostBuild.dirs, dir)
		path := filepath.Join(dir, "kernel.cpp")
		binaryPath = filepath.Join(dir, "kernel")
		shim := `#include <cstdio>
#include <vector>
#include <cstring>
#define __global__
#define __device__
#define __forceinline__ inline
struct { int x=0; } blockIdx,threadIdx;
unsigned __float_as_uint(float x){unsigned bits;std::memcpy(&bits,&x,4);return bits;}
float __fmul_rn(float a,float b){volatile float c=a*b;return c;}
float __fadd_rn(float a,float b){volatile float c=a+b;return c;}
float __fsub_rn(float a,float b){volatile float c=a-b;return c;}
`
		main := `
int main(){int t,h,n;float scale;
if(fread(&t,4,1,stdin)!=1||fread(&h,4,1,stdin)!=1||fread(&n,4,1,stdin)!=1||fread(&scale,4,1,stdin)!=1)return 2;
if(t<0||t>66||h<1||h>2||n<1||n>2)return 3;
int vc=t*h*128,sc=n*h*128*128;
std::vector<float> q(vc),k(vc),v(vc),d(vc),b(t*h),initial(sc),state(sc,-12345),out(vc,-12345);
for(auto a:{&q,&k,&v,&d,&b,&initial})if(fread(a->data(),4,a->size(),stdin)!=a->size())return 4;
std::vector<int> offsets(n+1);if(fread(offsets.data(),4,n+1,stdin)!=n+1)return 5;
int status=-1;kda_fp32(q.data(),k.data(),v.data(),d.data(),b.data(),initial.data(),offsets.data(),state.data(),out.data(),&status,n,h,scale);
fwrite(state.data(),4,sc,stdout);fwrite(out.data(),4,vc,stdout);fwrite(&status,4,1,stdout);return 0;}
`
		if err := os.WriteFile(path, []byte(shim+source+main), 0600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("c++", "-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math", path, "-o", binaryPath)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("host kernel build: %v\n%s", err, out)
		}
		if kdaHostBuild.binaries == nil {
			kdaHostBuild.binaries = make(map[string]string)
		}
		kdaHostBuild.binaries[digest] = binaryPath
	}
	return func(f kdaFixture) ([]float32, []float32, int32) {
		t.Helper()
		cmd := exec.Command(binaryPath)
		cmd.Stdin = bytes.NewReader(kdaInputBytes(f))
		data, err := cmd.Output()
		if err != nil {
			t.Fatal(err)
		}
		state := make([]float32, len(f.state))
		out := make([]float32, len(f.q))
		var status int32
		r := bytes.NewReader(data)
		for _, a := range []any{state, out, &status} {
			if err := binary.Read(r, binary.LittleEndian, a); err != nil {
				t.Fatal(err)
			}
		}
		if r.Len() != 0 {
			t.Fatal("trailing kernel output")
		}
		return state, out, status
	}
}

func kdaDigestHost(source string) string {
	sum := sha256.Sum256([]byte(source))
	return hex.EncodeToString(sum[:])
}
