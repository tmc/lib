package tileir_test

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func TestSemanticSourceHostOracle(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable; source execution unverified")
	}
	cases := []struct {
		name         string
		graph        func() *cudair.Graph
		arrays, call string
		want         []float32
	}{
		{"subtract", func() *cudair.Graph {
			g := cudair.NewGraph()
			a := g.AddInput("a", []int{3}, cudair.Float32)
			b := g.AddInput("b", []int{3}, cudair.Float32)
			g.AddBinary("sub", a, b)
			return g
		}, "float a[]={9,-3,7}, b[]={2,4,-1};", "k(a,b,out);", []float32{7, -7, 8}},
		{"sum rows", func() *cudair.Graph {
			g := cudair.NewGraph()
			a := g.AddInput("a", []int{2, 3}, cudair.Float32)
			g.AddReduce("sum", a, 1)
			return g
		}, "float a[]={1,2,3,4,5,6};", "k(a,out);", []float32{6, 15}},
		{"max columns", func() *cudair.Graph {
			g := cudair.NewGraph()
			a := g.AddInput("a", []int{2, 3}, cudair.Float32)
			g.AddReduce("max", a, 0)
			return g
		}, "float a[]={1,8,-3,4,5,-1};", "k(a,out);", []float32{4, 8, -1}},
		{"mean", func() *cudair.Graph {
			g := cudair.NewGraph()
			a := g.AddInput("a", []int{3}, cudair.Float32)
			g.AddReduce("mean", a, 0)
			return g
		}, "float a[]={3,6,9};", "k(a,out);", []float32{6}},
		{"matmul", func() *cudair.Graph {
			g := cudair.NewGraph()
			a := g.AddInput("a", []int{2, 3}, cudair.Float32)
			b := g.AddInput("b", []int{3, 2}, cudair.Float32)
			g.AddMatMul(a, b)
			return g
		}, "float a[]={1,2,3,4,5,6}, b[]={7,8,9,10,11,12};", "k(a,b,out);", []float32{58, 64, 139, 154}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			snap, err := tc.graph().Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			source, _, err := tileir.GenerateSemanticTileCPP(snap, tileir.LowerOptions{KernelName: "k"})
			if err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			// This is an independent scalar host interpreter for the documented Tile
			// operations, not NVIDIA compilation or a GPU execution qualification.
			header := `#define __tile_global__
static int testIndex;
namespace cuda { namespace tiles {
struct dim {int x;}; inline dim bid(){return {testIndex};}
inline float add(float a,float b){return a+b;}
inline float sub(float a,float b){return a-b;}
inline float mul(float a,float b){return a*b;}
inline float div(float a,float b){return a/b;}
inline float max(float a,float b){return a>b?a:b;}
}}
`
			if err := os.WriteFile(filepath.Join(dir, "cuda_tile.h"), []byte(header), 0600); err != nil {
				t.Fatal(err)
			}
			var checks strings.Builder
			for i, w := range tc.want {
				fmt.Fprintf(&checks, "if(out[%d]!=%g) return %d;\n", i, w, i+1)
			}
			program := source + fmt.Sprintf("\nint main(){%s float out[%d]; for(int i=0;i<%d;i++)out[i]=999; for(testIndex=0;testIndex<%d;testIndex++){%s} %s if(out[%d]!=999 || out[%d]!=999)return 90; return 0;}\n", tc.arrays, len(tc.want)+2, len(tc.want)+2, len(tc.want)+2, tc.call, checks.String(), len(tc.want), len(tc.want)+1)
			path := filepath.Join(dir, "source.cc")
			if err := os.WriteFile(path, []byte(program), 0600); err != nil {
				t.Fatal(err)
			}
			bin := filepath.Join(dir, "oracle")
			if out, err := exec.Command(compiler, "-std=c++20", "-I", dir, path, "-o", bin).CombinedOutput(); err != nil {
				t.Fatalf("host compile: %v\n%s\n%s", err, out, program)
			}
			if out, err := exec.Command(bin).CombinedOutput(); err != nil {
				t.Fatalf("independent output/guard oracle: %v %s", err, out)
			}
		})
	}
}

func TestSemanticSourceRefusals(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{65}, cudair.Float32)
	g.AddReduce("sum", a, 0)
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := tileir.GenerateSemanticTileCPP(snap, tileir.LowerOptions{}); err == nil {
		t.Fatal("unbounded reduction accepted")
	}
	g = cudair.NewGraph()
	a = g.AddInput("a", []int{4}, cudair.Int32)
	g.AddUnary("neg", a)
	snap, err = g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := tileir.GenerateSemanticTileCPP(snap, tileir.LowerOptions{}); err == nil {
		t.Fatal("integer arithmetic silently converted to float")
	}
	g = cudair.NewGraph()
	a = g.AddInput("a", []int{4}, cudair.Float32)
	g.AddUnary("neg", a)
	snap, err = g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := tileir.GenerateSemanticTileCPP(snap, tileir.LowerOptions{KernelName: "k) {}"}); err == nil {
		t.Fatal("source injection accepted")
	}
	if !strings.Contains(tileir.GeneratePinnedTileCPP("old", nil, 16, 16, 16), "#error") {
		t.Fatal("shape-only generator silently emitted executable empty body")
	}
}

func TestSubtractionIsNotAddition(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{16}, cudair.Float32)
	b := g.AddInput("b", []int{16}, cudair.Float32)
	g.AddBinary("sub", a, b)
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	r, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, op := range r.Module.AllOperations() {
		if op.OpCode == "cuda_tile.subf" {
			found = true
		}
	}
	if !found || !strings.Contains(r.Source, "ct::sub(") {
		t.Fatal("semantic subtraction disappeared")
	}
}

func ExampleGenerateSemanticTileCPP() {
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{2}, cudair.Float32)
	g.AddReduce("sum", a, 0)
	snap, _ := g.Snapshot()
	source, schema, err := tileir.GenerateSemanticTileCPP(snap, tileir.LowerOptions{KernelName: "sum"})
	fmt.Println(len(schema.Params), strings.Contains(source, "ct::add("), err)
	// Output: 2 true <nil>
}

func TestSemanticSourceCUDACompile(t *testing.T) {
	if os.Getenv("CUDA_TILE_COMPILE_TEST") != "1" {
		t.Skip("CUDA 13.4 source compilation gate disabled")
	}
	compiler := os.Getenv("NVCC")
	if compiler == "" {
		compiler = "nvcc"
	}
	path, err := exec.LookPath(compiler)
	if err != nil {
		t.Fatal(err)
	}
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{2, 3}, cudair.Float32)
	g.AddReduce("sum", a, 1)
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	source, _, err := tileir.GenerateSemanticTileCPP(snap, tileir.LowerOptions{KernelName: "sum_rows"})
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	input := filepath.Join(dir, "source.cu")
	if err := os.WriteFile(input, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	// Compilation creates a host object containing Tile device code. It does
	// not load a CUDA context or launch a device kernel.
	output := filepath.Join(dir, "source.o")
	if out, err := exec.CommandContext(ctx, path, "--enable-tile", "--tile-only", "-std=c++20", "-arch=sm_90", "-c", input, "-o", output).CombinedOutput(); err != nil {
		t.Fatalf("actual CUDA source compile: %v\n%s", err, out)
	}
	info, err := os.Stat(output)
	if err != nil || info.Size() == 0 {
		t.Fatalf("missing compiled object: %v", err)
	}
}

func TestSemanticTileCompileRefusals(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, c := range []context.Context{nil, ctx} {
		if _, _, err := tileir.CompileSemanticTile(c, nil, tileir.LowerOptions{}, 9, 0); err == nil {
			t.Fatal("missing or cancelled context accepted")
		}
	}
	if _, _, err := tileir.CompileSemanticTile(context.Background(), nil, tileir.LowerOptions{}, 7, 0); err == nil {
		t.Fatal("pre-Tile target accepted")
	}
}

func TestSemanticTileNVRTCCompile(t *testing.T) {
	if os.Getenv("CUDA_TILE_NVRTC_TEST") != "1" {
		t.Skip("actual NVRTC semantic artifact gate disabled")
	}
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{2, 3}, cudair.Float32)
	g.AddReduce("sum", a, 1)
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	artifact, log, err := tileir.CompileSemanticTile(ctx, snap, tileir.LowerOptions{KernelName: "sum_rows"}, 9, 0)
	if err != nil {
		t.Fatal(err)
	}
	if artifact.Kind != cudakernel.KindTileIR || len(artifact.Raw) == 0 || len(artifact.Schema.Params) != 2 || len(artifact.CompilerOpts) == 0 {
		t.Fatal("missing authentic artifact, ABI or compile identity")
	}
	if err := artifact.Validate(); err != nil {
		t.Fatal(err)
	}
	t.Logf("artifact=%s bytes=%d target=%s compiler=%s flags=%q log=%q", artifact.Digest, len(artifact.Raw), artifact.Target, artifact.Toolchain, artifact.CompilerOpts, log)
}

func ExampleCompileSemanticTile() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err := tileir.CompileSemanticTile(ctx, nil, tileir.LowerOptions{}, 9, 0)
	fmt.Println(err)
	// Output: context canceled
}
