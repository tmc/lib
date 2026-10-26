package egraph_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func declarativeMathExpr(t testing.TB, kind string) *egraph.Expr {
	t.Helper()
	if kind == "mma" {
		return egraph.NewNodeExpr("tile_mma", egraph.NewNodeExpr("a"), egraph.NewNodeExpr("b"))
	}
	g := egraph.New()
	x := g.Add(egraph.NewLeaf("x"))
	w := g.Add(egraph.NewLeaf("w"))
	down := g.Add(egraph.NewLeaf("down"))
	gate := g.Add(egraph.NewNode("matmul", x, w))
	activation := "gelu"
	if kind == "swiglu" {
		activation = "silu"
	}
	mid := g.Add(egraph.NewNode(activation, gate))
	if kind != "gelu" {
		up := g.Add(egraph.NewNode("matmul", x, g.Add(egraph.NewLeaf("up"))))
		mid = g.Add(egraph.NewNode("mul", mid, up))
	}
	root := g.Add(egraph.NewNode("matmul", mid, down))
	egraph.Saturate(g, egraph.TransformerTileRules(), 10)
	expr, _, err := egraph.ExtractBest(g, root, egraph.FusedMegakernelCost{})
	if err != nil || expr == nil || expr.Op != "tile_"+kind {
		t.Fatalf("ordinary public extraction %s = %v, %v", kind, expr, err)
	}
	return expr
}

func declarativeMathSources(t testing.TB) map[string]string {
	t.Helper()
	sources := make(map[string]string)
	for _, kind := range []string{"mma", "swiglu", "geglu", "gelu"} {
		source, err := egraph.GenerateDeclarativeTile(declarativeMathExpr(t, kind), egraph.CodegenOptions{KernelName: "math_" + kind, Precision: "float"})
		if err != nil {
			t.Fatal(err)
		}
		sources[kind] = source
	}
	return sources
}

func TestDeclarativeTileMathAdmission(t *testing.T) {
	for _, family := range [][]string{
		{"tile_mma", "tile_mma_aligned", "mma"},
		{"tile_swiglu", "fused_swiglu"},
		{"tile_geglu", "fused_geglu"},
		{"tile_gelu", "fused_gelu"},
	} {
		kind := strings.TrimPrefix(family[0], "tile_")
		expr := declarativeMathExpr(t, kind)
		for _, precision := range []string{"float", "float32", "half", "float16", "bf16", "bfloat16", "f8", "fp8", "f4", "fp4", "q8_0"} {
			var first string
			for i, op := range family {
				alias := *expr
				alias.Op = op
				source, err := egraph.GenerateDeclarativeTile(&alias, egraph.CodegenOptions{KernelName: "same", Precision: precision})
				if err != nil || !egraph.IsLowerableTileIROp(op) {
					t.Fatalf("ordinary alias %s/%s: %v", op, precision, err)
				}
				if i == 0 {
					first = source
				} else if source != first {
					t.Fatalf("ordinary alias %s changes source", op)
				}
			}
		}
	}
	for _, c := range declarativeMathCases(false) {
		value := float64(c.want[0])
		switch {
		case strings.HasPrefix(c.name, "positive_inf_"):
			if !math.IsInf(value, 1) {
				t.Fatalf("%s dense reference = %v, want +Inf", c.name, value)
			}
		case strings.HasPrefix(c.name, "negative_output_inf_"):
			if !math.IsInf(value, -1) {
				t.Fatalf("%s dense reference = %v, want -Inf", c.name, value)
			}
		case strings.HasPrefix(c.name, "negative_gate_inf_"), strings.HasPrefix(c.name, "valid_nan_"):
			if !math.IsNaN(value) {
				t.Fatalf("%s dense reference = %v, want NaN", c.name, value)
			}
		}
	}
}

type declarativeMathCase struct {
	name, kind string
	m, n, k    int
	x, w, u, d []float32
	want       []float32
}

// These references implement the whole dense operations, without consulting
// a Tile plan, source template or primitive shim. Every product is rounded.
func declarativeDenseProduct(a, b []float32, rows, cols, terms int) []float32 {
	out := make([]float32, rows*cols)
	for i := range rows {
		for j := range cols {
			for k := range terms {
				product := float32(a[i*terms+k] * b[k*cols+j])
				out[i*cols+j] = float32(float64(out[i*cols+j]) + float64(product))
			}
		}
	}
	return out
}

func declarativeDenseReference(c declarativeMathCase) []float32 {
	if c.kind == "mma" {
		return declarativeDenseProduct(c.x, c.w, c.m, c.n, c.k)
	}
	gate := declarativeDenseProduct(c.x, c.w, c.m, c.n, c.k)
	var up []float32
	if c.kind != "gelu" {
		up = declarativeDenseProduct(c.x, c.u, c.m, c.n, c.k)
	}
	for i, value := range gate {
		x := float64(value)
		if c.kind == "swiglu" {
			gate[i] = float32(x / (1 + math.Exp(-x)))
		} else {
			gate[i] = float32(0.5 * x * (1 + math.Erf(x/math.Sqrt2)))
		}
		if up != nil {
			gate[i] = float32(gate[i] * up[i])
		}
	}
	return declarativeDenseProduct(gate, c.d, c.m, c.k, c.n)
}

func declarativeMathCases(primaryOnly bool) []declarativeMathCase {
	var cases []declarativeMathCase
	pattern := func(n, salt int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((i*7+salt)%17-8) / 16
		}
		return v
	}
	add := func(name, kind string, m, n, k int, primary bool) {
		c := declarativeMathCase{name: name, kind: kind, m: m, n: n, k: k,
			x: pattern(m*k, 1), w: pattern(k*n, 2), u: pattern(k*n, 3), d: pattern(n*k, 4)}
		if primary {
			clear(c.x)
			clear(c.w)
			clear(c.u)
			clear(c.d)
			for i := 0; i < 16; i++ {
				c.x[i*16+i] = 1
				for j := 16; j < 32; j++ {
					c.w[i*32+j], c.u[i*32+j], c.d[j*16+i] = 1, 1, 1
				}
			}
		}
		if name == "mma_aligned_output" {
			for i := range c.x {
				c.x[i] = 1
			}
			for i := range c.w {
				c.w[i] = 1
			}
		}
		c.want = declarativeDenseReference(c)
		cases = append(cases, c)
	}
	if primaryOnly {
		add("one_tile_swiglu", "swiglu", 16, 16, 16, false)
		add("primary_swiglu", "swiglu", 16, 32, 16, true)
		return cases
	}
	for _, kind := range []string{"mma", "swiglu", "geglu", "gelu"} {
		add("one_tile_"+kind, kind, 16, 16, 16, false)
	}
	for _, kind := range []string{"swiglu", "geglu", "gelu"} {
		add("primary_"+kind, kind, 16, 32, 16, true)
	}
	add("mma_aligned_output", "mma", 32, 32, 16, false)
	for _, dims := range [][3]int{{1, 1, 1}, {15, 15, 15}, {16, 16, 16}, {17, 17, 17}, {31, 31, 31}, {32, 32, 32}, {17, 31, 15}, {31, 17, 32}, {15, 32, 17}} {
		for _, kind := range []string{"mma", "swiglu", "geglu", "gelu"} {
			add(fmt.Sprintf("edges_%s_%d_%d_%d", kind, dims[0], dims[1], dims[2]), kind, dims[0], dims[1], dims[2], false)
		}
	}
	// Padding is not a dense reduction term. Inactive hidden lanes must
	// contribute zero even when their intermediate activation becomes NaN.
	for _, kind := range []string{"swiglu", "geglu", "gelu"} {
		for _, tt := range []struct {
			name    string
			x, w, d float32
		}{
			{"positive_inf", float32(math.Inf(1)), 1, 1},
			{"negative_output_inf", float32(math.Inf(1)), 1, -1},
			{"negative_gate_inf", float32(math.Inf(1)), -1, 1},
			{"valid_nan", float32(math.Inf(1)), 0, 1},
		} {
			c := declarativeMathCase{name: tt.name + "_" + kind, kind: kind, m: 1, n: 17, k: 1,
				x: []float32{tt.x}, w: make([]float32, 17), u: make([]float32, 17), d: make([]float32, 17)}
			for i := range c.w {
				c.w[i], c.u[i], c.d[i] = tt.w, 1, tt.d
			}
			c.want = declarativeDenseReference(c)
			cases = append(cases, c)
		}
	}
	return cases
}

func declarativeMathProgram(sources map[string]string, primaryOnly bool) string {
	var b strings.Builder
	b.WriteString("#include \"cuda_tile.h\"\n")
	for _, kind := range []string{"mma", "swiglu", "geglu", "gelu"} {
		b.WriteString(sources[kind])
	}
	b.WriteString("int main() {\n")
	for _, c := range declarativeMathCases(primaryOnly) {
		b.WriteString("{\n")
		for i, values := range [][]float32{c.x, c.w, c.u, c.d, c.want} {
			fmt.Fprintf(&b, "std::vector<float> v%d = {", i)
			for _, v := range values {
				switch {
				case math.IsNaN(float64(v)):
					b.WriteString("std::numeric_limits<float>::quiet_NaN(),")
					continue
				case math.IsInf(float64(v), 1):
					b.WriteString("std::numeric_limits<float>::infinity(),")
					continue
				case math.IsInf(float64(v), -1):
					b.WriteString("-std::numeric_limits<float>::infinity(),")
					continue
				}
				literal := strconv.FormatFloat(float64(v), 'g', -1, 32)
				if !strings.ContainsAny(literal, ".eE") {
					literal += ".0"
				}
				b.WriteString(literal + "f,")
			}
			b.WriteString("};\n")
		}
		fmt.Fprintf(&b, "std::vector<float> out(v4.size(), -7.0f);\nmodel_begin({{v0.data(),v0.size()},{v1.data(),v1.size()},{v2.data(),v2.size()},{v3.data(),v3.size()}},out,%t);\n", primaryOnly)
		var args string
		switch c.kind {
		case "mma":
			args = "v0.data(),v1.data(),out.data()"
		case "gelu":
			args = "v0.data(),v1.data(),v3.data(),out.data()"
		default:
			args = "v0.data(),v1.data(),v2.data(),v3.data(),out.data()"
		}
		call := fmt.Sprintf("math_%s(%s,%d,%d,%d);\n", c.kind, args, c.m, c.n, c.k)
		b.WriteString(call)
		fmt.Fprintf(&b, "model_check(out,v4,%q);\n", c.name)
		if !primaryOnly {
			for _, bid := range []string{"{1,0,0}", "{0,1,0}", "{0,0,1}"} {
				b.WriteString("cuda::tiles::model_bid=" + bid + ";\n" + call)
				fmt.Fprintf(&b, "model_check(out,v4,%q);\n", c.name)
			}
			b.WriteString("cuda::tiles::model_bid={0,0,0};\n")
		}
		fmt.Fprintf(&b, "std::cout << %q << '\\n';\n}\n", "PASS:"+c.name)
	}
	if !primaryOnly {
		// Invalid shapes must return before touching even these tiny buffers.
		b.WriteString(`float x[1]={1}, w[1]={2}, u[1]={3}, d[1]={4}, out[1]={-7};
for (auto dims : std::vector<std::array<int,3>>{{0,1,1},{1,0,1},{1,1,0},{-1,1,1},{1,-1,1},{1,1,-1},{INT_MAX,INT_MAX,INT_MAX}}) {
  model_writes.clear();
  math_mma(x,w,out,dims[0],dims[1],dims[2]);
  math_swiglu(x,w,u,d,out,dims[0],dims[1],dims[2]);
  math_geglu(x,w,u,d,out,dims[0],dims[1],dims[2]);
  math_gelu(x,w,d,out,dims[0],dims[1],dims[2]);
  if (out[0] != -7) model_fail("invalid extent wrote output");
}
`)
	}
	b.WriteString("return 0;\n}\n")
	return b.String()
}

func runDeclarativeMathProgram(t *testing.T, program string) (string, error) {
	t.Helper()
	cxx := os.Getenv("EGRAPH_TILE_CPU_CXX")
	if cxx == "" {
		t.Skip("emitted-source CPU model requires explicit EGRAPH_TILE_CPU_CXX")
	}
	if !filepath.IsAbs(cxx) || len(program) > 2<<20 {
		t.Fatal("absolute admitted host C++ compiler and bounded source required")
	}
	dir := t.TempDir()
	for name, text := range map[string]string{"cuda_tile.h": declarativeTileCPUHeader, "math.cc": program} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(text), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(t.Context(), 75*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, cxx, "-std=c++20", "-O0", "-ffp-contract=off", "-ferror-limit=4", "-ftemplate-backtrace-limit=4", "-I", dir, filepath.Join(dir, "math.cc"), "-o", filepath.Join(dir, "math"))
	cmd.WaitDelay = 3 * time.Second
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("CPU model compile (not an omission kill): %v\n%s", err, out)
	}
	runCtx, runCancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer runCancel()
	cmd = exec.CommandContext(runCtx, filepath.Join(dir, "math"))
	cmd.WaitDelay = 3 * time.Second
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func TestDeclarativeTileMath(t *testing.T) {
	primary := os.Getenv("EGRAPH_TILE_CPU_PRIMARY_ONLY") == "1"
	out, err := runDeclarativeMathProgram(t, declarativeMathProgram(declarativeMathSources(t), primary))
	if err != nil {
		t.Fatalf("emitted-source numerical assertion: %v\n%s", err, out)
	}
	t.Log(out)
}

func TestDeclarativeTileMathOmissions(t *testing.T) {
	for _, tt := range []struct{ name, kind, old, replacement, assertion string }{
		{"hidden_block", "swiglu", "(long long)n * 16, K, N", "0LL, K, N", "ASSERT numeric"},
		{"output_block", "mma", "egraph_tile_store(C, sum, (long long)m * 16, (long long)n * 16, M, N);", "egraph_tile_store(C, sum, 0LL, 0LL, M, N);", "ASSERT stores"},
		{"reduction_tail", "mma", "(K - 1) / 16 + 1", "K / 16", "ASSERT numeric"},
		{"hidden_nonfinite_padding", "swiglu", "act = select(hidden < (long long)N, act, full<decltype(act)>(0.0f));", "", "ASSERT numeric classification"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			sources := declarativeMathSources(t)
			if !strings.Contains(sources[tt.kind], tt.old) {
				t.Fatal("omission source precondition absent")
			}
			sources[tt.kind] = strings.ReplaceAll(sources[tt.kind], tt.old, tt.replacement)
			out, err := runDeclarativeMathProgram(t, declarativeMathProgram(sources, false))
			var exit *exec.ExitError
			if !errors.As(err, &exit) || exit.ExitCode() != 42 || !strings.Contains(out, tt.assertion) || !strings.Contains(out, "PASS:one_tile_"+tt.kind) {
				t.Fatalf("omission did not fail its intended runtime assertion after its positive: %v\n%s", err, out)
			}
			t.Log(out)
		})
	}
}

// A closed CPU model of the emitted tile primitives, not a CUDA header or a
// second FFN implementation. All indexing, loop structure and activations
// above the primitive MMA live in the actual generated kernel source.
// partition_view is retained only so the aligned predecessor compiles too.
const declarativeTileCPUHeader = `#pragma once
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>
#define __tile_global__
[[noreturn]] inline void model_fail(const char* text) { std::cerr << "ASSERT " << text << '\n'; std::exit(42); }
struct model_span { const float* base; size_t count; };
inline std::vector<model_span> model_spans;
inline float* model_out;
inline std::vector<int> model_writes;
inline void model_address(const float* p) {
  auto a = reinterpret_cast<uintptr_t>(p);
  for (auto s : model_spans) {
    auto b = reinterpret_cast<uintptr_t>(s.base);
    if (a >= b && a-b < s.count*sizeof(float) && (a-b)%sizeof(float)==0) return;
  }
  model_fail("address");
}
inline void model_store(float* p, float value) {
  model_address(p);
  auto a = reinterpret_cast<uintptr_t>(p), b = reinterpret_cast<uintptr_t>(model_out);
  if (a < b || (a-b)/sizeof(float) >= model_writes.size()) model_fail("stores outside output");
  auto i = (a-b)/sizeof(float);
  if (++model_writes[i] != 1) model_fail("stores duplicate");
  *p=value;
}
inline void model_begin(std::vector<model_span> spans, std::vector<float>& out, bool require_aligned) {
  model_spans=spans; model_spans.push_back({out.data(),out.size()}); model_out=out.data(); model_writes.assign(out.size(),0);
  if (require_aligned) for (auto s:model_spans) if (reinterpret_cast<uintptr_t>(s.base)%16) {std::cerr << "SETUP alignment\n";std::exit(43);}
}
inline void model_check(const std::vector<float>& out, const std::vector<float>& want, const char* name) {
  for (size_t i=0;i<want.size();++i) {
    if (model_writes[i]!=1) model_fail("stores incomplete");
    bool same_class = std::isnan(want[i]) ? std::isnan(out[i]) :
      std::isinf(want[i]) ? out[i]==want[i] : std::isfinite(out[i]);
    if (!same_class) {
      std::cerr << name << " index=" << i << " got=" << out[i] << " want=" << want[i] << '\n'; model_fail("numeric classification");
    }
    if (std::isfinite(want[i]) && std::abs(out[i]-want[i]) > 0.0002f*(1+std::abs(want[i]))) {
      std::cerr << name << " index=" << i << " got=" << out[i] << " want=" << want[i] << '\n'; model_fail("numeric");
    }
  }
}
namespace cuda::tiles {
template<int... D> struct shape { static constexpr int count=(D*...*1); };
template<class T,class S> struct tile { using value_type=T; using shape_type=S; std::array<T,S::count> v{}; };
template<class> struct is_tile:std::false_type {};
template<class T,class S> struct is_tile<tile<T,S>>:std::true_type {};
template<class V,class U> V full(U value) { V out; for(auto& x:out.v)x=static_cast<typename V::value_type>(value);return out; }
template<class V> V iota() { V out;for(int i=0;i<V::shape_type::count;++i)out.v[i]=i;return out; }
template<class S,class T,class Old> auto reshape(tile<T,Old> a) {static_assert(S::count==Old::count);tile<T,S>out;out.v=a.v;return out;}
#define MODEL_BINARY(OP) \
template<class A,class B,class S> auto operator OP(tile<A,S>a,tile<B,S>b){tile<decltype(a.v[0] OP b.v[0]),S>o;for(int i=0;i<S::count;++i)o.v[i]=a.v[i] OP b.v[i];return o;} \
template<class A,class B,class S,std::enable_if_t<!is_tile<B>::value,int> =0> auto operator OP(tile<A,S>a,B b){return a OP full<tile<B,S>>(b);} \
template<class A,class B,class S,std::enable_if_t<!is_tile<A>::value,int> =0> auto operator OP(A a,tile<B,S>b){return full<tile<A,S>>(a) OP b;}
MODEL_BINARY(+) MODEL_BINARY(-) MODEL_BINARY(*) MODEL_BINARY(/) MODEL_BINARY(%) MODEL_BINARY(<)
#undef MODEL_BINARY
template<class S> auto operator&(tile<bool,S>a,tile<bool,S>b){for(int i=0;i<S::count;++i)a.v[i]=a.v[i]&&b.v[i];return a;}
template<class T,class S> auto operator-(tile<T,S>a){for(auto&x:a.v)x=-x;return a;}
template<class T,class S> auto exp(tile<T,S>a){for(auto&x:a.v)x=std::exp(x);return a;}
template<class T,class S> auto erf(tile<T,S>a){for(auto&x:a.v)x=std::erf(x);return a;}
template<class T,class S> auto select(tile<bool,S>m,tile<T,S>a,tile<T,S>b){for(int i=0;i<S::count;++i)if(!m.v[i])a.v[i]=b.v[i];return a;}
template<class T,class S> auto load_masked(tile<T*,S>p,tile<bool,S>m,tile<std::remove_const_t<T>,S>z){for(int i=0;i<S::count;++i)if(m.v[i]){model_address(p.v[i]);z.v[i]=*p.v[i];}return z;}
template<class T,class V,class S> void store_masked(tile<T*,S>p,tile<V,S>v,tile<bool,S>m){for(int i=0;i<S::count;++i)if(m.v[i])model_store(p.v[i],v.v[i]);}
template<class T,int M,int K,class U,int N,class V> auto mma(tile<T,shape<M,K>>a,tile<U,shape<K,N>>b,tile<V,shape<M,N>>c){
  for(int i=0;i<M;++i)for(int j=0;j<N;++j)for(int k=0;k<K;++k)c.v[i*N+j]=V(c.v[i*N+j]+V(a.v[i*K+k]*b.v[k*N+j]));return c;
}
struct block_id {int x,y,z;}; inline block_id model_bid{0,0,0}; inline auto bid(){return model_bid;}
struct integer_range {int first,last;struct iterator{int value;int operator*()const{return value;}void operator++(){++value;}bool operator!=(iterator b)const{return value!=b.value;}};auto begin()const{return iterator{first};}auto end()const{return iterator{last};}};
inline auto irange(int first,int last){return integer_range{first,last};}
template<int N,class T> T* assume_aligned(T*p){return p;}
struct extents {int rows,cols;};
template<class T> struct tensor_span {T*base;extents size;};
template<class T> tensor_span(T*,extents)->tensor_span<T>;
template<class T,int R,int C> struct partition_view {
  tensor_span<T> span; shape<R,C> form;
  auto load(int r,int c)const {tile<std::remove_const_t<T>,shape<R,C>>o;for(int i=0;i<R;++i)for(int j=0;j<C;++j)if(r*R+i<span.size.rows&&c*C+j<span.size.cols){auto p=span.base+(r*R+i)*span.size.cols+c*C+j;model_address(p);o.v[i*C+j]=*p;}return o;}
  template<class V> void store(V value,int r,int c){for(int i=0;i<R;++i)for(int j=0;j<C;++j)if(r*R+i<span.size.rows&&c*C+j<span.size.cols)model_store(span.base+(r*R+i)*span.size.cols+c*C+j,value.v[i*C+j]);}
};
template<class T,int R,int C> partition_view(tensor_span<T>,shape<R,C>)->partition_view<T,R,C>;
}
`
