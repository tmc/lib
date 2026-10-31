package frontend

//go:generate sh -c "cd testdata && protoc --encode=onnx.ModelProto onnx_subset.proto < llama_mlp.textproto | xxd -p > llama_mlp.onnx.hex"

import (
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func readTestdata(t testing.TB, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func onnxFixture(t testing.TB) []byte {
	t.Helper()
	b, err := hex.DecodeString(strings.Join(strings.Fields(string(readTestdata(t, "llama_mlp.onnx.hex"))), ""))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// importAll imports the same Llama MLP sub-block through each frontend.
func importAll(t *testing.T) map[string]ImportResult {
	t.Helper()
	_, hf, err := ImportHFConfig(readTestdata(t, "llama.json"), 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	shlo, err := ImportStableHLO(string(readTestdata(t, "llama_mlp.mlir")))
	if err != nil {
		t.Fatal(err)
	}
	onnx, err := ImportONNXWire(onnxFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	return map[string]ImportResult{"hf": hf, "stablehlo": shlo, "onnx": onnx}
}

func planShape(p *hopper.Plan) []string {
	var out []string
	for _, k := range p.Kernels {
		switch {
		case k.GEMM != nil:
			out = append(out, fmt.Sprintf("gemm M=%d", k.M))
		case k.Tile != nil:
			out = append(out, fmt.Sprintf("tile M=%d", k.M))
		case k.Row != nil:
			out = append(out, fmt.Sprintf("row rows=%d", k.Rows))
		default:
			out = append(out, "other")
		}
	}
	return out
}

// TestCrossFrontendConvergence imports one MLP sub-block from a config,
// from StableHLO and from ONNX and checks the optimized terms and plans
// agree, and that saturation fused the block.
func TestCrossFrontendConvergence(t *testing.T) {
	res := importAll(t)
	want := CanonicalFingerprint(res["hf"].Expr)
	for name, r := range res {
		if got := CanonicalFingerprint(r.Expr); got != want {
			t.Errorf("%s term differs from hf:\n%s\n%s", name, got, want)
		}
		if got, want := planShape(r.Plan), planShape(res["hf"].Plan); !slices.Equal(got, want) {
			t.Errorf("%s plan %v, hf plan %v", name, got, want)
		}
	}
	e := res["hf"].Expr
	if e.Op != hopper.OpGEMM || e.Children[0].Op != "fn:(add $0 $1)" {
		t.Fatalf("root is %s %s, want the down projection with the residual in its epilogue", e.Op, e.Children[0].Op)
	}
	src := e.String()
	for _, want := range []string{
		"fn:(mul (mul $0 (rsqrt (add (div $1 256) 1e-05))) $2)",
		"fn:(mul (mul $0 (sigmoid $0)) $1)",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("optimized term lacks fused %s:\n%s", want, src)
		}
	}
	for _, r := range []ImportResult{res["stablehlo"], res["onnx"]} {
		for _, w := range r.Weights {
			if len(w.Shape) == 2 && !w.Transposed {
				t.Errorf("weight %s stored [in, out] is not marked transposed", w.Name)
			}
		}
	}
}

// TestConvergenceNegative changes one constant in the StableHLO module
// and checks the terms no longer agree.
func TestConvergenceNegative(t *testing.T) {
	_, hf, err := ImportHFConfig(readTestdata(t, "llama.json"), 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	src := strings.Replace(string(readTestdata(t, "llama_mlp.mlir")), "9.99999974E-6", "9.99999997E-7", 1)
	shlo, err := ImportStableHLO(src)
	if err != nil {
		t.Fatal(err)
	}
	if CanonicalFingerprint(shlo.Expr) == CanonicalFingerprint(hf.Expr) {
		t.Fatal("a different epsilon gave the same term")
	}
}

// reference computes the MLP sub-block in float64 from BF16 inputs,
// rounding to BF16 where the block casts.
func reference(h, gamma, wg, wu, wd []float64, m, d, inter int, eps float64) []float64 {
	bf := func(f float64) float64 { return float64(hopper.BF16.Round(float32(f))) }
	out := make([]float64, m*d)
	for r := range m {
		x := h[r*d : (r+1)*d]
		ss := 0.0
		for _, v := range x {
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss/float64(d)+eps)
		n := make([]float64, d)
		for c := range d {
			n[c] = bf(x[c] * inv * gamma[c])
		}
		p := make([]float64, inter)
		for i := range inter {
			var g, u float64
			for c := range d {
				g += n[c] * wg[i*d+c]
				u += n[c] * wu[i*d+c]
			}
			p[i] = bf(g / (1 + math.Exp(-g)) * u)
		}
		for c := range d {
			var s float64
			for i := range inter {
				s += p[i] * wd[c*inter+i]
			}
			out[r*d+c] = s + x[c]
		}
	}
	return out
}

func randValues(r *rand.Rand, t hopper.Tensor, scale float64) hopper.Value {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	v := make([]float64, n)
	for i := range v {
		v[i] = float64(t.DType.Round(float32(r.NormFloat64() * scale)))
	}
	return hopper.Value{Shape: slices.Clone(t.Shape), Data: v}
}

// TestImportedBlockNumerics evaluates the optimized term against an
// independent float64 reference, then runs the compiled plan's CUDA source
// under host emulation and compares it with the term.
func TestImportedBlockNumerics(t *testing.T) {
	const m, d, inter = 4, 256, 512
	res := importAll(t)["hf"]
	r := rand.New(rand.NewPCG(1, 2))
	env := make(map[string]hopper.Value)
	leaves := append(slices.Clone(res.Inputs), func() []hopper.Tensor {
		var ts []hopper.Tensor
		for _, w := range res.Weights {
			ts = append(ts, hopper.Tensor{Name: w.Leaf, DType: w.DType, Shape: w.Shape})
		}
		return ts
	}()...)
	for _, lt := range leaves {
		scale := 0.05
		if len(lt.Shape) == 1 {
			scale = 1
		}
		env[lt.Name] = randValues(r, lt, scale)
	}
	got, err := hopper.Eval(res.Expr, env)
	if err != nil {
		t.Fatal(err)
	}
	leaf := func(suffix string) []float64 {
		for _, w := range res.Weights {
			if strings.HasSuffix(w.Leaf, suffix) {
				return env[w.Leaf].Data
			}
		}
		t.Fatalf("no weight %s", suffix)
		return nil
	}
	want := reference(env["hidden_states"].Data, leaf("layernorm_weight"), leaf("gate_proj_weight"), leaf("up_proj_weight"), leaf("down_proj_weight"), m, d, inter, 1e-5)
	for i := range want {
		if diff := math.Abs(got.Data[i] - want[i]); diff > 2e-3*(1+math.Abs(want[i])) {
			t.Fatalf("element %d: term %g, reference %g", i, got.Data[i], want[i])
		}
	}

	// The emulator runs each CUDA thread as a host thread; this plan takes
	// about two minutes, so it is opt-in.
	if os.Getenv("CUDA_EGRAPH_FRONTEND_EMU") != "1" {
		t.Skip("set CUDA_EGRAPH_FRONTEND_EMU=1 to run the plan under host emulation")
	}
	src, err := res.Plan.Source()
	if err != nil {
		t.Fatal(err)
	}
	var bufs []cudahostemu.Buffer
	var args []string
	for _, in := range res.Plan.Inputs {
		bufs = append(bufs, cudahostemu.Buffer{Name: in.Name, CType: cType(in.DType), Data: encode(in.DType, env[in.Name].Data)})
		args = append(args, in.Name)
	}
	for _, tmp := range append(slices.Clone(res.Plan.Temps), res.Plan.Output) {
		bufs = append(bufs, cudahostemu.Buffer{Name: tmp.Name, CType: cType(tmp.DType), Data: cudahostemu.Poison(size(tmp.Shape) * tmp.DType.Bits() / 8)})
		args = append(args, tmp.Name)
	}
	out, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  src,
		Launch:  fmt.Sprintf("%s_run(%s, nullptr)", res.Plan.Name, strings.Join(args, ", ")),
		Buffers: bufs,
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; plan not run")
	}
	if err != nil {
		t.Fatalf("emulated plan: %v", err)
	}
	ob := out[res.Plan.Output.Name]
	for i := range got.Data {
		e := float64(math.Float32frombits(binary.LittleEndian.Uint32(ob[4*i:])))
		if diff := math.Abs(e - got.Data[i]); !(diff <= 2e-3*(1+math.Abs(got.Data[i]))) {
			t.Fatalf("element %d: emulated %g, term %g", i, e, got.Data[i])
		}
	}
}

func cType(d hopper.DType) string {
	switch d {
	case hopper.BF16:
		return "uint16_t"
	case hopper.F32:
		return "float"
	}
	return "uint8_t"
}

func encode(d hopper.DType, v []float64) []byte {
	var b []byte
	for _, f := range v {
		if d == hopper.BF16 {
			b = binary.LittleEndian.AppendUint16(b, uint16(math.Float32bits(float32(f))>>16))
		} else {
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(float32(f)))
		}
	}
	return b
}

func TestImportHFConfigFamilies(t *testing.T) {
	tests := []struct {
		file      string
		family    ArchFamily
		weights   []string
		lambda    string
		unlowered string
	}{
		{"llama.json", ArchLlama, []string{"model.layers.0.self_attn.k_proj.weight", "model.layers.0.mlp.gate_proj.weight"}, "(mul (mul $0 (sigmoid $0)) $1)", "softmax"},
		{"qwen3.json", ArchQwen3, []string{"model.layers.0.self_attn.q_norm.weight", "model.layers.0.mlp.down_proj.weight"}, "(mul (mul $0 (sigmoid $0)) $1)", "softmax"},
		{"deepseek_v3.json", ArchDeepseekV3, []string{"model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "model.layers.0.mlp.gate_proj.weight"}, "(mul (mul $0 (sigmoid $0)) $1)", "latent"},
		{"deepseek_v3_moe.json", ArchDeepseekV3, []string{"model.layers.0.mlp.experts.7.up_proj.weight", "model.layers.0.mlp.shared_experts.gate_proj.weight"}, "(mul (mul $0 (sigmoid $0)) $1)", "routed experts"},
		{"nemotron_h.json", ArchNemotronH, []string{"backbone.layers.1.mixer.up_proj.weight", "backbone.layers.1.norm.weight"}, "(mul (max $0 0) (max $0 0))", ""},
	}
	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			cfg, res, err := ImportHFConfig(readTestdata(t, tt.file), 2, 2)
			if err != nil {
				t.Fatal(err)
			}
			if f, _ := cfg.Family(); f != tt.family {
				t.Errorf("family %s, want %s", f, tt.family)
			}
			for _, w := range tt.weights {
				if _, ok := res.WeightMap[w]; !ok {
					t.Errorf("weight map lacks %s", w)
				}
			}
			if s := res.Expr.String(); !strings.Contains(s, tt.lambda) {
				t.Errorf("term lacks %s:\n%s", tt.lambda, s)
			}
			if tt.unlowered != "" && !strings.Contains(strings.Join(res.Unlowered, "\n"), tt.unlowered) {
				t.Errorf("unlowered %q lacks %q", res.Unlowered, tt.unlowered)
			}
			for _, w := range res.Weights {
				shape, ok := res.WeightMap[w.Name]
				if !ok || !slices.Equal(shape, w.Shape) {
					t.Errorf("lowered weight %s %v, map has %v", w.Name, w.Shape, shape)
				}
			}
		})
	}
	// The DeepSeek-V3 MLA projection shapes follow the config.
	_, res, err := ImportHFConfig(readTestdata(t, "deepseek_v3.json"), 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := res.WeightMap["model.layers.0.self_attn.kv_b_proj.weight"]; !slices.Equal(got, []int{4 * (32 + 32), 64}) {
		t.Errorf("kv_b_proj %v", got)
	}
}

func TestParseHFConfigRejects(t *testing.T) {
	tests := []struct {
		json, want string
	}{
		{`{`, "parse hf config"},
		{`{"architectures": ["GPT2LMHeadModel"]}`, "unsupported architecture"},
		{`{"architectures": ["LlamaForCausalLM"], "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 3, "intermediate_size": 8, "rms_norm_eps": 1e-5}`, "do not divide"},
		{`{"architectures": ["LlamaForCausalLM"], "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4, "intermediate_size": 8}`, "rms_norm_eps"},
		{`{"architectures": ["NemotronHForCausalLM"], "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "intermediate_size": 8, "rms_norm_eps": 1e-5, "hybrid_override_pattern": "M*"}`, "no MLP layer"},
		{`{"architectures": ["DeepseekV3ForCausalLM"], "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4, "rms_norm_eps": 1e-5}`, "kv_lora_rank"},
	}
	for _, tt := range tests {
		if _, err := ParseHFConfig([]byte(tt.json)); err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("ParseHFConfig(%s) = %v, want %q", tt.json, err, tt.want)
		}
	}
}

func TestImportStableHLORejects(t *testing.T) {
	good := string(readTestdata(t, "llama_mlp.mlir"))
	tests := []struct {
		name, old, new, want string
	}{
		{"rank mismatch", "%1 = stablehlo.multiply %0, %0 : tensor<4x256xf32>", "%1 = stablehlo.multiply %0, %0 : tensor<4x256x1xf32>", "line 7: %1: multiply operand tensor<4x256xf32> does not match result tensor<4x256x1xf32>"},
		{"declared operand", "(tensor<4x256xbf16>, tensor<256x512xbf16>) -> tensor<4x512xf32>", "(tensor<4x256xbf16>, tensor<256x512x1xbf16>) -> tensor<4x512xf32>", "line 23: %14: operand 1 is tensor<256x512xbf16>, declared tensor<256x512x1xbf16>"},
		{"undefined", "%16 = stablehlo.multiply %14, %15", "%16 = stablehlo.multiply %14, %99", "line 25: %16: undefined value %99"},
		{"unsupported", "stablehlo.logistic", "stablehlo.tanh", "unsupported operation stablehlo.tanh"},
		{"reduce dim", "across dimensions = [1]", "across dimensions = [0]", "line 9: %2: reduce across [0]: only the last dimension"},
		{"broadcast", "dims = [1] : (tensor<256xf32>)", "dims = [0] : (tensor<256xf32>)", "line 20: %11: broadcast_in_dim dimension 0 of size 256 cannot become 4"},
		{"contracting", "contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x512xbf16>", "contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x512xbf16>", "line 29: %20: contracting_dims"},
		{"contracting out of range", "contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x256xbf16>, tensor<256x512xbf16>) -> tensor<4x512xf32>\n    %15", "contracting_dims = [1] x [6], precision = [DEFAULT, DEFAULT] : (tensor<4x256xbf16>, tensor<256x512xbf16>) -> tensor<4x512xf32>\n    %15", "line 23: %14: contracting_dims"},
		{"no return", "    return %21 : tensor<4x256xf32>\n", "", "no return"},
		{"redefined", "%15 = stablehlo.logistic", "%14 = stablehlo.logistic", "line 24: %14 redefined"},
		{"non-splat", "dense<0.000000e+00>", "dense<[0.0, 1.0]>", "only splat"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := strings.Replace(good, tt.old, tt.new, 1)
			if src == good {
				t.Fatalf("fixture lacks %q", tt.old)
			}
			_, err := ImportStableHLO(src)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestImportONNXWireRejects(t *testing.T) {
	good := onnxFixture(t)
	mutate := func(f func(b []byte) []byte) []byte { return f(slices.Clone(good)) }
	tests := []struct {
		name string
		data []byte
		want string
	}{
		{"empty", nil, "model has no graph"},
		{"invalid tag", []byte{0x07}, "offset 0: invalid tag 0x7 (field 0, wire type 7)"},
		{"group wire type", []byte{0x0b}, "offset 0: invalid tag 0xb (field 1, wire type 3)"},
		{"truncated", good[:len(good)/2], "overruns"},
		{"bad varint", []byte{0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01}, "offset 1: bad varint"},
		{"unsupported op", mutate(func(b []byte) []byte {
			i := strings.Index(string(b), "Sigmoid")
			copy(b[i:], "Softmax")
			return b
		}), "node \"sigmoid\" (Softmax): unsupported operator"},
		{"wire type in node", mutate(func(b []byte) []byte {
			// The first node's first field (input, tag 0x0a) becomes
			// field 1 with wire type 5.
			i := strings.Index(string(b), "\x0a\x0dhidden_states\x12")
			b[i] = 0x0d
			return b
		}), "string field with wire type 5"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ImportONNXWire(tt.data)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
		})
	}
}

// TestWireReaderAgainstProtoc reads fields protoc encoded: a negative
// int64 attribute (ten-byte varint), a float attribute and packed dims.
func TestWireReaderAgainstProtoc(t *testing.T) {
	g, err := readModel(onnxFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(g.nodes) != 12 || len(g.initializers) != 4 || len(g.inputs) != 1 || len(g.outputs) != 1 {
		t.Fatalf("graph has %d nodes, %d initializers, %d inputs, %d outputs", len(g.nodes), len(g.initializers), len(g.inputs), len(g.outputs))
	}
	n := g.nodes[2]
	if n.opType != "SimplifiedLayerNormalization" || n.domain != "com.microsoft" || n.attrs["axis"].i != -1 || n.attrs["epsilon"].f != 1e-5 {
		t.Errorf("rmsnorm node %+v", n)
	}
	if w := g.initializers[1]; w.name != "model.layers.0.mlp.gate_proj.weight" || !slices.Equal(w.dims, []int64{256, 512}) || w.dataType != onnxBFloat16 {
		t.Errorf("gate initializer %+v", w)
	}
	if in := g.inputs[0]; in.name != "hidden_states" || !slices.Equal(in.dims, []int64{4, 256}) || in.elemType != onnxBFloat16 {
		t.Errorf("input %+v", in)
	}
}

// TestNoProtobufDependency checks the package's own imports are the
// standard library and this module, and nothing pulls in a protobuf
// library.
func TestNoProtobufDependency(t *testing.T) {
	out, err := exec.Command("go", "list", "-f", "{{join .Imports \"\\n\"}}", ".").Output()
	if err != nil {
		t.Skipf("go list: %v", err)
	}
	for _, p := range strings.Fields(string(out)) {
		if strings.Contains(p, ".") && !strings.HasPrefix(p, "github.com/tmc/lib/nvidia/") {
			t.Errorf("imports %s", p)
		}
	}
	deps, err := exec.Command("go", "list", "-deps", ".").Output()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(deps), "protobuf") {
		t.Error("dependencies include a protobuf library")
	}
}

func FuzzImportONNXWire(f *testing.F) {
	f.Add(onnxFixture(f))
	f.Add([]byte{0x3a, 0x02, 0x0a, 0x00})
	f.Fuzz(func(t *testing.T, b []byte) {
		ImportONNXWire(b)
	})
}

func FuzzImportStableHLO(f *testing.F) {
	f.Add(string(readTestdata(f, "llama_mlp.mlir")))
	f.Add("func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {\n%0 = stablehlo.exponential %arg0 : tensor<4xf32>\nreturn %0 : tensor<4xf32>\n}")
	f.Fuzz(func(t *testing.T, s string) {
		ImportStableHLO(s)
	})
}

func TestFuseMapsNormalizes(t *testing.T) {
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.F32, 2, 8)
	inner := b.Map(hopper.MustLambda("(sigmoid $0)"), x)
	outer := b.Map(hopper.MustLambda("(mul $0 $1)"), x, inner)
	if _, err := FuseMaps(b.G); err != nil {
		t.Fatal(err)
	}
	e, _, err := egraph.ExtractBest(b.G, outer, hopper.Cost{})
	if err != nil {
		t.Fatal(err)
	}
	if e.Op != hopper.OpMap || e.Children[0].Op != "fn:(mul $0 (sigmoid $0))" || len(e.Children) != 2 {
		t.Fatalf("FuseMaps gave %s", e)
	}
}

func ExampleImportStableHLO() {
	res, err := ImportStableHLO(`func.func @main(%arg0: tensor<2x64xf32>) -> tensor<2x64xf32> {
  %0 = stablehlo.logistic %arg0 : tensor<2x64xf32>
  %1 = stablehlo.multiply %arg0, %0 : tensor<2x64xf32>
  return %1 : tensor<2x64xf32>
}`)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Expr.Op, res.Expr.Children[0].Op, len(res.Plan.Kernels))
	// Output: map fn:(mul $0 (sigmoid $0)) 1
}
