package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

const (
	moe8Theta    = 500000.0
	moe8QKFactor = 5.4730064863875
)

// moe8QKV is the attention boundary before FA3 for one batch of tokens
// (moe8_attention_decode/reference.py:301-313, FP8 query path): per-head
// RMSNorm of q and k, interleaved RoPE, q rounded to E4M3 before the
// qk_scale_factor, k and v divided by their scales and stored as E4M3 at
// cache_loc[t] in a [rows, 2, 64] cache.
func moe8QKV() []RowKernel {
	norm := []RowValue{
		{Name: "ss", Reduce: "sum", Body: MustLambda("(mul $0 $0)"), Args: []string{"x"}},
		{Name: "n", Body: MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), Args: []string{"x", "ss"}},
		{Name: "ang", Body: MustLambda("(mul $0 $1)"), Args: []string{"pos", "inv_freq"}},
		{Name: "rot", Body: MustLambda("(add (mul $0 (cos $1)) (mul (mul $2 $3) (sin $1)))"), Args: []string{"n", "ang", "sgn", "~n"}},
	}
	x := func(offset, heads int) RowTensor {
		return RowTensor{Name: "x", DType: BF16, Kind: RowData, Offset: offset, Group: heads, GroupStride: 2304, RowStride: 64}
	}
	pos := func(heads int) RowTensor { return RowTensor{Name: "pos", DType: I64, Kind: RowScalar, Group: heads} }
	freq := RowTensor{Name: "inv_freq", DType: F32, Kind: ColVector}
	cache := func(name, value string) RowTensor {
		return RowTensor{Name: name, DType: E4M3, Kind: RowData, Value: value, Group: 2, GroupStride: 128, RowStride: 64, Index: "loc", IndexDType: I64}
	}
	q := RowKernel{Name: "q_norm_rope", D: 64,
		Inputs:  []RowTensor{x(0, 16), pos(16), freq},
		Values:  norm,
		Outputs: []RowTensor{{Name: "q8", DType: E4M3, Kind: RowData, Value: "rot", Group: 16, RowStride: 64}},
	}
	k := RowKernel{Name: "k_norm_rope_store", D: 64,
		Inputs: []RowTensor{x(1024, 2), pos(2), freq, {Name: "k_scale", DType: F32, Kind: Scalar}},
		Values: append(append([]RowValue(nil), norm...),
			RowValue{Name: "ks", Body: MustLambda("(div $0 $1)"), Args: []string{"rot", "k_scale"}}),
		Outputs: []RowTensor{cache("k_cache", "ks")},
	}
	v := RowKernel{Name: "v_store", D: 64,
		Inputs:  []RowTensor{x(1152, 2), {Name: "v_scale", DType: F32, Kind: Scalar}},
		Values:  []RowValue{{Name: "vs", Body: MustLambda("(div $0 $1)"), Args: []string{"x", "v_scale"}}},
		Outputs: []RowTensor{cache("v_cache", "vs")},
	}
	return []RowKernel{q, k, v}
}

// moe8Residual is FusedPostNormResidualPreNorm (moe8_decode_layer/reference.py:
// 339-344 and 453-454): stream1 = alpha·stream + beta·rms(a, 1e-8),
// h = rms(stream1, 1e-5)·(1 + pre_ffn_norm).
func moe8Residual() RowKernel {
	return RowKernel{Name: "residual_prenorm", D: 8192,
		Inputs: []RowTensor{
			{Name: "resid", DType: F32, Kind: RowData},
			{Name: "a", DType: BF16, Kind: RowData},
			{Name: "gate", DType: BF16, Kind: ColVector},
			{Name: "pre", DType: BF16, Kind: ColVector},
		},
		Values: []RowValue{
			{Name: "asq", Reduce: "sum", Body: MustLambda("(mul $0 $0)"), Args: []string{"a"}},
			{Name: "s1", Body: MustLambda(`(add
				(mul (sqrt (max (mul (sigmoid (neg (div $2 0.3))) (add 1 (sigmoid (div $2 0.3)))) 0.001)) $0)
				(mul (sigmoid (div $2 0.3)) (mul $1 (rsqrt (add (div $3 8192) 1e-8)))))`),
				Args: []string{"resid", "a", "gate", "asq"}},
			{Name: "s1sq", Reduce: "sum", Body: MustLambda("(mul $0 $0)"), Args: []string{"s1"}},
			{Name: "h", Body: MustLambda("(mul (mul $0 (rsqrt (add (div $1 8192) 1e-5))) (add 1 $2))"), Args: []string{"s1", "s1sq", "pre"}},
		},
		Outputs: []RowTensor{
			{Name: "stream1", DType: F32, Kind: RowData, Value: "s1"},
			{Name: "hidden", DType: BF16, Kind: RowData, Value: "h"},
		},
	}
}

// moe8Gate is rms_norm_gate (moe8_attention_decode/reference.py:336):
// rms(o per head, 1e-5) · sigmoid(gate), gate at column 1280 of the fused row.
func moe8Gate() RowKernel {
	return RowKernel{Name: "rms_norm_gate", D: 64,
		Inputs: []RowTensor{
			{Name: "o", DType: F32, Kind: RowData},
			{Name: "qkv", DType: BF16, Kind: RowData, Offset: 1280, Group: 16, GroupStride: 2304, RowStride: 64},
		},
		Values: []RowValue{
			{Name: "ss", Reduce: "sum", Body: MustLambda("(mul $0 $0)"), Args: []string{"o"}},
			{Name: "y", Body: MustLambda("(mul (mul $0 (rsqrt (add (div $1 64) 1e-5))) (sigmoid $2))"), Args: []string{"o", "ss", "qkv"}},
		},
		Outputs: []RowTensor{{Name: "out", DType: BF16, Kind: RowData, Value: "y"}},
	}
}

func invFreq() []float32 {
	f := make([]float32, 64)
	for i := 0; i < 32; i++ {
		v := float32(1 / math.Pow(moe8Theta, float64(2*i)/64))
		f[2*i], f[2*i+1] = v, v
	}
	return f
}

// typed holds one test buffer in its storage type and as values.
type typed struct {
	dtype DType
	vals  []float64
}

func (b typed) bytes() []byte {
	switch b.dtype {
	case BF16:
		v := make([]uint16, len(b.vals))
		for i, f := range b.vals {
			v[i] = BF16Bits(float32(f))
		}
		return u16Bytes(v)
	case E4M3:
		v := make([]byte, len(b.vals))
		for i, f := range b.vals {
			v[i] = E4M3Bits(float32(f))
		}
		return v
	case I64:
		v := make([]int64, len(b.vals))
		for i, f := range b.vals {
			v[i] = int64(f)
		}
		return i64Bytes(v)
	case I32:
		v := make([]int32, len(b.vals))
		for i, f := range b.vals {
			v[i] = int32(f)
		}
		return i32Bytes(v)
	}
	v := make([]float32, len(b.vals))
	for i, f := range b.vals {
		v[i] = float32(f)
	}
	return f32Bytes(v)
}

func decode(d DType, raw []byte) []float64 {
	var out []float64
	switch d {
	case BF16:
		for _, b := range bytesU16(raw) {
			out = append(out, float64(FromBF16Bits(b)))
		}
	case E4M3:
		for _, b := range raw {
			out = append(out, float64(FromE4M3Bits(b)))
		}
	default:
		for _, f := range bytesF32(raw) {
			out = append(out, float64(f))
		}
	}
	return out
}

// runRowKernel runs k emulated and by Reference on the same buffers and
// returns both versions of every output.
func runRowKernel(t *testing.T, k RowKernel, rows int, bufs map[string]typed) (got, want map[string][]float64) {
	t.Helper()
	src, err := k.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	ref := make(map[string][]float64)
	for name, b := range bufs {
		v := make([]float64, len(b.vals))
		for i, f := range b.vals {
			v[i] = float64(b.dtype.Round(float32(f)))
		}
		ref[name] = v
	}
	if err := k.Reference(rows, ref); err != nil {
		t.Fatal(err)
	}
	var eb []emuBuf
	var names []string
	for _, p := range k.params()[1:] {
		f := strings.Fields(p)
		name := f[len(f)-1]
		names = append(names, name)
		b := bufs[name]
		eb = append(eb, emuBuf{name, b.dtype.cType(), b.bytes()})
	}
	grid, block := k.Grid(rows)
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)", grid[0], block[0], k.Name, rows, strings.Join(names, ", "))
	res := runEmu(t, src, launch, eb...)
	got, want = make(map[string][]float64), make(map[string][]float64)
	for _, o := range k.Outputs {
		got[o.Name] = decode(o.DType, res[o.Name])
		want[o.Name] = ref[o.Name]
	}
	return got, want
}

func randVals(r *rand.Rand, n int, scale float64) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = r.NormFloat64() * scale
	}
	return v
}

// agree checks got against want: BF16 and E4M3 outputs may differ by one
// rounding step where FP32 evaluation orders differ.
func agree(t *testing.T, name string, d DType, got, want []float64) {
	t.Helper()
	rel := map[DType]float64{F32: 1e-4, BF16: 1.0 / 64, E4M3: 1.0 / 4}[d]
	bad := 0
	for i := range want {
		if diff := math.Abs(got[i] - want[i]); diff > rel*math.Abs(want[i])+1e-6 || math.IsNaN(got[i]) {
			if bad < 3 {
				t.Errorf("%s[%d] = %g, want %g", name, i, got[i], want[i])
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%s: %d of %d values differ", name, bad, len(want))
	}
}

func TestRowKernelMoe8QKV(t *testing.T) {
	const tokens = 3
	r := rand.New(rand.NewPCG(3, 4))
	freq := invFreq()
	f64 := func(v []float32) []float64 {
		out := make([]float64, len(v))
		for i, f := range v {
			out[i] = float64(f)
		}
		return out
	}
	qkv := randVals(r, tokens*2304, 2)
	pos := []float64{5, 130000, 2047}
	loc := []float64{7, -1, 2} // token 1 has no cache slot
	bufs := map[string]typed{
		"x": {BF16, qkv}, "pos": {I64, pos}, "inv_freq": {F32, f64(freq)},
		"q8":      {E4M3, make([]float64, tokens*16*64)},
		"k_cache": {E4M3, make([]float64, 8*128)}, "v_cache": {E4M3, make([]float64, 8*128)},
		"loc": {I64, loc}, "k_scale": {F32, []float64{0.5}}, "v_scale": {F32, []float64{1.0 / 64}},
	}
	for _, k := range moe8QKV() {
		rows := tokens * 16
		if k.Name != "q_norm_rope" {
			rows = tokens * 2
		}
		got, want := runRowKernel(t, k, rows, bufs)
		for _, o := range k.Outputs {
			agree(t, k.Name+"."+o.Name, o.DType, got[o.Name], want[o.Name])
		}
	}
	// The spec matches reference.py, computed directly: q head 3 of token 1.
	got, _ := runRowKernel(t, moe8QKV()[0], tokens*16, bufs)
	row := make([]float64, 64)
	var ss float64
	for c := range row {
		row[c] = float64(BF16.Round(float32(qkv[2304+3*64+c])))
		ss += row[c] * row[c]
	}
	inv := 1 / math.Sqrt(ss/64+1e-5)
	for i := 0; i < 32; i++ {
		ang := float64(float32(pos[1]) * freq[2*i])
		e, o := row[2*i]*inv, row[2*i+1]*inv
		want := []float64{e*math.Cos(ang) - o*math.Sin(ang), e*math.Sin(ang) + o*math.Cos(ang)}
		for j, w := range want {
			g := got["q8"][(1*16+3)*64+2*i+j]
			if math.Abs(g-w) > math.Abs(w)/8+1.0/512 {
				t.Errorf("q8[t1,h3,%d] = %g, direct reference %g", 2*i+j, g, w)
			}
		}
	}
}

func TestRowKernelMoe8Residual(t *testing.T) {
	const tokens = 2
	r := rand.New(rand.NewPCG(5, 6))
	k := moe8Residual()
	bufs := map[string]typed{
		"resid": {F32, randVals(r, tokens*8192, 1)}, "a": {BF16, randVals(r, tokens*8192, 3)},
		"gate": {BF16, randVals(r, 8192, 0.3)}, "pre": {BF16, randVals(r, 8192, 0.1)},
		"stream1": {F32, make([]float64, tokens*8192)}, "hidden": {BF16, make([]float64, tokens*8192)},
	}
	got, want := runRowKernel(t, k, tokens, bufs)
	for _, o := range k.Outputs {
		agree(t, o.Name, o.DType, got[o.Name], want[o.Name])
	}
	// Direct formula for one element.
	const tok, c = 1, 4321
	at := func(name string, i int) float64 { return float64(bufs[name].dtype.Round(float32(bufs[name].vals[i]))) }
	var asq float64
	for i := 0; i < 8192; i++ {
		asq += at("a", tok*8192+i) * at("a", tok*8192+i)
	}
	w := at("gate", c) / 0.3
	beta := 1 / (1 + math.Exp(-w))
	alpha := math.Sqrt(math.Max(1/(1+math.Exp(w))*(1+beta), 1e-3))
	s1 := alpha*at("resid", tok*8192+c) + beta*at("a", tok*8192+c)/math.Sqrt(asq/8192+1e-8)
	if g := got["stream1"][tok*8192+c]; math.Abs(g-s1) > 1e-4*math.Abs(s1) {
		t.Errorf("stream1[%d,%d] = %g, direct reference %g", tok, c, g, s1)
	}
}

func TestRowKernelMoe8Gate(t *testing.T) {
	const tokens = 2
	r := rand.New(rand.NewPCG(7, 8))
	k := moe8Gate()
	bufs := map[string]typed{
		"o": {F32, randVals(r, tokens*1024, 0.7)}, "qkv": {BF16, randVals(r, tokens*2304, 1)},
		"out": {BF16, make([]float64, tokens*1024)},
	}
	got, want := runRowKernel(t, k, tokens*16, bufs)
	agree(t, "out", BF16, got["out"], want["out"])
}

func TestRowKernelReduceMax(t *testing.T) {
	k := RowKernel{Name: "softmax_row", D: 128,
		Inputs: []RowTensor{{Name: "x", DType: F32, Kind: RowData}},
		Values: []RowValue{
			{Name: "m", Reduce: "max", Body: Identity, Args: []string{"x"}},
			{Name: "e", Body: MustLambda("(exp (sub $0 $1))"), Args: []string{"x", "m"}},
			{Name: "z", Reduce: "sum", Body: Identity, Args: []string{"e"}},
			{Name: "y", Body: MustLambda("(div $0 $1)"), Args: []string{"e", "z"}},
		},
		Outputs: []RowTensor{{Name: "y_out", DType: F32, Kind: RowData, Value: "y"}},
	}
	r := rand.New(rand.NewPCG(9, 10))
	bufs := map[string]typed{"x": {F32, randVals(r, 9*128, 3)}, "y_out": {F32, make([]float64, 9*128)}}
	got, want := runRowKernel(t, k, 9, bufs)
	agree(t, "y", F32, got["y_out"], want["y_out"])
}

func TestRowKernelErrors(t *testing.T) {
	base := moe8Gate()
	for name, mutate := range map[string]func(*RowKernel){
		"width":      func(k *RowKernel) { k.D = 96 },
		"unknown":    func(k *RowKernel) { k.Values[1].Args[2] = "nope" },
		"scalar_out": func(k *RowKernel) { k.Outputs[0].Value = "ss" },
		"reserved":   func(k *RowKernel) { k.Inputs[0].Name = "lane" },
		"duplicate":  func(k *RowKernel) { k.Values[1].Name = "ss" },
		"reduce":     func(k *RowKernel) { k.Values[0].Reduce = "mean" },
		"misaligned": func(k *RowKernel) { k.Inputs[1].Offset = 1281 },
		"arity":      func(k *RowKernel) { k.Values[1].Args = k.Values[1].Args[:2] },
	} {
		k := base
		k.Inputs = append([]RowTensor(nil), base.Inputs...)
		k.Values = append([]RowValue(nil), base.Values...)
		for i := range k.Values {
			k.Values[i].Args = append([]string(nil), base.Values[i].Args...)
		}
		k.Outputs = append([]RowTensor(nil), base.Outputs...)
		mutate(&k)
		if _, err := k.CUDA(); err == nil {
			t.Errorf("%s: CUDA succeeded", name)
		}
	}
}

func TestRowKernelGolden(t *testing.T) {
	for _, k := range append(moe8QKV(), moe8Residual(), moe8Gate()) {
		src, err := k.Source()
		if err != nil {
			t.Fatal(err)
		}
		checkGolden(t, k.Name+".cu", src)
	}
}

func ExampleRowKernel() {
	// Post-expert norm (moe8_moe_w4a16/reference.py:88-90): scale by 1 + w,
	// then RMSNorm with eps 1e-8, over rows of 4096.
	k := RowKernel{Name: "post_expert_norm", D: 4096,
		Inputs: []RowTensor{{Name: "y", DType: F32, Kind: RowData}, {Name: "w", DType: BF16, Kind: ColVector}},
		Values: []RowValue{
			{Name: "s", Body: MustLambda("(mul $0 (add 1 $1))"), Args: []string{"y", "w"}},
			{Name: "ss", Reduce: "sum", Body: MustLambda("(mul $0 $0)"), Args: []string{"s"}},
			{Name: "n", Body: MustLambda("(mul $0 (rsqrt (add (div $1 4096) 1e-8)))"), Args: []string{"s", "ss"}},
		},
		Outputs: []RowTensor{{Name: "out", DType: F32, Kind: RowData, Value: "n"}},
	}
	grid, block := k.Grid(8)
	_, err := k.Source()
	fmt.Println(grid, block, err)
	// Output:
	// [8 1 1] [256 1 1] <nil>
}
