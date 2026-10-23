package hopper

import (
	"fmt"
	"math"
	"strings"
)

// A RowKernel computes elementwise maps and reductions over rows of D values:
// per-head RMSNorm (D = 64), whole-row norms (D = 8192), gates, RoPE, casts,
// and stores to strided or scattered rows. One warp owns a row when D ≤ 1024
// and one block of 256 threads otherwise; each thread holds its row elements
// in registers, so every value is computed once and every tensor is read and
// written once.
//
// Rows are numbered r = 0..rows-1 at run time. A row tensor addresses row r,
// column c at
//
//	Offset + g·GroupStride + (r mod Group)·RowStride + c,  g = r / Group
//
// (Group 0 means 1), so the q heads of a fused [T, 2304] projection are
// {Group: 16, GroupStride: 2304, RowStride: 64}, and an output with an Index
// tensor writes group g at Index[g] instead (skipping negative indices).
type RowKernel struct {
	Name    string
	D       int
	Inputs  []RowTensor
	Values  []RowValue
	Outputs []RowTensor
}

// A RowKind says how a tensor is indexed by row r and column c.
type RowKind uint8

const (
	RowData   RowKind = iota + 1 // a row of D values per r (see RowKernel)
	ColVector                    // [D], shared by all rows (norm weights, inverse frequencies)
	RowScalar                    // one value per row group: T[r / Group] (positions, gates)
	Scalar                       // T[0] (per-tensor scales)
)

// A RowTensor is a kernel input or output.
type RowTensor struct {
	Name  string
	DType DType // F32, BF16 or E4M3 data; I32/I64 allowed for RowScalar inputs
	Kind  RowKind
	Value string // outputs: the value stored

	Offset, Group, GroupStride, RowStride int    // RowData addressing; zero strides mean D
	Index                                 string // outputs: optional I64 or I32 row-group index tensor
	IndexDType                            DType
}

// A RowValue defines a named value. A map is a vector over the row's D
// columns; a reduction ("sum" or "max" of a map body over the row) is a row
// scalar that later values may use.
//
// Args name inputs or earlier values, or one of:
//
//	col   the column index c
//	sgn   -1 for even c, +1 for odd c (RoPE)
//	~v    vector v at column c^1, the other element of c's pair (RoPE)
type RowValue struct {
	Name   string
	Reduce string // "", "sum" or "max"
	Body   Lambda
	Args   []string
}

// rowReserved are names the emitted kernel uses for itself.
var rowReserved = map[string]bool{"col": true, "sgn": true, "rows": true, "r": true, "lane": true,
	"p": true, "g": true, "j": true, "PER": true, "VEC": true, "THREADS": true, "stream": true}

type rowSym struct {
	vector bool
	tensor *RowTensor
	value  *RowValue
}

func (k RowKernel) check() (map[string]rowSym, error) {
	if !cIdent(k.Name) {
		return nil, fmt.Errorf("row kernel: bad name %q", k.Name)
	}
	if k.D < 64 || k.D > 1024 && k.D%(256*2) != 0 || k.D <= 1024 && k.D%64 != 0 {
		return nil, fmt.Errorf("row kernel %s: unsupported width %d", k.Name, k.D)
	}
	syms := make(map[string]rowSym)
	define := func(name string, s rowSym) error {
		if !cIdent(name) || rowReserved[name] {
			return fmt.Errorf("row kernel %s: bad name %q", k.Name, name)
		}
		if _, dup := syms[name]; dup {
			return fmt.Errorf("row kernel %s: %q defined twice", k.Name, name)
		}
		syms[name] = s
		return nil
	}
	for i := range k.Inputs {
		in := &k.Inputs[i]
		switch in.DType {
		case F32, BF16, E4M3:
		case I32, I64:
			if in.Kind != RowScalar && in.Kind != Scalar {
				return nil, fmt.Errorf("row kernel %s: integer input %s must be a scalar", k.Name, in.Name)
			}
		default:
			return nil, fmt.Errorf("row kernel %s: input %s has dtype %v", k.Name, in.Name, in.DType)
		}
		if in.Kind < RowData || in.Kind > Scalar {
			return nil, fmt.Errorf("row kernel %s: input %s has no kind", k.Name, in.Name)
		}
		if err := define(in.Name, rowSym{vector: in.Kind == RowData || in.Kind == ColVector, tensor: in}); err != nil {
			return nil, err
		}
	}
	for i := range k.Values {
		v := &k.Values[i]
		if v.Body.root == nil {
			return nil, fmt.Errorf("row kernel %s: value %s has no body", k.Name, v.Name)
		}
		if v.Body.Arity() > len(v.Args) {
			return nil, fmt.Errorf("row kernel %s: value %s: %s uses %d args, have %d", k.Name, v.Name, v.Body, v.Body.Arity(), len(v.Args))
		}
		vector := false
		for _, a := range v.Args {
			switch {
			case a == "col" || a == "sgn":
				vector = true
			case strings.HasPrefix(a, "~"):
				s, ok := syms[a[1:]]
				if !ok || !s.vector {
					return nil, fmt.Errorf("row kernel %s: value %s: %q is not a vector", k.Name, v.Name, a)
				}
				vector = true
			default:
				s, ok := syms[a]
				if !ok {
					return nil, fmt.Errorf("row kernel %s: value %s: unknown arg %q", k.Name, v.Name, a)
				}
				vector = vector || s.vector
			}
		}
		switch v.Reduce {
		case "":
		case "sum", "max":
			vector = false
		default:
			return nil, fmt.Errorf("row kernel %s: value %s: unknown reduction %q", k.Name, v.Name, v.Reduce)
		}
		if err := define(v.Name, rowSym{vector: vector, value: v}); err != nil {
			return nil, err
		}
	}
	if len(k.Outputs) == 0 {
		return nil, fmt.Errorf("row kernel %s: no outputs", k.Name)
	}
	_, _, vec := k.geometry()
	for _, t := range append(append([]RowTensor(nil), k.Inputs...), k.Outputs...) {
		gs, rs := t.strides(k.D)
		if t.Kind == RowData && (t.Offset%vec != 0 || gs%vec != 0 || rs%vec != 0) {
			return nil, fmt.Errorf("row kernel %s: %s offset and strides must be multiples of %d", k.Name, t.Name, vec)
		}
	}
	for i := range k.Outputs {
		o := &k.Outputs[i]
		if o.Kind != RowData || o.DType != F32 && o.DType != BF16 && o.DType != E4M3 {
			return nil, fmt.Errorf("row kernel %s: output %s must be F32, BF16 or E4M3 row data", k.Name, o.Name)
		}
		s, ok := syms[o.Value]
		if !ok {
			return nil, fmt.Errorf("row kernel %s: output %s stores unknown value %q", k.Name, o.Name, o.Value)
		}
		if !s.vector {
			return nil, fmt.Errorf("row kernel %s: output %s stores scalar %q", k.Name, o.Name, o.Value)
		}
		if o.Index != "" && o.IndexDType != I32 && o.IndexDType != I64 {
			return nil, fmt.Errorf("row kernel %s: output %s index must be i32 or i64", k.Name, o.Name)
		}
		if err := define(o.Name, rowSym{tensor: o}); err != nil {
			return nil, err
		}
		if o.Index != "" {
			if err := define(o.Index, rowSym{tensor: &RowTensor{Name: o.Index, DType: o.IndexDType, Kind: RowScalar}}); err != nil {
				return nil, err
			}
		}
	}
	return syms, nil
}

func (t RowTensor) group() int {
	if t.Group <= 0 {
		return 1
	}
	return t.Group
}

func (t RowTensor) strides(d int) (gs, rs int) {
	gs, rs = t.GroupStride, t.RowStride
	if rs == 0 {
		rs = d
	}
	if gs == 0 {
		gs = rs * t.group()
	}
	return gs, rs
}

// geometry: threads per row, values per thread, contiguous chunk per thread.
func (k RowKernel) geometry() (threads, per, vec int) {
	threads = 32
	if k.D > 1024 {
		threads = 256
	}
	per = k.D / threads
	vec = per
	if vec > 8 {
		vec = 8
	}
	for per%vec != 0 {
		vec /= 2
	}
	return threads, per, vec
}

// Grid returns the launch grid and block for rows rows.
func (k RowKernel) Grid(rows int) (grid, block [3]int) {
	threads, _, _ := k.geometry()
	if threads == 32 {
		return [3]int{(rows + 7) / 8, 1, 1}, [3]int{256, 1, 1}
	}
	return [3]int{rows, 1, 1}, [3]int{256, 1, 1}
}

func (k RowKernel) params() []string {
	p := []string{"int rows"}
	for _, in := range k.Inputs {
		p = append(p, "const "+in.DType.cType()+"* __restrict__ "+in.Name)
	}
	for _, o := range k.Outputs {
		p = append(p, o.DType.cType()+"* __restrict__ "+o.Name)
		if o.Index != "" {
			p = append(p, "const "+o.IndexDType.cType()+"* __restrict__ "+o.Index)
		}
	}
	return p
}

// CUDA returns the kernel source (without [Prelude]):
// __global__ void Name(int rows, inputs..., outputs and their indices...).
func (k RowKernel) CUDA() (string, error) {
	body, err := k.body()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: row kernel over rows of %d values.\n", k.Name, k.D)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n", k.Name, strings.Join(k.params(), ",\n    "))
	b.WriteString(body)
	b.WriteString("}\n")
	return b.String(), nil
}

// body emits the statements of one row-kernel instance, assuming rows and
// the tensor parameters are in scope; it computes the row from blockIdx.x.
func (k RowKernel) body() (string, error) {
	syms, err := k.check()
	if err != nil {
		return "", err
	}
	threads, per, vec := k.geometry()
	var b strings.Builder
	if threads == 32 {
		b.WriteString("    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;\n    if (r >= rows) return;\n")
	} else {
		b.WriteString("    const int r = blockIdx.x, lane = threadIdx.x;\n    __shared__ float eg_red[8];\n")
	}
	fmt.Fprintf(&b, "    constexpr int PER = %d, VEC = %d, THREADS = %d;\n", per, vec, threads)
	b.WriteString("    // Element j of this thread is column c(j).\n")
	b.WriteString("#define EG_COL(j) (((j) / VEC) * (THREADS * VEC) + lane * VEC + (j) % VEC)\n")
	// Row tensor base offsets.
	addr := func(t RowTensor, group string) string {
		gs, rs := t.strides(k.D)
		return fmt.Sprintf("(size_t)%d + (size_t)(%s) * %d + (size_t)(r %% %d) * %d", t.Offset, group, gs, t.group(), rs)
	}
	for _, in := range k.Inputs {
		switch in.Kind {
		case RowData:
			fmt.Fprintf(&b, "    float %s_v[PER];\n    {\n        const %s* p = %s + %s;\n", in.Name, in.DType.cType(), in.Name, addr(in, fmt.Sprintf("r / %d", in.group())))
			b.WriteString("#pragma unroll\n        for (int j = 0; j < PER; j += VEC) eg_load<VEC>(p + EG_COL(j), &" + in.Name + "_v[j]);\n    }\n")
		case ColVector:
			fmt.Fprintf(&b, "    float %s_v[PER];\n", in.Name)
			b.WriteString("#pragma unroll\n    for (int j = 0; j < PER; j += VEC) eg_load<VEC>(" + in.Name + " + EG_COL(j), &" + in.Name + "_v[j]);\n")
		case RowScalar:
			fmt.Fprintf(&b, "    const float %s_s = %s;\n", in.Name, scalarLoad(in.DType, fmt.Sprintf("%s[r / %d]", in.Name, in.group())))
		case Scalar:
			fmt.Fprintf(&b, "    const float %s_s = %s;\n", in.Name, scalarLoad(in.DType, in.Name+"[0]"))
		}
	}
	ref := func(a string) (string, error) {
		switch a {
		case "col":
			return "(float)EG_COL(j)", nil
		case "sgn":
			return "((EG_COL(j) & 1) ? 1.0f : -1.0f)", nil
		}
		if strings.HasPrefix(a, "~") {
			return a[1:] + "_v[j ^ 1]", nil
		}
		if syms[a].vector {
			return a + "_v[j]", nil
		}
		return a + "_s", nil
	}
	for _, v := range k.Values {
		args := make([]string, len(v.Args))
		for i, a := range v.Args {
			if args[i], err = ref(a); err != nil {
				return "", err
			}
		}
		expr, err := v.Body.CUDA(args...)
		if err != nil {
			return "", err
		}
		switch {
		case v.Reduce != "":
			init, comb, warp := "0.0f", "%s += %s", "eg_warp_sum"
			if v.Reduce == "max" {
				init, comb, warp = "-__uint_as_float(0x7f800000u)", "%s = fmaxf(%s, %s)", "eg_warp_max"
			}
			fmt.Fprintf(&b, "    float %s_s = %s;\n#pragma unroll\n    for (int j = 0; j < PER; ++j) ", v.Name, init)
			if v.Reduce == "max" {
				fmt.Fprintf(&b, comb+";\n", v.Name+"_s", v.Name+"_s", expr)
			} else {
				fmt.Fprintf(&b, comb+";\n", v.Name+"_s", expr)
			}
			fmt.Fprintf(&b, "    %s_s = %s(%s_s);\n", v.Name, warp, v.Name)
			if threads > 32 {
				op := 0
				if v.Reduce == "max" {
					op = 1
				}
				fmt.Fprintf(&b, "    %s_s = eg_block_combine<%d>(%s_s, eg_red);\n", v.Name, op, v.Name)
			}
		case syms[v.Name].vector:
			fmt.Fprintf(&b, "    float %s_v[PER];\n#pragma unroll\n    for (int j = 0; j < PER; ++j) %s_v[j] = %s;\n", v.Name, v.Name, expr)
		default:
			fmt.Fprintf(&b, "    const float %s_s = %s;\n", v.Name, expr)
		}
	}
	for _, o := range k.Outputs {
		group := fmt.Sprintf("r / %d", o.group())
		if o.Index != "" {
			fmt.Fprintf(&b, "    {\n        const long long g = (long long)%s[r / %d];\n        if (g >= 0) {\n", o.Index, o.group())
			group = "g"
		} else {
			b.WriteString("    {\n        {\n")
		}
		fmt.Fprintf(&b, "            %s* p = %s + %s;\n", o.DType.cType(), o.Name, addr(o, group))
		fmt.Fprintf(&b, "#pragma unroll\n            for (int j = 0; j < PER; j += VEC) eg_store<VEC>(p + EG_COL(j), &%s_v[j]);\n", o.Value)
		b.WriteString("        }\n    }\n")
	}
	b.WriteString("#undef EG_COL\n")
	return b.String(), nil
}

func scalarLoad(d DType, e string) string {
	switch d {
	case BF16:
		return "eg_bf16_to_f32(" + e + ")"
	case E4M3:
		return "eg_e4m3_to_f32(" + e + ")"
	}
	return "(float)" + e
}

// Launcher returns a host function
//
//	void Name_launch(<params...>, cudaStream_t stream)
func (k RowKernel) Launcher() (string, error) {
	if _, err := k.check(); err != nil {
		return "", err
	}
	return launcher(k.Name, k.params(), func(b *strings.Builder) {
		threads, _, _ := k.geometry()
		if threads == 32 {
			b.WriteString("    const dim3 grid((rows + 7) / 8), block(256);\n")
		} else {
			b.WriteString("    const dim3 grid(rows), block(256);\n")
		}
	}), nil
}

func launcher(name string, params []string, grid func(*strings.Builder)) string {
	names := make([]string, len(params))
	for i, p := range params {
		f := strings.Fields(p)
		names[i] = f[len(f)-1]
	}
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", name, strings.Join(params, ", "))
	grid(&b)
	fmt.Fprintf(&b, "    if (grid.x > 0) %s<<<grid, block, 0, stream>>>(%s);\n}\n", name, strings.Join(names, ", "))
	return b.String()
}

// Source returns a complete load_inline translation unit: [Prelude], the
// kernel and its launcher.
func (k RowKernel) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// Reference runs k on the CPU. bufs maps every input, output and index
// tensor name to its full buffer as float64 (low-precision data as the
// values it encodes); outputs are written in place, rounded to their dtype.
// Reductions accumulate in float64.
func (k RowKernel) Reference(rows int, bufs map[string][]float64) error {
	syms, err := k.check()
	if err != nil {
		return err
	}
	for name, s := range syms {
		if s.tensor != nil && bufs[name] == nil {
			return fmt.Errorf("row kernel %s reference: missing buffer %s", k.Name, name)
		}
	}
	D := k.D
	for r := 0; r < rows; r++ {
		vec := make(map[string][]float32)
		sca := make(map[string]float32)
		for _, in := range k.Inputs {
			buf := bufs[in.Name]
			switch in.Kind {
			case RowData:
				gs, rs := in.strides(D)
				base := in.Offset + r/in.group()*gs + r%in.group()*rs
				v := make([]float32, D)
				for c := range v {
					v[c] = float32(buf[base+c])
				}
				vec[in.Name] = v
			case ColVector:
				v := make([]float32, D)
				for c := range v {
					v[c] = float32(buf[c])
				}
				vec[in.Name] = v
			case RowScalar:
				sca[in.Name] = float32(buf[r/in.group()])
			case Scalar:
				sca[in.Name] = float32(buf[0])
			}
		}
		arg := func(a string, c int) float32 {
			switch {
			case a == "col":
				return float32(c)
			case a == "sgn":
				return float32(2*(c&1) - 1)
			case strings.HasPrefix(a, "~"):
				return vec[a[1:]][c^1]
			}
			if v, ok := vec[a]; ok {
				return v[c]
			}
			return sca[a]
		}
		for _, v := range k.Values {
			at := func(c int) float32 {
				args := make([]float32, len(v.Args))
				for i, a := range v.Args {
					args[i] = arg(a, c)
				}
				return v.Body.Eval(args...)
			}
			switch {
			case v.Reduce == "sum":
				var s float64
				for c := 0; c < D; c++ {
					s += float64(at(c))
				}
				sca[v.Name] = float32(s)
			case v.Reduce == "max":
				m := math.Inf(-1)
				for c := 0; c < D; c++ {
					m = math.Max(m, float64(at(c)))
				}
				sca[v.Name] = float32(m)
			case syms[v.Name].vector:
				out := make([]float32, D)
				for c := range out {
					out[c] = at(c)
				}
				vec[v.Name] = out
			default:
				sca[v.Name] = at(0)
			}
		}
		for _, o := range k.Outputs {
			g := r / o.group()
			if o.Index != "" {
				g = int(bufs[o.Index][g])
				if g < 0 {
					continue
				}
			}
			gs, rs := o.strides(D)
			base := o.Offset + g*gs + r%o.group()*rs
			for c, f := range vec[o.Value] {
				bufs[o.Name][base+c] = float64(o.DType.Round(f))
			}
		}
	}
	return nil
}
