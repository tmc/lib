package hopper

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// A Plan is the kernel sequence that computes an extracted term.
type Plan struct {
	Name    string
	Inputs  []Tensor // the term's Tensor leaves, in first-use order
	Temps   []Tensor // intermediates the caller allocates
	Output  Tensor
	Kernels []PlanKernel
}

// A PlanKernel is one launch of a Plan. Exactly one of GEMM, Tile, Row,
// Attention, TopK, Align and ExpertSum is set.
type PlanKernel struct {
	GEMM      *SkinnyGEMM
	Tile      *TileGEMM
	Row       *RowKernel
	Attention *PagedAttention
	TopK      *RowTopK
	Align     *ExpertAlign
	ExpertSum *ExpertSum
	M         int      // GEMM, Tile: rows of x
	MaxRows   int      // Grouped Tile: max rows per expert
	Experts   int      // Grouped Tile: local expert count
	Batch     int      // Attention: batch size
	MaxKeys   int      // Attention: max keys per request
	MaxPages  int      // Attention: max pages per request
	Rows      int      // Row, TopK, Align, ExpertSum: rows or tokens
	Args      []string // buffer names, in kernel parameter order (after leading size args)
}

// Compile lowers an extracted term to a Plan whose output is a tensor named
// name+"_out" of dtype out (F32 or BF16, or E4M3 for row kernels). gemm terms
// become [SkinnyGEMM] launches for up to 16 rows and [TileGEMM] launches
// beyond; trees of map, reduction, cast, col, sgn and
// pair over a common row width become one [RowKernel]; slices and reshapes of
// inputs become row addressing. A gemm whose result feeds a row tree is
// materialized to an FP32 temporary first.
func Compile(e *egraph.Expr, name string, out DType) (*Plan, error) {
	if !cIdent(name) {
		return nil, fmt.Errorf("compile: bad name %q", name)
	}
	c := &compiler{
		plan:   &Plan{Name: name},
		inputs: make(map[string]bool),
		done:   make(map[egraph.Id]string),
		multi:  make(map[egraph.Id][]Tensor),
	}
	if err := c.emit(e, name+"_out", out); err != nil {
		return nil, fmt.Errorf("compile %s: %w", name, err)
	}
	return c.plan, nil
}

type compiler struct {
	plan   *Plan
	inputs map[string]bool
	done   map[egraph.Id]string   // e-class → buffer holding it
	multi  map[egraph.Id][]Tensor // e-class → buffers holding a multi-output op
	temps  int
}

func (c *compiler) input(t Tensor) {
	if !c.inputs[t.Name] {
		c.inputs[t.Name] = true
		c.plan.Inputs = append(c.plan.Inputs, t)
	}
}

// buffer returns the name of a buffer holding e, emitting kernels as needed.
func (c *compiler) buffer(e *egraph.Expr) (Tensor, error) {
	if t, ok := e.Data.(Tensor); ok {
		c.input(t)
		return t, nil
	}
	if t, ok, err := c.modelProjectionBuffer(e); ok || err != nil {
		return t, err
	}
	if name, ok := c.done[e.Class]; ok && e.Class != 0 {
		for _, t := range c.plan.Temps {
			if t.Name == name {
				return t, nil
			}
		}
	}
	c.temps++
	name := fmt.Sprintf("%s_t%d", c.plan.Name, c.temps)
	if err := c.emit(e, name, defaultTempDType(e)); err != nil {
		return Tensor{}, err
	}
	return c.plan.Temps[len(c.plan.Temps)-1], nil
}

// emit appends the kernels computing e into a new tensor dst.
func (c *compiler) emit(e *egraph.Expr, dst string, d DType) error {
	if e.Op == OpCast {
		cd, ok := e.Children[0].Data.(DType)
		if !ok {
			return fmt.Errorf("cast without dtype")
		}
		if cd == d {
			return c.emit(e.Children[1], dst, d)
		}
	}
	shape, err := shapeOf(e)
	if err != nil {
		return err
	}
	t := Tensor{Name: dst, DType: d, Shape: shape}
	switch {
	case e.Op == OpMatMul:
		gemmExpr := &egraph.Expr{
			Op:       OpGEMM,
			Class:    e.Class,
			Children: []*egraph.Expr{{Op: "fn:$0"}, e.Children[0], e.Children[1]},
		}
		if err := c.gemm(gemmExpr, t); err != nil {
			return err
		}
	case e.Op == OpGEMM:
		if err := c.gemm(e, t); err != nil {
			return err
		}
	case isModelOp(e.Op):
		if err := c.emitModel(e, t); err != nil {
			return err
		}
	default:
		if err := c.rows(e, t); err != nil {
			return err
		}
	}
	if dst == c.plan.Name+"_out" {
		c.plan.Output = t
	} else {
		c.plan.Temps = append(c.plan.Temps, t)
	}
	if e.Class != 0 {
		c.done[e.Class] = dst
	}
	return nil
}

func (c *compiler) gemm(e *egraph.Expr, out Tensor) error {
	f, ok := lambdaOf(egraph.ENode{Op: e.Children[0].Op})
	if !ok {
		return fmt.Errorf("gemm: bad epilogue %q", e.Children[0].Op)
	}
	x, err := c.buffer(e.Children[1])
	if err != nil {
		return err
	}
	if x.DType != BF16 || len(x.Shape) != 2 {
		return fmt.Errorf("gemm: x must be a BF16 matrix, have %v %v (cast it)", x.DType, x.Shape)
	}
	w := e.Children[2]
	g := SkinnyGEMM{Name: c.plan.Name + "_" + out.Name, MaxM: x.Shape[0], Out: out.DType, Epilogue: f, K: x.Shape[1]}
	args := []string{x.Name}
	switch w.Op {
	case OpHiLo:
		g.Weight = WeightBF16HiLo
		for _, k := range w.Children {
			t, ok := k.Data.(Tensor)
			if !ok || t.DType != BF16 {
				return fmt.Errorf("gemm: hilo weights must be BF16 inputs")
			}
			c.input(t)
			args = append(args, t.Name)
			g.N = t.Shape[0]
		}
	case OpNVFP4:
		g.Weight = WeightNVFP4
		for _, k := range w.Children {
			t, ok := k.Data.(Tensor)
			if !ok {
				return fmt.Errorf("gemm: nvfp4 operands must be inputs")
			}
			c.input(t)
			args = append(args, t.Name)
		}
		g.N = w.Children[0].Data.(Tensor).Shape[0]
	default:
		t, ok := w.Data.(Tensor)
		if !ok || t.DType != BF16 || len(t.Shape) != 2 {
			return fmt.Errorf("gemm: weight must be a BF16 [N, K] input, hilo or nvfp4")
		}
		c.input(t)
		g.Weight = WeightBF16
		g.N = t.Shape[0]
		args = append(args, t.Name)
	}
	m, n := x.Shape[0], g.N
	for i, k := range e.Children[3:] {
		t, err := c.buffer(k)
		if err != nil {
			return err
		}
		var bc Broadcast
		switch {
		case slices.Equal(trim1(t.Shape), []int{n}) && t.Shape[len(t.Shape)-1] == n:
			bc = PerCol
		case slices.Equal(t.Shape, []int{m, n}):
			bc = Full
		case slices.Equal(t.Shape, []int{m, 1}) || slices.Equal(t.Shape, []int{m}):
			bc = PerRow
		default:
			return fmt.Errorf("gemm: epilogue operand %d shaped %v does not broadcast to [%d %d]", i+1, t.Shape, m, n)
		}
		if t.DType != F32 && t.DType != BF16 {
			return fmt.Errorf("gemm: epilogue operand %s is %v", t.Name, t.DType)
		}
		g.Extras = append(g.Extras, Operand{Name: fmt.Sprintf("e%d", i+1), DType: t.DType, Broadcast: bc})
		args = append(args, t.Name)
	}
	args = append(args, out.Name)
	if m > 16 {
		t := TileGEMM{Name: g.Name, N: g.N, K: g.K, Weight: g.Weight, Out: g.Out, Epilogue: g.Epilogue, Extras: g.Extras}
		if t.N%128 != 0 {
			t.BN = 64
		}
		if m <= 64 {
			t.BM = 64
		}
		if _, err := t.norm(); err != nil {
			return err
		}
		c.plan.Kernels = append(c.plan.Kernels, PlanKernel{Tile: &t, M: m, Args: args})
		return nil
	}
	if _, err := g.norm(); err != nil {
		return err
	}
	c.plan.Kernels = append(c.plan.Kernels, PlanKernel{GEMM: &g, M: m, Args: args})
	return nil
}

func trim1(s []int) []int {
	for len(s) > 1 && s[0] == 1 {
		s = s[1:]
	}
	return s
}

// rows lowers a map/reduce tree with out's shape to one RowKernel.
func (c *compiler) rows(e *egraph.Expr, out Tensor) error {
	if len(out.Shape) == 0 {
		return fmt.Errorf("rows: scalar result")
	}
	domain := out.Shape
	d := domain[len(domain)-1]
	nrows := size(domain) / d
	k := &RowKernel{Name: c.plan.Name + "_" + out.Name, D: d}
	rb := &rowBuilder{c: c, k: k, domain: domain, names: make(map[egraph.Id]string), bufs: make(map[string]bool)}
	v, err := rb.value(e)
	if err != nil {
		return err
	}
	if !rb.isVector(v) {
		return fmt.Errorf("rows: result is not a row of %d values", d)
	}
	group, gs, rs, err := rowGeometry(domain, compact(domain), 0)
	if err != nil {
		return err
	}
	k.Outputs = append(k.Outputs, RowTensor{Name: out.Name, DType: out.DType, Kind: RowData, Value: v, Group: group, GroupStride: gs, RowStride: rs})
	if _, err := k.check(); err != nil {
		return err
	}
	pk := PlanKernel{Row: k, Rows: nrows}
	for _, p := range k.params()[1:] {
		f := strings.Fields(p)
		pk.Args = append(pk.Args, f[len(f)-1])
	}
	c.plan.Kernels = append(c.plan.Kernels, pk)
	return nil
}

type rowBuilder struct {
	c      *compiler
	k      *RowKernel
	domain []int
	names  map[egraph.Id]string
	vector map[string]bool
	bufs   map[string]bool
	n      int
}

func (rb *rowBuilder) fresh(prefix string) string {
	rb.n++
	return fmt.Sprintf("%s%d", prefix, rb.n)
}

// value returns the RowKernel name holding e.
func (rb *rowBuilder) value(e *egraph.Expr) (string, error) {
	if rb.vector == nil {
		rb.vector = make(map[string]bool)
	}
	if n, ok := rb.names[e.Class]; ok && e.Class != 0 {
		return n, nil
	}
	n, err := rb.value1(e)
	if err != nil {
		return "", err
	}
	if e.Class != 0 {
		rb.names[e.Class] = n
	}
	return n, nil
}

func (rb *rowBuilder) def(v RowValue, vector bool) string {
	rb.k.Values = append(rb.k.Values, v)
	rb.vector[v.Name] = vector
	return v.Name
}

func (rb *rowBuilder) value1(e *egraph.Expr) (string, error) {
	switch e.Op {
	case OpMap:
		f, ok := lambdaOf(egraph.ENode{Op: e.Children[0].Op})
		if !ok {
			return "", fmt.Errorf("rows: bad lambda %q", e.Children[0].Op)
		}
		var args []string
		vec := false
		for _, k := range e.Children[1:] {
			a, err := rb.arg(k)
			if err != nil {
				return "", err
			}
			args = append(args, a)
			vec = vec || rb.isVector(a)
		}
		return rb.def(RowValue{Name: rb.fresh("v"), Body: f, Args: args}, vec), nil
	case OpReduceSum, OpReduceMax:
		a, err := rb.arg(e.Children[0])
		if err != nil {
			return "", err
		}
		if !rb.isVector(a) {
			return "", fmt.Errorf("rows: reduction over a row scalar")
		}
		op := "sum"
		if e.Op == OpReduceMax {
			op = "max"
		}
		return rb.def(RowValue{Name: rb.fresh("s"), Reduce: op, Body: Identity, Args: []string{a}}, false), nil
	case OpCast:
		dt, _ := e.Children[0].Data.(DType)
		a, err := rb.arg(e.Children[1])
		if err != nil {
			return "", err
		}
		body := Identity
		switch dt {
		case BF16:
			body = MustLambda("(bf16 $0)")
		case E4M3:
			body = MustLambda("(e4m3 $0)")
		case F32:
		default:
			return "", fmt.Errorf("rows: cast to %v", dt)
		}
		return rb.def(RowValue{Name: rb.fresh("v"), Body: body, Args: []string{a}}, rb.isVector(a)), nil
	}
	a, err := rb.arg(e)
	if err != nil {
		return "", err
	}
	if strings.HasPrefix(a, "~") || a == "col" || a == "sgn" {
		return rb.def(RowValue{Name: rb.fresh("v"), Body: Identity, Args: []string{a}}, true), nil
	}
	return a, nil
}

func (rb *rowBuilder) isVector(a string) bool {
	if a == "col" || a == "sgn" || strings.HasPrefix(a, "~") {
		return true
	}
	if v, ok := rb.vector[a]; ok {
		return v
	}
	for _, in := range rb.k.Inputs {
		if in.Name == a {
			return in.Kind == RowData || in.Kind == ColVector
		}
	}
	return false
}

// arg returns the RowValue argument for e: an input, a value, or a special.
func (rb *rowBuilder) arg(e *egraph.Expr) (string, error) {
	switch e.Op {
	case OpCol, OpSgn:
		return e.Op, nil
	case OpPair:
		v, err := rb.value(e.Children[0])
		if err != nil {
			return "", err
		}
		return "~" + v, nil
	case OpMap, OpReduceSum, OpReduceMax, OpCast:
		return rb.value(e)
	}
	if n, ok := rb.names[e.Class]; ok && e.Class != 0 {
		return n, nil
	}
	// A view of a buffer: a leaf, a gemm result, or slices and reshapes of one.
	view, err := rb.view(e)
	if err != nil {
		return "", err
	}
	in, err := rb.classify(view)
	if err != nil {
		return "", err
	}
	in.Name = rb.fresh("in")
	rb.k.Inputs = append(rb.k.Inputs, in)
	if e.Class != 0 {
		rb.names[e.Class] = in.Name
	}
	return in.Name, nil
}

// A view is an affine window onto a buffer.
type view struct {
	buf     Tensor
	offset  int
	shape   []int
	strides []int
}

func compact(shape []int) []int {
	s := make([]int, len(shape))
	n := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = n
		n *= shape[i]
	}
	return s
}

func (rb *rowBuilder) view(e *egraph.Expr) (view, error) {
	switch e.Op {
	case OpSlice:
		v, err := rb.view(e.Children[0])
		if err != nil {
			return view{}, err
		}
		start, _ := e.Children[1].Data.(int)
		n, _ := e.Children[2].Data.(int)
		last := len(v.shape) - 1
		if start < 0 || n < 1 || start+n > v.shape[last] {
			return view{}, fmt.Errorf("rows: bad slice")
		}
		v.offset += start * v.strides[last]
		v.shape = append(slices.Clone(v.shape[:last]), n)
		return v, nil
	case OpReshape:
		v, err := rb.view(e.Children[0])
		if err != nil {
			return view{}, err
		}
		dims, _ := e.Children[1].Data.(Dims)
		if size(dims.Shape) != size(v.shape) {
			return view{}, fmt.Errorf("rows: bad reshape")
		}
		if slices.Equal(v.strides, compact(v.shape)) {
			v.shape, v.strides = slices.Clone(dims.Shape), compact(dims.Shape)
			return v, nil
		}
		// Split the last axis of a strided view: [..., a*b] -> [..., a, b].
		last := len(v.shape) - 1
		if len(dims.Shape) == len(v.shape)+1 && slices.Equal(dims.Shape[:last], v.shape[:last]) && v.strides[last] == 1 {
			b := dims.Shape[last+1]
			v.shape = slices.Clone(dims.Shape)
			v.strides = append(slices.Clone(v.strides[:last]), b, 1)
			return v, nil
		}
		return view{}, fmt.Errorf("rows: reshape of a strided view %v to %v", v.shape, dims.Shape)
	}
	t, err := rb.c.buffer(e)
	if err != nil {
		return view{}, err
	}
	return view{buf: t, shape: slices.Clone(t.Shape), strides: compact(t.Shape)}, nil
}

// classify turns a view into a RowTensor input relative to the row domain.
func (rb *rowBuilder) classify(v view) (RowTensor, error) {
	dom := rb.domain
	if len(v.shape) > len(dom) {
		return RowTensor{}, fmt.Errorf("rows: operand rank %d above domain %v", len(v.shape), dom)
	}
	// Left-pad to the domain's rank.
	pad := len(dom) - len(v.shape)
	shape := append(make([]int, 0, len(dom)), v.shape...)
	strides := append(make([]int, 0, len(dom)), v.strides...)
	for range pad {
		shape = append([]int{1}, shape...)
		strides = append([]int{0}, strides...)
	}
	in := RowTensor{DType: v.buf.DType, Offset: v.offset, Buffer: v.buf.Name}
	last := len(dom) - 1
	ones := func(s []int) bool {
		for _, d := range s {
			if d != 1 {
				return false
			}
		}
		return true
	}
	switch {
	case slices.Equal(shape, dom):
		if strides[last] != 1 {
			return RowTensor{}, fmt.Errorf("rows: operand columns are strided")
		}
		g, gs, rs, err := rowGeometry(dom, strides, v.offset)
		if err != nil {
			return RowTensor{}, err
		}
		in.Kind, in.Group, in.GroupStride, in.RowStride = RowData, g, gs, rs
	case ones(shape[:last]) && shape[last] == dom[last]:
		if strides[last] != 1 {
			return RowTensor{}, fmt.Errorf("rows: column vector is strided")
		}
		in.Kind = ColVector
	case ones(shape):
		in.Kind = Scalar
	case shape[last] == 1:
		// A prefix of the domain's leading dims, then ones: one value per row group.
		i := 0
		for i < last && shape[i] == dom[i] {
			i++
		}
		if !ones(shape[i:]) || !slices.Equal(strides[:i], compact(shape)[:i]) {
			return RowTensor{}, fmt.Errorf("rows: operand %v does not broadcast to %v", v.shape, dom)
		}
		in.Kind, in.Group = RowScalar, size(dom[i:last])
	default:
		return RowTensor{}, fmt.Errorf("rows: operand %v does not broadcast to %v", v.shape, dom)
	}
	return in, nil
}

// rowGeometry maps the leading dims of a strided [d0, ..., dk, D] view onto
// RowTensor's (group, group stride, row stride) addressing.
func rowGeometry(dom, strides []int, offset int) (group, gs, rs int, err error) {
	lead := dom[:len(dom)-1]
	st := strides[:len(strides)-1]
	switch len(lead) {
	case 0:
		return 1, dom[0], dom[0], nil
	case 1:
		return 1, st[0], st[0], nil
	}
	// Merge all but the last leading dim into one, if compact among themselves.
	k := len(lead) - 1
	for i := 0; i < k-1; i++ {
		if st[i] != st[i+1]*lead[i+1] {
			return 0, 0, 0, fmt.Errorf("rows: leading dims %v are not mergeable", lead)
		}
	}
	return lead[k], st[k-1], st[k], nil
}

// shapeOf infers e's shape.
func shapeOf(e *egraph.Expr) ([]int, error) {
	if t, ok := e.Data.(Tensor); ok {
		return t.Shape, nil
	}
	kid := func(i int) ([]int, error) { return shapeOf(e.Children[i]) }
	switch e.Op {
	case OpMap:
		var shapes [][]int
		for i := 1; i < len(e.Children); i++ {
			s, err := kid(i)
			if err != nil {
				return nil, err
			}
			shapes = append(shapes, s)
		}
		return broadcastShape(shapes...)
	case OpReduceSum, OpReduceMax:
		s, err := kid(0)
		if err != nil {
			return nil, err
		}
		return append(slices.Clone(s[:len(s)-1]), 1), nil
	case OpSlice:
		s, err := kid(0)
		if err != nil {
			return nil, err
		}
		n, _ := e.Children[2].Data.(int)
		return append(slices.Clone(s[:len(s)-1]), n), nil
	case OpReshape:
		dims, ok := e.Children[1].Data.(Dims)
		if !ok {
			return nil, fmt.Errorf("reshape without dims")
		}
		return dims.Shape, nil
	case OpCast:
		return kid(1)
	case OpCol, OpSgn, OpPair:
		return kid(0)
	case OpMatMul, OpGEMM:
		xi, wi := 0, 1
		if e.Op == OpGEMM {
			xi, wi = 1, 2
		}
		x, err := kid(xi)
		if err != nil {
			return nil, err
		}
		w := e.Children[wi]
		var n int
		switch w.Op {
		case OpHiLo, OpNVFP4:
			s, err := shapeOf(w.Children[0])
			if err != nil {
				return nil, err
			}
			n = s[0]
		default:
			s, err := shapeOf(w)
			if err != nil {
				return nil, err
			}
			n = s[0]
		}
		mm := []int{x[0], n}
		if e.Op == OpMatMul {
			return mm, nil
		}
		shapes := [][]int{mm}
		for i := 3; i < len(e.Children); i++ {
			s, err := kid(i)
			if err != nil {
				return nil, err
			}
			shapes = append(shapes, s)
		}
		return broadcastShape(shapes...)
	}
	return shapeOfModel(e)
}

// Source returns one load_inline translation unit with every kernel of the
// plan and a host function
//
//	void Name_run(<inputs...>, <temps...>, <output>, cudaStream_t stream)
//
// that launches them in order. Split-K GEMMs are not used by plans.
func (p *Plan) Source() (string, error) {
	var b strings.Builder
	b.WriteString(Prelude)
	for _, k := range p.Kernels {
		var src, l string
		var err error
		switch {
		case k.GEMM != nil:
			if src, err = k.GEMM.CUDA(); err == nil {
				l, err = k.GEMM.Launcher()
			}
		case k.Tile != nil:
			if src, err = k.Tile.CUDA(); err == nil {
				l, err = k.Tile.Launcher()
			}
		case k.Attention != nil:
			if src, err = k.Attention.CUDA(); err == nil {
				l, err = k.Attention.Launcher()
			}
		case k.TopK != nil:
			if src, err = k.TopK.CUDA(); err == nil {
				l, err = k.TopK.Launcher()
			}
		case k.Align != nil:
			if src, err = k.Align.CUDA(); err == nil {
				l, err = k.Align.Launcher()
			}
		case k.ExpertSum != nil:
			if src, err = k.ExpertSum.CUDA(); err == nil {
				l, err = k.ExpertSum.Launcher()
			}
		default:
			if src, err = k.Row.CUDA(); err == nil {
				l, err = k.Row.Launcher()
			}
		}
		if err != nil {
			return "", err
		}
		b.WriteString("\n" + src + "\n" + l)
	}
	var params []string
	for _, t := range p.Inputs {
		params = append(params, "const "+t.DType.cType()+"* "+t.Name)
	}
	for _, t := range append(slices.Clone(p.Temps), p.Output) {
		params = append(params, t.DType.cType()+"* "+t.Name)
	}
	fmt.Fprintf(&b, "\ninline void %s_run(%s, cudaStream_t stream) {\n", p.Name, strings.Join(params, ", "))
	for _, k := range p.Kernels {
		switch {
		case k.GEMM != nil:
			fmt.Fprintf(&b, "    %s_launch(%d, %s, stream);\n", k.GEMM.Name, k.M, strings.Join(k.Args, ", "))
		case k.Tile != nil:
			if k.Tile.Grouped {
				fmt.Fprintf(&b, "    %s_launch(%d, %d, %d, %s, stream);\n", k.Tile.Name, k.M, k.MaxRows, k.Experts, strings.Join(k.Args, ", "))
			} else {
				fmt.Fprintf(&b, "    %s_launch(%d, %s, stream);\n", k.Tile.Name, k.M, strings.Join(k.Args, ", "))
			}
		case k.Attention != nil:
			// Args are q, k_cache, v_cache, page_table, seqlens, k_scale, v_scale, [gate,] out, ws;
			// Launcher expects max_pages after page_table.
			args := slices.Clone(k.Args)
			if len(args) >= 4 {
				args = append(args[:4], append([]string{fmt.Sprint(k.MaxPages)}, args[4:]...)...)
			}
			fmt.Fprintf(&b, "    %s_launch(%d, %d, %s, stream);\n", k.Attention.Name, k.Batch, k.MaxKeys, strings.Join(args, ", "))
		case k.TopK != nil:
			fmt.Fprintf(&b, "    %s_launch(%d, %s, stream);\n", k.TopK.Name, k.Rows, strings.Join(k.Args, ", "))
		case k.Align != nil:
			fmt.Fprintf(&b, "    %s_launch(%d, %s, stream);\n", k.Align.Name, k.Rows, strings.Join(k.Args, ", "))
		case k.ExpertSum != nil:
			fmt.Fprintf(&b, "    %s_launch(%d, %s, stream);\n", k.ExpertSum.Name, k.Rows, strings.Join(k.Args, ", "))
		default:
			fmt.Fprintf(&b, "    %s_launch(%d, %s, stream);\n", k.Row.Name, k.Rows, strings.Join(k.Args, ", "))
		}
	}
	b.WriteString("}\n")
	return b.String(), nil
}
