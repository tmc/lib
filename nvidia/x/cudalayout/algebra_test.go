package cudalayout

import (
	"bufio"
	"errors"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func mustParse(t testing.TB, s string) Layout {
	t.Helper()
	l, err := ParseLayout(s)
	if err != nil {
		t.Fatalf("ParseLayout(%q): %v", s, err)
	}
	return l
}

// offsets returns every offset of an integer or F2 layout, in index order.
func offsets(t testing.TB, l Layout) []int {
	t.Helper()
	out := make([]int, l.Size())
	for i := range out {
		o, err := l.Index(i)
		if err != nil {
			t.Fatalf("%v.Index(%d): %v", l, i, err)
		}
		out[i] = o
	}
	return out
}

// The hand oracles are colexicographic by construction: the first leaf of a
// coordinate varies fastest. A lexicographic evaluator reverses the leaves
// and fails every non-symmetric row.
func TestShapeStrideColexicographicalMapping(t *testing.T) {
	tests := []struct {
		layout string
		want   []int
	}{
		{"8:1", []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{"(4,2):(1,4)", []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{"(4,2):(2,1)", []int{0, 2, 4, 6, 1, 3, 5, 7}},
		{"((2,2),2):((4,1),2)", []int{0, 4, 1, 5, 2, 6, 3, 7}},
		{"(2,(2,2)):(0,(1,10))", []int{0, 0, 1, 1, 10, 10, 11, 11}},
		{"(3,2):(-1,3)", []int{0, -1, -2, 3, 2, 1}},
	}
	for _, tt := range tests {
		l := mustParse(t, tt.layout)
		if got := offsets(t, l); !slices.Equal(got, tt.want) {
			t.Errorf("%s: offsets %v, want %v", tt.layout, got, tt.want)
		}
		for i, want := range tt.want {
			natural, err := l.Coordinate(i, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got, err := l.Offset(natural...); err != nil || got != want {
				t.Errorf("%s: Offset(natural %v) = %d, %v, want %d", tt.layout, natural, got, err, want)
			}
			if got, err := l.Offset(i); err != nil || got != want {
				t.Errorf("%s: Offset(%d) = %d, %v, want %d", tt.layout, i, got, err, want)
			}
		}
	}

	// A per-top-mode coordinate decomposes each mode's integer on its own.
	l := mustParse(t, "((2,2),2):((4,1),2)")
	for _, tt := range []struct {
		crd  []int
		want int
	}{
		{[]int{3, 1}, 7},    // mode 0: 3 -> (1,1) -> 5; mode 1: 1 -> 2
		{[]int{2, 0}, 1},    // mode 0: 2 -> (0,1) -> 1
		{[]int{1, 0, 1}, 6}, // natural: 4 + 0 + 2
		{[]int{0, 1, 0}, 1},
	} {
		if got, err := l.Offset(tt.crd...); err != nil || got != tt.want {
			t.Errorf("Offset(%v) = %d, %v, want %d", tt.crd, got, err, tt.want)
		}
	}
	for _, crd := range [][]int{{8}, {-1}, {4, 0}, {0, 2}, {2, 0, 0}, {0, 0, 0, 0}} {
		if _, err := l.Offset(crd...); !errors.Is(err, ErrOutOfBounds) {
			t.Errorf("Offset(%v) = %v, want ErrOutOfBounds", crd, err)
		}
	}
	if got, _ := l.Coordinate(5, nil); !slices.Equal(got, []int{1, 0, 1}) {
		t.Errorf("Coordinate(5) = %v, want [1 0 1]", got)
	}

	// Basis strides give coordinate-valued offsets, the codomain of an
	// identity tensor.
	b := mustParse(t, "(4,(2,3)):(1@0,(4@0,1@1))")
	var got [][]int
	for i := range b.Size() {
		o, err := b.BasisOffset(i, nil)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, o)
	}
	if want := [][]int{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {0, 1}}; !slices.EqualFunc(got[:9], want, slices.Equal) {
		t.Errorf("basis offsets %v, want prefix %v", got[:9], want)
	}
	if _, err := b.Index(0); !errors.Is(err, ErrUnsupported) {
		t.Errorf("Index on a basis layout = %v, want ErrUnsupported", err)
	}
	buf := make([]int, 0, 4)
	if o, _ := b.BasisOffset(9, buf); &o[0] != &buf[:1][0] || !slices.Equal(o, []int{1, 1}) {
		t.Errorf("BasisOffset(9, buf) = %v, want [1 1] in buf", o)
	}
	if _, err := (Layout{}).Index(0); !errors.Is(err, ErrUninitialized) {
		t.Errorf("zero Layout Index = %v, want ErrUninitialized", err)
	}
	if _, err := l.Mode(2); !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("Mode(2) of a rank-2 layout = %v, want ErrOutOfBounds", err)
	}
	if _, err := mustParse(t, "2:"+strconv.Itoa(math.MaxInt)).Cosize(); !errors.Is(err, ErrOverflow) {
		t.Errorf("Cosize reaching MaxInt = %v, want ErrOverflow", err)
	}
	for _, s := range []string{"(3,2):(-1,3)", "(4,2):(1@0,1@1)"} {
		if _, err := mustParse(t, s).Cosize(); !errors.Is(err, ErrUnsupported) {
			t.Errorf("Cosize of %s = %v, want ErrUnsupported", s, err)
		}
	}
}

func TestLayoutIncongruentShapeStrideRejection(t *testing.T) {
	tests := []struct {
		name   string
		shape  Shape
		stride Stride
		want   error
	}{
		{"tuple shape, leaf stride", TupleShape(IntShape(4), IntShape(2)), IntStride(1), ErrUnsupported},
		{"leaf shape, tuple stride", IntShape(4), TupleStride(IntStride(1), IntStride(4)), ErrUnsupported},
		{"rank mismatch", TupleShape(IntShape(4), IntShape(2)), TupleStride(IntStride(1)), ErrUnsupported},
		{"stride has more modes", TupleShape(IntShape(4)), TupleStride(IntStride(1), IntStride(4)), ErrUnsupported},
		{"nested mismatch", TupleShape(IntShape(4), TupleShape(IntShape(2), IntShape(2))), TupleStride(IntStride(1), IntStride(4)), ErrUnsupported},
		{"empty tuple", TupleShape(), TupleStride(), ErrUnsupported},
		{"zero extent", IntShape(0), IntStride(1), ErrUnsupported},
		{"negative extent", TupleShape(IntShape(4), IntShape(-2)), TupleStride(IntStride(1), IntStride(4)), ErrUnsupported},
		{"mixed int and basis", TupleShape(IntShape(4), IntShape(2)), TupleStride(IntStride(1), BasisStride(1, 1)), ErrUnsupported},
		{"mixed F2 and int", TupleShape(IntShape(4), IntShape(2)), TupleStride(F2Stride(1, 2), IntStride(4)), ErrUnsupported},
		{"F2 bits for the wrong extent", IntShape(8), F2Stride(1, 2), ErrUnsupported},
		{"F2 negative bit", IntShape(2), F2Stride(-1), ErrUnsupported},
		{"negative basis axis", IntShape(2), BasisStride(-1, 1), ErrUnsupported},
		{"size overflow", TupleShape(IntShape(1<<40), IntShape(1<<40)), TupleStride(IntStride(0), IntStride(0)), ErrOverflow},
		{"offset overflow", IntShape(1 << 20), IntStride(1 << 50), ErrOverflow},
	}
	for _, tt := range tests {
		if _, err := NewLayout(tt.shape, tt.stride); !errors.Is(err, tt.want) {
			t.Errorf("%s: NewLayout(%v, %v) = %v, want %v", tt.name, tt.shape, tt.stride, err, tt.want)
		}
	}

	// The same trees, made congruent, are accepted: the rejections above
	// are about congruence, not about the trees.
	for _, s := range []string{"(4,2):(1,4)", "(4):(1)", "(4,(2,2)):(1,(4,8))", "(4,2):(0,1@1)", "(4,2):(1@0,0)"} {
		if _, err := ParseLayout(s); err != nil {
			t.Errorf("ParseLayout(%q) = %v, want accepted", s, err)
		}
	}
	if _, err := NewLayout(TupleShape(IntShape(4), IntShape(2)), TupleStride(F2Stride(1, 2), IntStride(0))); err != nil {
		t.Errorf("F2 with an integer stride-0 mode = %v, want accepted", err)
	}
	for _, s := range []string{"(4,2):1", "4:(1,2)", "(4,2)", "(4,2:(1,4)", "(4,x):(1,4)", "4:1 junk"} {
		if _, err := ParseLayout(s); !errors.Is(err, ErrUnsupported) {
			t.Errorf("ParseLayout(%q) = %v, want ErrUnsupported", s, err)
		}
	}
}

func TestLayoutCoalesceModeReduction(t *testing.T) {
	tests := []struct {
		in, want, byMode string
	}{
		{"(2,4):(1,2)", "8:1", "(2,4):(1,2)"},
		{"(2,4):(4,1)", "(2,4):(4,1)", "(2,4):(4,1)"},
		{"((2,1),4):((1,9),2)", "8:1", "(2,4):(1,2)"},
		{"((2,2),(2,3)):((1,2),(8,16))", "(4,6):(1,8)", "(4,6):(1,8)"},
		{"(1,1):(3,5)", "1:0", "(1,1):(0,0)"},
		{"(4,2):(0,0)", "8:0", "(4,2):(0,0)"},
		{"(2,4):(1@0,2@0)", "8:1@0", "(2,4):(1@0,2@0)"},
		{"(2,4):(1@0,2@1)", "(2,4):(1@0,2@1)", "(2,4):(1@0,2@1)"},
	}
	for _, tt := range tests {
		l := mustParse(t, tt.in)
		if got := Coalesce(l).String(); got != tt.want {
			t.Errorf("Coalesce(%s) = %s, want %s", tt.in, got, tt.want)
		}
		if got := CoalesceByMode(l).String(); got != tt.byMode {
			t.Errorf("CoalesceByMode(%s) = %s, want %s", tt.in, got, tt.byMode)
		}
	}

	f2, err := NewLayout(TupleShape(IntShape(2), IntShape(4)), TupleStride(F2Stride(1), F2Stride(6, 16)))
	if err != nil {
		t.Fatal(err)
	}
	c := Coalesce(f2)
	if got, want := c.String(), "8:xor(1,6,16)"; got != want {
		t.Errorf("Coalesce(%v) = %s, want %s", f2, got, want)
	}
	if !slices.Equal(offsets(t, c), offsets(t, f2)) {
		t.Errorf("Coalesce(%v) changed the function", f2)
	}

	// Coalesce must not change the function, and must not leave two modes
	// it could have merged.
	for _, s := range []string{"((2,2),(2,3)):((1,2),(8,16))", "(3,(2,4)):(1,(3,6))", "(2,3,4):(12,4,1)", "((4,1),(2,2)):((0,5),(0,4))"} {
		l := mustParse(t, s)
		c := Coalesce(l)
		if !slices.Equal(offsets(t, c), offsets(t, l)) {
			t.Errorf("Coalesce(%s) = %v computes a different function", s, c)
		}
		for i := 1; i < len(c.leaves); i++ {
			if p, q := c.leaves[i-1], c.leaves[i]; p.n*p.d == q.d {
				t.Errorf("Coalesce(%s) = %v left mergeable modes %d and %d", s, c, i-1, i)
			}
		}
		b := CoalesceByMode(l)
		if b.Rank() != l.Rank() {
			t.Errorf("CoalesceByMode(%s) = %v changed rank", s, b)
		}
		for i := range l.Size() {
			crd := l.modeCoordinate(i)
			want, err1 := l.Offset(crd...)
			got, err2 := b.Offset(crd...)
			if err1 != nil || err2 != nil || got != want {
				t.Errorf("CoalesceByMode(%s): Offset(%v) = %d, %v, want %d, %v", s, crd, got, err2, want, err1)
			}
		}
	}
}

// modeCoordinate returns the per-top-mode coordinate of integral index i.
func (l Layout) modeCoordinate(i int) []int {
	var crd []int
	for m := range l.Rank() {
		n := 1
		for _, f := range l.leaves[l.modes[m]:l.modes[m+1]] {
			n *= f.n
		}
		crd = append(crd, i%n)
		i /= n
	}
	return crd
}

func TestLayoutCompositionInvariants(t *testing.T) {
	// Expected results are from the CuTe layout algebra documentation
	// (media/docs/cpp/cute/02_layout_algebra.md) and agree with pycute.
	tests := []struct {
		a, b, want string
	}{
		{"20:2", "(5,4):(4,1)", "(5,4):(8,2)"},
		{"(6,2):(8,2)", "(4,3):(3,1)", "((2,2),3):((24,2),8)"},
		{"(10,2):(16,4)", "(5,4):(1,5)", "(5,(2,2)):(16,(80,4))"},
		{"8:1", "4:0", "4:0"},
		{"(4,4):(4,1)", "8:1", "(4,2):(4,1)"},
		{"(4,8):(1@0,1@1)", "(2,4):(2,8)", "(2,4):(2@0,2@1)"},
	}
	for _, tt := range tests {
		a, b := mustParse(t, tt.a), mustParse(t, tt.b)
		r, err := Composition(a, b)
		if err != nil {
			t.Errorf("Composition(%s, %s): %v", tt.a, tt.b, err)
			continue
		}
		if r.String() != tt.want {
			t.Errorf("Composition(%s, %s) = %v, want %s", tt.a, tt.b, r, tt.want)
		}
		checkComposes(t, a, b, r)
	}

	// By mode: the CuTe documentation's tiler example.
	a := mustParse(t, "(12,(4,8)):(59,(13,1))")
	tiler := []Layout{mustParse(t, "3:4"), mustParse(t, "8:2")}
	r, err := ComposeByMode(a, tiler)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := r.String(), "(3,(2,4)):(236,(26,1))"; got != want {
		t.Errorf("ComposeByMode = %s, want %s", got, want)
	}

	// A shape used as a tiler takes the leading block of each mode.
	st, err := ShapeTiler(3, 4)
	if err != nil {
		t.Fatal(err)
	}
	r, err = ComposeByMode(mustParse(t, "(6,8):(1,6)"), st)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := r.String(), "(3,4):(1,6)"; got != want {
		t.Errorf("ComposeByMode with shape (3,4) = %s, want %s", got, want)
	}
	if _, err := ShapeTiler(0); !errors.Is(err, ErrUnsupported) {
		t.Errorf("ShapeTiler(0) = %v, want ErrUnsupported", err)
	}
	if _, err := ComposeByMode(mustParse(t, "8:1"), st); !errors.Is(err, ErrUnsupported) {
		t.Errorf("ComposeByMode with a tiler longer than the rank = %v, want ErrUnsupported", err)
	}

	// Associativity: (a∘b)∘c == a∘(b∘c) as functions, where both exist.
	a, b, c := mustParse(t, "(4,4):(4,1)"), mustParse(t, "(2,8):(8,1)"), mustParse(t, "(2,4):(4,1)")
	ab, err1 := Composition(a, b)
	bc, err2 := Composition(b, c)
	if err1 != nil || err2 != nil {
		t.Fatal(err1, err2)
	}
	left, err1 := Composition(ab, c)
	right, err2 := Composition(a, bc)
	if err1 != nil || err2 != nil {
		t.Fatal(err1, err2)
	}
	if !slices.Equal(offsets(t, left), offsets(t, right)) {
		t.Errorf("(a∘b)∘c = %v, a∘(b∘c) = %v: not associative", left, right)
	}
}

// checkComposes asserts r computes a(b(i)) for every i and keeps b's rank.
func checkComposes(t *testing.T, a, b, r Layout) {
	t.Helper()
	if r.Size() != b.Size() || b.shape.tuple && r.Rank() != b.Rank() {
		t.Errorf("%v ∘ %v = %v: size %d rank %d, want b's %d, %d", a, b, r, r.Size(), r.Rank(), b.Size(), b.Rank())
		return
	}
	for i := range b.Size() {
		j, err := b.Index(i)
		if err != nil {
			t.Fatal(err)
		}
		want, got := a.eval(t, j), r.eval(t, i)
		if !slices.Equal(got, want) {
			t.Errorf("%v ∘ %v = %v: at %d got %v, want a(%d) = %v", a, b, r, i, got, j, want)
			return
		}
	}
}

// eval evaluates l at an integral coordinate, extending the last mode past
// Size as CuTe does, and returns a coordinate-valued offset.
func (l Layout) eval(t *testing.T, i int) []int {
	out := make([]int, 2)
	for k, f := range l.leaves {
		c := i % f.n
		if k == len(l.leaves)-1 {
			c = i
		}
		i /= f.n
		switch f.kind {
		case strideBasis:
			out[f.axis] += c * f.d
		case strideF2:
			out[0] ^= f2Eval(f.bits, c)
		default:
			out[0] += c * f.d
		}
	}
	return out
}

func TestCompositionIncompatibleDivisibilityFalsifier(t *testing.T) {
	tests := []struct {
		name, a, b string
		accepted   string // what an implementation missing the check returns
		by         string
	}{
		// CuTe's composition_impl checks rest_stride % curr_shape == 0 ||
		// rest_stride < curr_shape, which accepts stride 4 against extent 6.
		{"stride", "(6,2):(1,7)", "4:4", "(2,2):(4,7)", "CuTe C++ stride condition"},
		// pycute checks no shape divisibility and truncates 6 to 4.
		{"shape", "(4,3):(1,8)", "6:1", "4:1", "pycute (no shape condition)"},
	}
	for _, tt := range tests {
		a, b := mustParse(t, tt.a), mustParse(t, tt.b)
		if r, err := Composition(a, b); !errors.Is(err, ErrUnsupported) {
			t.Errorf("%s: Composition(%s, %s) = %v, %v, want ErrUnsupported", tt.name, tt.a, tt.b, r, err)
		} else if !strings.Contains(err.Error(), tt.name+" divisibility") {
			t.Errorf("%s: error %q does not name the %s condition", tt.name, err, tt.name)
		}

		// The rejection is not caution. The answer the weaker check
		// returns is wrong, and the brute force says where.
		wrong := mustParse(t, tt.accepted)
		agrees := wrong.Size() == b.Size()
		for i := 0; agrees && i < b.Size(); i++ {
			j, _ := b.Index(i)
			w, _ := wrong.Index(i)
			agrees = w == a.eval(t, j)[0]
		}
		if agrees {
			t.Errorf("%s: %s's answer %s is correct; the check rejects a valid composition", tt.name, tt.by, tt.accepted)
		}
	}

	// Neighbours that meet both conditions compose.
	for _, pair := range [][2]string{{"(6,2):(1,7)", "3:2"}, {"(4,3):(1,8)", "8:1"}, {"(4,3):(1,8)", "4:1"}} {
		a, b := mustParse(t, pair[0]), mustParse(t, pair[1])
		r, err := Composition(a, b)
		if err != nil {
			t.Errorf("Composition(%s, %s) = %v, want success", pair[0], pair[1], err)
			continue
		}
		checkComposes(t, a, b, r)
	}

	for _, pair := range [][2]string{{"8:1", "4:-1"}, {"8:1", "4:1@0"}} {
		if _, err := Composition(mustParse(t, pair[0]), mustParse(t, pair[1])); !errors.Is(err, ErrUnsupported) {
			t.Errorf("Composition(%s, %s) = %v, want ErrUnsupported", pair[0], pair[1], err)
		}
	}
}

func TestLayoutComplementMinimalCodomain(t *testing.T) {
	tests := []struct {
		l        string
		cotarget int
		want     string
	}{
		{"4:1", 24, "6:4"},
		{"6:4", 24, "4:1"},
		{"(2,2):(1,6)", 24, "(3,2):(2,12)"},
		{"(4,6):(1,4)", 24, "1:0"},
		{"(2,4):(0,2)", 16, "(2,2):(1,8)"},
		{"4:2", 1, "2:1"},
		{"4:2", 20, "(2,3):(1,8)"},
	}
	for _, tt := range tests {
		l := mustParse(t, tt.l)
		r, err := Complement(l, tt.cotarget)
		if err != nil {
			t.Errorf("Complement(%s, %d): %v", tt.l, tt.cotarget, err)
			continue
		}
		if r.String() != tt.want {
			t.Errorf("Complement(%s, %d) = %v, want %s", tt.l, tt.cotarget, r, tt.want)
		}
		checkComplement(t, l, r, tt.cotarget)
	}

	// (2,3):(1,3) reaches {0,1,3,4,6,7}. No ordered layout fills {2,5}
	// without reaching past 8, so no exact complement exists.
	for _, s := range []string{"(2,3):(1,3)", "(2,2):(1,1)", "(4,2):(1,3)"} {
		if r, err := Complement(mustParse(t, s), 24); !errors.Is(err, ErrUnsupported) {
			t.Errorf("Complement(%s) = %v, %v, want ErrUnsupported", s, r, err)
		}
	}
	for _, s := range []string{"4:-1", "4:1@0"} {
		if _, err := Complement(mustParse(t, s), 8); !errors.Is(err, ErrUnsupported) {
			t.Errorf("Complement(%s) = %v, want ErrUnsupported", s, err)
		}
	}
	if _, err := Complement(mustParse(t, "4:1"), 0); !errors.Is(err, ErrUnsupported) {
		t.Errorf("Complement with cotarget 0 = %v, want ErrUnsupported", err)
	}
	// A cotarget near the int limit rounds up without wrapping.
	for _, n := range []int{2, 3} {
		r, err := Complement(mustParse(t, strconv.Itoa(n)+":1"), math.MaxInt)
		if want := math.MaxInt/n + 1; err != nil || r.Size() != want {
			t.Errorf("Complement(%d:1, MaxInt) = %v, %v, want size %d", n, r, err, want)
		}
	}
}

// checkComplement asserts that (l without its stride-0 modes, r) maps
// bijectively onto [0, n) for the smallest n that is a multiple of l's
// footprint and at least cotarget, and that r is ordered.
func checkComplement(t *testing.T, l, r Layout, cotarget int) {
	t.Helper()
	var f []leaf
	foot := 1
	for _, x := range l.leaves {
		if x.d != 0 && x.n != 1 {
			f = append(f, x)
			foot = max(foot, x.d*x.n)
		}
	}
	all := append(f, r.leaves...)
	cat := fromLeaves(all)
	got := offsets(t, cat)
	slices.Sort(got)
	n := (cotarget + foot - 1) / foot * foot
	if len(got) != n {
		t.Errorf("Complement(%v, %d) = %v: (l, r) has size %d, want the minimal %d", l, cotarget, r, len(got), n)
		return
	}
	for i, o := range got {
		if o != i {
			t.Errorf("Complement(%v, %d) = %v: (l, r) is not a bijection onto [0, %d)", l, cotarget, r, n)
			return
		}
	}
	for i := 1; i < len(r.leaves); i++ {
		if r.leaves[i].d <= r.leaves[i-1].d {
			t.Errorf("Complement(%v, %d) = %v: strides not increasing", l, cotarget, r)
		}
	}
}

func TestLogicalDivideTilePartitioning(t *testing.T) {
	d, err := LogicalDivide(mustParse(t, "(4,2,3):(2,1,8)"), mustParse(t, "4:2"))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := d.String(), "((2,2),(2,3)):((4,1),(2,8))"; got != want {
		t.Errorf("LogicalDivide = %s, want %s (CuTe documentation)", got, want)
	}

	// An 8x8 column-major tile divided into 4x2 blocks.
	a := mustParse(t, "(8,8):(1,8)")
	tiler, err := ShapeTiler(4, 2)
	if err != nil {
		t.Fatal(err)
	}
	byMode, err := LogicalDivideByMode(a, tiler)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := byMode.String(), "((4,2),(2,4)):((1,4),(8,16))"; got != want {
		t.Errorf("LogicalDivideByMode = %s, want %s", got, want)
	}
	z, err := ZippedDivide(a, tiler)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := z.String(), "((4,2),(2,4)):((1,8),(4,16))"; got != want {
		t.Errorf("ZippedDivide = %s, want %s", got, want)
	}

	// Tile k must hold exactly block (k%2, k/2): rows 4*(k%2).., columns
	// 2*(k/2)... Every element must be in exactly one tile.
	seen := make([]int, a.Size())
	for k := range 8 {
		br, bc := k%2, k/2
		var got, want []int
		for i := range 8 {
			o, err := z.Offset(i, k)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, o)
			seen[o]++
			w, _ := a.Offset(4*br+i%4, 2*bc+i/4)
			want = append(want, w)
		}
		if !slices.Equal(got, want) {
			t.Errorf("tile %d holds %v, want block (%d,%d) = %v", k, got, br, bc, want)
		}
	}
	for o, n := range seen {
		if n != 1 {
			t.Errorf("offset %d is in %d tiles, want 1", o, n)
		}
	}

	// Undivided modes go to the rest mode.
	z, err = ZippedDivide(mustParse(t, "(8,8,3):(1,8,64)"), tiler)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := z.String(), "((4,2),(2,4,3)):((1,8),(4,16,64))"; got != want {
		t.Errorf("ZippedDivide with a trailing mode = %s, want %s", got, want)
	}

	// A tile that does not divide the mode gets a partial last tile, not an
	// error: CuTe's complement rounds the tile count up.
	d, err = LogicalDivide(mustParse(t, "6:1"), mustParse(t, "4:1"))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := d.String(), "(4,2):(1,4)"; got != want {
		t.Errorf("LogicalDivide(6:1, 4:1) = %s, want %s", got, want)
	}
	if _, err := LogicalDivide(mustParse(t, "8:1"), mustParse(t, "(2,3):(1,3)")); !errors.Is(err, ErrUnsupported) {
		t.Errorf("LogicalDivide by a tile with no complement = %v, want ErrUnsupported", err)
	}
}

//go:generate sh -c "cd testdata && PYTHONPATH=$CUTLASS/python python3 pycute_golden.py > pycute.golden"

// TestPycuteGolden checks the algebra against testdata/pycute.golden, written
// by testdata/pycute_golden.py. pycute is an independent implementation (the
// one CUTLASS ships), and the generator checks each of its answers by brute
// force; answers that fail are recorded as "error".
func TestPycuteGolden(t *testing.T) {
	f, err := os.Open("testdata/pycute.golden")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	counts := map[string]int{}
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 1<<20)
	for line := 1; sc.Scan(); line++ {
		text := sc.Text()
		if strings.HasPrefix(text, "#") {
			continue
		}
		fields := strings.Split(text, "\t")
		op, want := fields[0], fields[len(fields)-1]
		l := mustParse(t, fields[1])
		var got Layout
		var err error
		switch op {
		case "index":
			var s []string
			for _, o := range offsets(t, l) {
				s = append(s, strconv.Itoa(o))
			}
			got, err = l, nil
			if g := strings.Join(s, ","); g != want {
				t.Errorf("line %d: %v offsets %s, want %s", line, l, g, want)
			}
			counts[op]++
			continue
		case "coalesce":
			got = Coalesce(l)
		case "coalescebymode":
			got = CoalesceByMode(l)
		case "compose":
			got, err = Composition(l, mustParse(t, fields[2]))
		case "complement":
			n, _ := strconv.Atoi(fields[2])
			got, err = Complement(l, n)
		case "divide":
			got, err = LogicalDivide(l, mustParse(t, fields[2]))
		case "zipped":
			var tiler []Layout
			for _, s := range strings.Split(fields[2], ";") {
				tiler = append(tiler, mustParse(t, s))
			}
			got, err = ZippedDivide(l, tiler)
		default:
			t.Fatalf("line %d: unknown op %q", line, op)
		}
		counts[op]++
		switch {
		case want == "error" && err == nil:
			t.Errorf("line %d: %s = %v, want error", line, text, got)
		case want == "error":
			counts[op+" rejected"]++
		case err != nil:
			t.Errorf("line %d: %s: %v", line, text, err)
		case got.String() != want:
			t.Errorf("line %d: %s = %v", line, text, got)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	// An empty or truncated file passes every line it has.
	for _, op := range []string{"index", "coalesce", "coalescebymode", "compose", "complement", "divide", "zipped", "compose rejected", "complement rejected", "divide rejected"} {
		if counts[op] < 40 {
			t.Errorf("%d %s cases, want at least 40", counts[op], op)
		}
	}
	t.Logf("cases: %v", counts)
}

func TestLayoutEvaluationDoesNotAllocate(t *testing.T) {
	l := mustParse(t, "((2,2),(2,3)):((1,2),(8,16))")
	f2, err := TMASwizzleLayout(8, 64, 2, TMASwizzle128B)
	if err != nil {
		t.Fatal(err)
	}
	var sink int
	allocs := testing.AllocsPerRun(100, func() {
		for i := range l.Size() {
			a, _ := l.Index(i)
			b, _ := l.Offset(i%4, i/4)
			c, _ := l.Offset(i%2, i/2%2, i/4%2, i/8)
			d, _ := f2.Offset(i%8, i%64)
			sink += a + b + c + d
		}
	})
	if allocs != 0 {
		t.Errorf("Index and Offset allocate %v times per run, want 0", allocs)
	}
	_ = sink
}

func TestLayoutStringRoundTrip(t *testing.T) {
	for _, s := range []string{"8:1", "(4,2):(1,4)", "((2,2),(2,3)):((1,2),(8,16))", "(4,(2,3)):(1@0,(4@0,1@1))", "(3,2):(-1,3)", "(4):(1)"} {
		l := mustParse(t, s)
		if l.String() != s {
			t.Errorf("ParseLayout(%q).String() = %q", s, l)
		}
		m := mustParse(t, l.String())
		if !l.Equal(m) {
			t.Errorf("round trip of %q gave %v", s, m)
		}
	}
	m, err := CompactLayout(TupleShape(IntShape(2), IntShape(3)))
	if err != nil || m.String() != "(2,3):(1,2)" {
		t.Errorf("CompactLayout((2,3)) = %v, %v, want (2,3):(1,2)", m, err)
	}
	if _, err := CompactLayout(TupleShape(IntShape(1<<40), IntShape(1<<40))); !errors.Is(err, ErrOverflow) {
		t.Errorf("CompactLayout overflow = %v, want ErrOverflow", err)
	}
}

func FuzzParseLayout(f *testing.F) {
	for _, s := range []string{"8:1", "((2,2),(2,3)):((1,2),(8,16))", "(4,(2,3)):(1@0,(4@0,1@1))", "(3,2):(-1,3)", "(2,3):(1,3)"} {
		f.Add(s, 16)
	}
	f.Fuzz(func(t *testing.T, s string, cotarget int) {
		l, err := ParseLayout(s)
		if err != nil {
			return
		}
		m, err := ParseLayout(l.String())
		if err != nil || !m.Equal(l) {
			t.Fatalf("%q: String %q does not round-trip: %v", s, l, err)
		}
		if l.Size() > 1<<12 || l.kind == strideBasis {
			return
		}
		c := Coalesce(l)
		for i := range l.Size() {
			x, err1 := l.Index(i)
			y, err2 := c.Index(i)
			if err1 != nil || err2 != nil || x != y {
				t.Fatalf("%v: Coalesce %v differs at %d: %d, %d (%v, %v)", l, c, i, x, y, err1, err2)
			}
		}
		r, err := Complement(l, cotarget)
		if err == nil && cotarget <= 1<<12 && r.Size() <= 1<<12 {
			checkComplement(t, l, r, cotarget)
		}
	})
}

func TestLayoutRightInverse(t *testing.T) {
	tests := []struct {
		layout string
		want   string
	}{
		{"8:1", "8:1"},
		{"(4,2):(1,4)", "8:1"},
		{"(4,2):(2,8)", "1:0"},
		{"(4,2):(1,8)", "4:1"},
		{"((2,2),2):((4,1),2)", "(4,2):(2,1)"},
	}
	if len(tests) < 5 {
		t.Fatalf("expected at least 5 test cases, got %d", len(tests))
	}
	for _, tt := range tests {
		l := mustParse(t, tt.layout)
		inv := RightInverse(l)
		if inv.String() != tt.want {
			t.Errorf("RightInverse(%s) = %v, want %s", tt.layout, inv, tt.want)
		}
		// Invariant: for all i in [0, inv.Size()), l(inv(i)) == i
		for i := 0; i < inv.Size(); i++ {
			invOff, err := inv.Index(i)
			if err != nil {
				t.Fatalf("%v.Index(%d): %v", inv, i, err)
			}
			off, err := l.Index(invOff)
			if err != nil {
				t.Fatalf("%v.Index(%d): %v", l, invOff, err)
			}
			if off != i {
				t.Errorf("%s∘RightInverse at %d: got %d, want %d", tt.layout, i, off, i)
			}
		}
	}
}

func TestLayoutNullspace(t *testing.T) {
	tests := []struct {
		layout string
		want   string
	}{
		{"(4,2):(1,0)", "2:4"},
		{"(4,2):(1,4)", "1:0"},
		{"(4,2,3):(1,0,0)", "6:4"},
		{"(2,(2,2)):(0,(1,10))", "2:1"},
	}
	if len(tests) < 4 {
		t.Fatalf("expected at least 4 test cases, got %d", len(tests))
	}
	for _, tt := range tests {
		l := mustParse(t, tt.layout)
		null := Nullspace(l)
		if null.String() != tt.want {
			t.Errorf("Nullspace(%s) = %v, want %s", tt.layout, null, tt.want)
		}
		// Invariant: for all i in [0, null.Size()), l(null(i)) == 0
		for i := 0; i < null.Size(); i++ {
			nullOff, err := null.Index(i)
			if err != nil {
				t.Fatalf("%v.Index(%d): %v", null, i, err)
			}
			off, err := l.Index(nullOff)
			if err != nil {
				t.Fatalf("%v.Index(%d): %v", l, nullOff, err)
			}
			if off != 0 {
				t.Errorf("%s∘Nullspace at %d: got %d, want 0", tt.layout, i, off)
			}
		}
	}
}

func TestLayoutGreatestCommonDomain(t *testing.T) {
	tests := []struct {
		a    string
		b    string
		want string
	}{
		{"(8,4):(1,8)", "(8,4):(1,8)", "32:1"},
		{"(4,4):(1,8)", "(4,4):(1,4)", "4:1"},
		{"4:2", "4:1", "1:0"},
		{"(4,4):(1,8)", "16:1", "4:1"},
	}
	if len(tests) < 4 {
		t.Fatalf("expected at least 4 test cases, got %d", len(tests))
	}
	for _, tt := range tests {
		a := mustParse(t, tt.a)
		b := mustParse(t, tt.b)
		gcd := GreatestCommonDomain(a, b)
		if gcd.String() != tt.want {
			t.Errorf("GreatestCommonDomain(%s, %s) = %v, want %s", tt.a, tt.b, gcd, tt.want)
		}
		// Invariant: for all i in [0, gcd.Size()), a(gcd(i)) == i and b(gcd(i)) == i
		for i := 0; i < gcd.Size(); i++ {
			gcdOff, err := gcd.Index(i)
			if err != nil {
				t.Fatalf("%v.Index(%d): %v", gcd, i, err)
			}
			aOff, err := a.Index(gcdOff)
			if err != nil {
				t.Fatalf("%v.Index(%d): %v", a, gcdOff, err)
			}
			bOff, err := b.Index(gcdOff)
			if err != nil {
				t.Fatalf("%v.Index(%d): %v", b, gcdOff, err)
			}
			if aOff != i || bOff != i {
				t.Errorf("GCD(%s, %s) at %d: a=%d, b=%d, want %d", tt.a, tt.b, i, aOff, bOff, i)
			}
		}
	}
}
