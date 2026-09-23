package cudalayout

import (
	"errors"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaliveness"
)

func TestF2StrideSwizzleEvaluation(t *testing.T) {
	// Hand-evaluated: coordinate bit k contributes bits[k], combined by XOR.
	tests := []struct {
		shape  Shape
		stride Stride
		want   []int
	}{
		{IntShape(8), F2Stride(1, 2, 4), []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{IntShape(4), F2Stride(3, 1), []int{0, 3, 1, 2}},
		{IntShape(4), F2Stride(6, 6), []int{0, 6, 6, 0}},
		// The span of 6 and 5 is {0,6,5,3}: Cosize 7, not the 8 an OR of the
		// bits would give.
		{IntShape(4), F2Stride(6, 5), []int{0, 6, 5, 3}},
		{TupleShape(IntShape(2), IntShape(4)), TupleStride(F2Stride(16), F2Stride(1, 17)), []int{0, 16, 1, 17, 17, 1, 16, 0}},
		{TupleShape(IntShape(3), IntShape(2)), TupleStride(IntStride(0), F2Stride(5)), []int{0, 0, 0, 5, 5, 5}},
	}
	for _, tt := range tests {
		l, err := NewLayout(tt.shape, tt.stride)
		if err != nil {
			t.Fatal(err)
		}
		if got := offsets(t, l); !slices.Equal(got, tt.want) {
			t.Errorf("%v: offsets %v, want %v", l, got, tt.want)
		}
		if got, want := mustCosize(t, l), slices.Max(tt.want)+1; got != want {
			t.Errorf("%v: Cosize %d, want %d", l, got, want)
		}
	}

	// TMASwizzleXOR against CuTe's Swizzle<B,4,3>, transcribed separately
	// in tmaswizzle_test.go.
	for mode, bbits := range map[TMASwizzle]int{TMASwizzleNone: 0, TMASwizzle32B: 1, TMASwizzle64B: 2, TMASwizzle128B: 3} {
		s := TMASwizzleXOR(mode)
		for x := range 1 << 12 {
			if got, want := s.Apply(x), cuteSwizzleApply(x, bbits); got != want {
				t.Fatalf("%v.Apply(%d) = %d, CuTe gives %d", s, x, got, want)
			}
			if s.Apply(s.Apply(x)) != x {
				t.Fatalf("%v is not an involution at %d", s, x)
			}
		}
	}

	// A swizzle composed with a compact layout equals applying it to each
	// offset, for positive and negative shifts.
	for _, p := range [][3]int{{3, 4, 3}, {2, 0, 4}, {1, 5, -3}, {0, 0, 0}} {
		s, err := NewXORSwizzle(p[0], p[1], p[2])
		if err != nil {
			t.Fatal(err)
		}
		base := mustParse(t, "((4,8),(2,4)):((1,4),(32,64))")
		l, err := s.Compose(base)
		if err != nil {
			t.Fatal(err)
		}
		for i := range base.Size() {
			o, _ := base.Index(i)
			if got, _ := l.Index(i); got != s.Apply(o) {
				t.Fatalf("%v∘%v at %d = %d, want %d", s, base, i, got, s.Apply(o))
			}
		}
		// Composition of an F2 layout with an F2-exact integer layout.
		inner := mustParse(t, "(4,2):(2,1)")
		r, err := Composition(l, inner)
		if err != nil {
			t.Fatal(err)
		}
		for i := range inner.Size() {
			j, _ := inner.Index(i)
			want, _ := l.Index(j)
			if got, _ := r.Index(i); got != want {
				t.Fatalf("(%v)∘%v at %d = %d, want %d", l, inner, i, got, want)
			}
		}
	}

	for _, p := range [][3]int{{-1, 0, 3}, {3, -1, 3}, {3, 4, 2}, {3, 4, -2}, {3, 60, 3}} {
		if _, err := NewXORSwizzle(p[0], p[1], p[2]); err == nil {
			t.Errorf("NewXORSwizzle%v accepted", p)
		}
	}
	// An integer layout whose offsets carry has no F2 form.
	for _, s := range []string{"3:1", "(2,2):(1,1)", "(4,2):(1,3)", "4:-1", "4:1@0"} {
		if _, err := TMASwizzleXOR(TMASwizzle128B).Compose(mustParse(t, s)); !errors.Is(err, ErrUnsupported) {
			t.Errorf("Compose(%s) = %v, want ErrUnsupported", s, err)
		}
	}
	f2, err := TMASwizzleLayout(8, 64, 2, TMASwizzle128B)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{"3:1", "(2,2):(1,1)", "2:512"} {
		if _, err := Composition(f2, mustParse(t, s)); !errors.Is(err, ErrUnsupported) {
			t.Errorf("Composition(F2, %s) = %v, want ErrUnsupported", s, err)
		}
	}
}

func mustCosize(t *testing.T, l Layout) int {
	t.Helper()
	n, err := l.Cosize()
	if err != nil {
		t.Fatal(err)
	}
	return n
}

// carryEval evaluates an F2 layout with integer addition in place of XOR:
// the mutant a stride implementation that forgot its kind would be.
func carryEval(l Layout, i int) int {
	off := 0
	for _, f := range l.leaves {
		c := i % f.n
		i /= f.n
		for k, b := range f.bits {
			if c>>k&1 != 0 {
				off += b
			}
		}
	}
	return off
}

// TestF2SwizzleXORCarryInjectedMutant shows the agreement in
// TestF2SwizzleMatchesTMASwizzledTile can fail. Evaluating the same F2 strides
// with carries must disagree with SwizzledTile for every swizzled mode. If it
// agreed, the XOR would be untested: the swizzle bits would never overlap the
// linear bits they are XORed into.
func TestF2SwizzleXORCarryInjectedMutant(t *testing.T) {
	for _, mode := range []TMASwizzle{TMASwizzle32B, TMASwizzle64B, TMASwizzle128B} {
		l, err := TMASwizzleLayout(16, 128, 2, mode)
		if err != nil {
			t.Fatal(err)
		}
		tile, err := NewSwizzledTile(16, 128, 2, mode)
		if err != nil {
			t.Fatal(err)
		}
		diff := 0
		for row := range 16 {
			for col := range 128 {
				want, _ := tile.ByteOffset(row, col)
				i := row + 16*col
				if got, _ := l.Index(i); got != want {
					t.Fatalf("%s: real evaluator disagrees at (%d,%d): %d, want %d", mode, row, col, got, want)
				}
				if carryEval(l, i) != want {
					diff++
				}
			}
		}
		if diff == 0 {
			t.Errorf("%s: a carrying evaluator agrees with SwizzledTile everywhere", mode)
		}
		t.Logf("%s: carrying evaluator wrong at %d of %d coordinates", mode, diff, 16*128)
	}
	// Without a swizzle the images are disjoint, so the mutant is invisible
	// there, and must be.
	l, err := TMASwizzleLayout(16, 128, 2, TMASwizzleNone)
	if err != nil {
		t.Fatal(err)
	}
	for i := range l.Size() {
		if got, _ := l.Index(i); got != carryEval(l, i) {
			t.Fatalf("unswizzled: XOR and carry differ at %d", i)
		}
	}
}

// TestF2SwizzleMatchesTMASwizzledTile compares two independent descriptions
// of the tensor-map addressing: SwizzledTile's modulo-and-divide arithmetic,
// and an F2 layout built from a column grouping and CuTe's Swizzle<B,4,3>.
func TestF2SwizzleMatchesTMASwizzledTile(t *testing.T) {
	modes := []TMASwizzle{TMASwizzleNone, TMASwizzle32B, TMASwizzle64B, TMASwizzle128B}
	checked := map[TMASwizzle]int{}
	for _, mode := range modes {
		for _, rows := range []int{1, 8, 16, 64} {
			for _, eb := range []int{1, 2, 4, 8, 16} {
				for _, rowBytes := range []int{16, 32, 64, 128, 256, 512} {
					cols := rowBytes / eb
					if cols == 0 {
						continue
					}
					tile, err := NewSwizzledTile(rows, cols, eb, mode)
					if err != nil {
						continue
					}
					l, err := TMASwizzleLayout(rows, cols, eb, mode)
					if err != nil {
						t.Fatalf("TMASwizzleLayout(%d, %d, %d, %s): %v", rows, cols, eb, mode, err)
					}
					for row := range rows {
						for col := range cols {
							want, _ := tile.ByteOffset(row, col)
							got, err := l.Offset(row, col)
							if err != nil || got != want {
								t.Fatalf("%s %dx%dx%dB (%d,%d): F2 layout %v gives %d, %v; SwizzledTile gives %d",
									mode, rows, cols, eb, row, col, l, got, err, want)
							}
							checked[mode]++
						}
					}
					// Offsets are element starts, so the last element's
					// start is eb-1 bytes short of the tile's end.
					if got, want := mustCosize(t, l), tile.Bytes()-eb+1; got != want {
						t.Errorf("%s %dx%dx%dB: Cosize %d, want %d", mode, rows, cols, eb, got, want)
					}
				}
			}
		}
	}
	for _, mode := range modes {
		if checked[mode] < 1000 {
			t.Errorf("%s: only %d coordinates compared", mode, checked[mode])
		}
	}
	t.Logf("coordinates compared: %v", checked)

	// A tile SwizzledTile addresses but no F2 layout can: 12 rows do not
	// split into bits.
	if _, err := NewSwizzledTile(12, 64, 2, TMASwizzle128B); err != nil {
		t.Fatal(err)
	}
	if _, err := TMASwizzleLayout(12, 64, 2, TMASwizzle128B); !errors.Is(err, ErrUnsupported) {
		t.Errorf("TMASwizzleLayout with 12 rows = %v, want ErrUnsupported", err)
	}
	if _, err := TMASwizzleLayout(8, 8, 2, TMASwizzle128B); !errors.Is(err, ErrUnsupported) {
		t.Errorf("TMASwizzleLayout with a 16-byte row under 128B = %v, want ErrUnsupported", err)
	}
}

// TestF2SwizzleLogicalDivide divides a swizzled tile into 4x8 tiles of
// every other column and checks, through SwizzledTile, that each element of
// each tile is the byte the divide says it is. The row and column each
// divided index stands for comes from dividing the integer identity layouts
// the same way, which TestPycuteGolden checks against pycute.
func TestF2SwizzleLogicalDivide(t *testing.T) {
	const rows, cols, eb = 8, 64, 2
	for _, mode := range []TMASwizzle{TMASwizzle32B, TMASwizzle64B, TMASwizzle128B} {
		tile, err := NewSwizzledTile(rows, cols, eb, mode)
		if err != nil {
			t.Fatal(err)
		}
		l, err := TMASwizzleLayout(rows, cols, eb, mode)
		if err != nil {
			t.Fatal(err)
		}
		// Flatten the column grouping so the tiler meets one column mode.
		flat, err := NewLayout(TupleShape(IntShape(rows), IntShape(cols)), TupleStride(l.stride.modes[0], flatF2(t, l.stride.modes[1])))
		if err != nil {
			t.Fatal(err)
		}
		tiler := []Layout{mustParse(t, "4:1"), mustParse(t, "8:2")}
		d, err := LogicalDivideByMode(flat, tiler)
		if err != nil {
			t.Fatalf("%s: LogicalDivideByMode(%v): %v", mode, flat, err)
		}
		rowOf, err := LogicalDivide(mustParse(t, "8:1"), tiler[0])
		if err != nil {
			t.Fatal(err)
		}
		colOf, err := LogicalDivide(mustParse(t, "64:1"), tiler[1])
		if err != nil {
			t.Fatal(err)
		}
		n := 0
		for i := range rows {
			for j := range cols {
				row, _ := rowOf.Index(i)
				col, _ := colOf.Index(j)
				want, _ := tile.ByteOffset(row, col)
				got, err := d.Offset(i, j)
				if err != nil || got != want {
					t.Fatalf("%s: divided %v at (%d,%d) = %d, %v; element (%d,%d) is at %d", mode, d, i, j, got, err, row, col, want)
				}
				n++
			}
		}
		if n != rows*cols {
			t.Fatalf("%s: compared %d elements, want %d", mode, n, rows*cols)
		}
	}
}

// flatF2 concatenates the bit images of an F2 stride tree, lowest
// coordinate bit first, which is the F2 stride of the flattened mode.
func flatF2(t *testing.T, d Stride) Stride {
	t.Helper()
	var bits []int
	var walk func(Stride)
	walk = func(d Stride) {
		if d.tuple {
			for _, m := range d.modes {
				walk(m)
			}
			return
		}
		if d.kind != strideF2 {
			t.Fatalf("stride %v is not F2", d)
		}
		bits = append(bits, d.bits...)
	}
	walk(d)
	return F2Stride(bits...)
}

// bankConflicts returns the conflict degree of one warp's shared-memory
// access: the largest number of distinct 4-byte words that fall in one of
// the 32 banks. Two threads reading the same word are a broadcast, not a
// conflict. A degree of 1 is conflict-free.
func bankConflicts(addrs []int) int {
	var words [32][]int
	degree := 0
	for _, a := range addrs {
		bank := a / 4 % 32
		if !slices.Contains(words[bank], a/4) {
			words[bank] = append(words[bank], a/4)
		}
		degree = max(degree, len(words[bank]))
	}
	return degree
}

// warpAccess returns the byte addresses of one warp reading a 16-byte column
// chunk from each of 8 rows, as ldmatrix does for one 8x8 matrix of 2-byte
// elements: thread t reads 4-byte word t%4 of row t/4, at column chunk c.
func warpAccess(t *testing.T, l Layout, eb, c int) []int {
	t.Helper()
	var addrs []int
	for thread := range 32 {
		row, col := thread/4, (c*16+thread%4*4)/eb
		a, err := l.Offset(row, col)
		if err != nil {
			t.Fatal(err)
		}
		addrs = append(addrs, a)
	}
	return addrs
}

// TestSwizzleCompositionBankConflictFree counts bank conflicts from the
// addresses the F2 layouts compute. Each mode's swizzle must make the
// 8-row chunk access conflict-free, and the same grouping without the XOR
// must conflict with the degree the geometry predicts. The counter is itself
// qualified with controls so a counter that always reports 1 cannot pass.
func TestSwizzleCompositionBankConflictFree(t *testing.T) {
	const eb = 2
	for _, tt := range []struct {
		mode       TMASwizzle
		unswizzled int
		why        string
	}{
		{TMASwizzle32B, 2, "rows are 32 B apart, so row r and r+4 share banks: 8 rows / 4 = 2 words per bank"},
		{TMASwizzle64B, 4, "rows are 64 B apart, so rows r, r+2, r+4, r+6 share banks: 4 words per bank"},
		{TMASwizzle128B, 8, "rows are 128 B apart, a whole bank cycle, so all 8 rows share banks"},
	} {
		t.Run(tt.mode.String(), func(t *testing.T) {
			span := tt.mode.Bytes()
			cols := span / eb
			swizzled, err := TMASwizzleLayout(8, cols, eb, tt.mode)
			if err != nil {
				t.Fatal(err)
			}
			base, err := NewLayout(
				TupleShape(IntShape(8), TupleShape(IntShape(cols), IntShape(1))),
				TupleStride(IntStride(span), TupleStride(IntStride(eb), IntStride(8*span))))
			if err != nil {
				t.Fatal(err)
			}
			// An XOR that only moves row bits among themselves, above
			// the bank bits (2..6), must not change the count.
			above, err := NewXORSwizzle(1, 7, 2)
			if err != nil {
				t.Fatal(err)
			}
			for c := range span / 16 {
				got := bankConflicts(warpAccess(t, swizzled, eb, c))
				if got != 1 {
					t.Errorf("chunk %d: %d-way conflict under %s, want conflict-free", c, got, tt.mode)
				}
				planted := bankConflicts(warpAccess(t, base, eb, c))
				adj, err := above.Compose(swizzled)
				if err != nil {
					t.Fatal(err)
				}
				adjacent := bankConflicts(warpAccess(t, adj, eb, c))
				if err := cudaliveness.Assert(
					cudaliveness.Planted{Site: "unswizzled grouping", Reported: planted - 1, Ran: true},
					cudaliveness.Adjacent{Site: "XOR above the bank bits", Reported: adjacent - 1, Ran: true},
					cudaliveness.Counted{Site: "unswizzled conflict degree", Want: int64(tt.unswizzled), Got: int64(planted), Why: tt.why, Ran: true},
				); err != nil {
					t.Errorf("chunk %d: bank counter not live: %v", c, err)
				}
			}
		})
	}

	// A swizzle with the wrong shift is not conflict-free: Swizzle<3,4,4>
	// XORs bits 8..10 into 4..6, so row r's chunk moves by r>>1 and rows 2k
	// and 2k+1 still collide.
	wrong, err := NewXORSwizzle(3, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	base := mustParse(t, "(8,(64,1)):(128,(2,1024))")
	l, err := wrong.Compose(base)
	if err != nil {
		t.Fatal(err)
	}
	if got := bankConflicts(warpAccess(t, l, eb, 0)); got != 2 {
		t.Errorf("%v: %d-way conflict, want 2", wrong, got)
	}
}

func TestHardwareExecutionPending(t *testing.T) {
	for _, target := range []string{TargetSM100a, TargetSM121a} {
		for _, mode := range []TMASwizzle{TMASwizzle32B, TMASwizzle64B, TMASwizzle128B} {
			err := TMASwizzleXOR(mode).CheckDevice(target)
			if !errors.Is(err, cudaerrors.ErrHardwarePending) {
				t.Errorf("%s on %s: %v, want cudaerrors.ErrHardwarePending", mode, target, err)
			}
			// Pending is not a refusal, by either package's meaning.
			if errors.Is(err, cudaerrors.ErrUnsupported) || errors.Is(err, ErrUnsupported) {
				t.Errorf("%s on %s: %v reads as unsupported", mode, target, err)
			}
		}
	}
	// The pending answer is about the known targets, not every input.
	err := TMASwizzleXOR(TMASwizzle128B).CheckDevice("sm_90a")
	if !errors.Is(err, ErrUnsupported) || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Errorf("unknown target: %v, want ErrUnsupported and not pending", err)
	}
}
