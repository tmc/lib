package cudalayout

import (
	"errors"
	"fmt"
	"testing"
)

// Oracles were produced by Python 3.12 calling right_inverse, nullspace and
// greatest_common_domain on Layout._set(shape, stride) in NVlabs/CuTe commit
// 111253d17e2f0f8631f43999b43ac4afa5954b04. Python tuple spaces and trailing
// commas are omitted in Go notation; singleton tuples retain parentheses.
// The pinned reference requires Python 3.12: under Python 3.14 its
// prefix_product raises NameError. See testdata/common_domain_oracles.py for
// reproduction and testdata/common_domain_oracles.txt for the reference output.
func TestInversePyCuTe(t *testing.T) {
	for _, test := range []struct{ input, inverse, null string }{
		{"8:1", "8:1", "1:0"},
		{"(4,2):(2,1)", "(2,4):(4,1)", "1:0"},
		{"((2,2),2):((4,1),2)", "(4,2):(2,1)", "1:0"},
		{"(3,4):(0,1)", "4:3", "3:1"},
		{"(2,3,4):(0,1,0)", "3:2", "(2,4):(1,6)"},
		{"(2,3):(1,8)", "2:1", "1:0"},
		{"(2,3):(2,8)", "1:0", "1:0"},
		{"(2,2):(1,1)", "2:1", "1:0"},
		{"(1,4):(0,1)", "4:1", "1:1"},
		{"(2,3):(0,0)", "1:0", "(2,3):(1,2)"},
	} {
		t.Run(test.input, func(t *testing.T) {
			l := mustParse(t, test.input)
			r, err := RightInverse(l)
			if err != nil || r.String() != test.inverse {
				t.Fatalf("RightInverse = %v, %v; want %s", r, err, test.inverse)
			}
			n, err := Nullspace(l)
			if err != nil || n.String() != test.null {
				t.Fatalf("Nullspace = %v, %v; want %s", n, err, test.null)
			}
		})
	}
}

func TestGreatestCommonDomainPyCuTe(t *testing.T) {
	for _, test := range []struct{ a, b, want string }{
		{"(5,7):(7,1)", "(7,5):(5,1)", "(1):(0)"},
		{"(16,3):(1,16)", "(16,3):(3,1)", "(16,3):(1,16)"},
		{"(5,3,4):(1,5,15)", "(10,6):(1,10)", "(5,2):(1,30)"},
		{"(6,35):(1,6)", "(15,14):(1,15)", "(3,7):(1,30)"},
		{"((2,3),4):((1,2),6)", "(6,4):(1,6)", "(2,3,4):(1,2,6)"},
		{"(1,8):(0,1)", "(2,4):(1,2)", "(2,4):(1,2)"},
		{"8:2", "8:0", "(8):(1)"},
		{"(4,6,4):(1,4,24)", "(3,3):(1,3)", "(1):(0)"},
		{"(10,15,10):(1,10,150)", "((3,5),10):((1,3),15)", "(5):(30)"},
		{"(2,3,5):(1,2,6)", "(5,3,2):(1,5,15)", "(1):(0)"},
		{"(1,4,6,4):(1,1,4,24)", "((3,1),3):((1,3),3)", "(1):(0)"},
		{"(2,6):(1,2)", "(3,4):(1,3)", "(2):(6)"},
		{"(4,3):(1,4)", "(2,6):(1,2)", "(2,2,3):(1,2,4)"},
		{"(2,5,3):(1,2,10)", "(5,2,3):(1,5,10)", "(3):(10)"},
	} {
		t.Run(test.a+"/"+test.b, func(t *testing.T) {
			a, b := mustParse(t, test.a), mustParse(t, test.b)
			for _, pair := range [][2]Layout{{a, b}, {b, a}} {
				got, err := GreatestCommonDomain(pair[0], pair[1])
				if err != nil || got.String() != test.want {
					t.Fatalf("GreatestCommonDomain = %v, %v; want %s", got, err, test.want)
				}
			}
		})
	}
}

func TestInverseProperties(t *testing.T) {
	for x := 1; x <= 4; x++ {
		for y := 1; y <= 4; y++ {
			for dx := 0; dx <= 4; dx++ {
				for dy := 0; dy <= 4; dy++ {
					l := mustParse(t, fmt.Sprintf("(%d,%d):(%d,%d)", x, y, dx, dy))
					r, err := RightInverse(l)
					if err != nil {
						t.Fatal(err)
					}
					for i := 0; i < r.Size(); i++ {
						j, err := r.Index(i)
						if err != nil {
							t.Fatal(err)
						}
						got, err := l.Index(j)
						if err != nil || got != i {
							t.Fatalf("%v inverse %v at %d: %d, %v", l, r, i, got, err)
						}
					}
					n, err := Nullspace(l)
					if err != nil {
						t.Fatal(err)
					}
					seen := make(map[int]bool)
					for i := 0; i < n.Size(); i++ {
						j, err := n.Index(i)
						if err != nil {
							t.Fatal(err)
						}
						got, err := l.Index(j)
						if err != nil || got != 0 || seen[j] {
							t.Fatalf("%v nullspace %v at %d: %d, %v", l, n, i, got, err)
						}
						seen[j] = true
					}
					for i := 0; i < l.Size(); i++ {
						got, _ := l.Index(i)
						if (got == 0) != seen[i] {
							t.Fatalf("%v nullspace missed zero coordinate %d", l, i)
						}
					}
				}
			}
		}
	}
}

func TestCommonDomainDivideProperty(t *testing.T) {
	for x := 1; x <= 6; x++ {
		for y := 1; y <= 6; y++ {
			for z := 1; z <= 6; z++ {
				a := mustParse(t, fmt.Sprintf("(%d,%d):(%d,1)", x, y*z, y*z))
				b := mustParse(t, fmt.Sprintf("(%d,%d):(%d,1)", x*y, z, z))
				tile, err := GreatestCommonDomain(a, b)
				if err != nil {
					t.Fatal(err)
				}
				for _, l := range []Layout{a, b} {
					d, err := LogicalDivide(l, tile)
					if err != nil {
						t.Fatalf("divide %v by %v: %v", l, tile, err)
					}
					first, err := d.Mode(0)
					if err != nil || first.Size() != tile.Size() || d.Size() != l.Size() {
						t.Fatalf("%v split by %v = %v, %v", l, tile, d, err)
					}
					for i := 0; i < tile.Size(); i++ {
						j, _ := tile.Index(i)
						want, _ := l.Index(j)
						got, err := first.Index(i)
						if err != nil || got != want {
							t.Fatalf("first mode %v at %d = %d, %v; want %d", first, i, got, err, want)
						}
					}
				}
			}
		}
	}
}

func TestCommonDomainGeneratedShapePairs(t *testing.T) {
	var layouts []Layout
	for x := 1; x <= 3; x++ {
		layouts = append(layouts, mustParse(t, fmt.Sprintf("%d:1", x)))
		for y := 1; y <= 3; y++ {
			layouts = append(layouts, mustParse(t, fmt.Sprintf("(%d,%d):(1,%d)", x, y, x)))
			for z := 1; z <= 3; z++ {
				layouts = append(layouts,
					mustParse(t, fmt.Sprintf("(%d,%d,%d):(1,%d,%d)", x, y, z, x, x*y)),
					mustParse(t, fmt.Sprintf("((%d,%d),%d):((1,%d),%d)", x, y, z, x, x*y)))
			}
		}
	}
	// 66 layouts yield 4,356 ordered pairs with equal and unequal rank,
	// nested profiles, size-one modes and different total sizes.
	for _, a := range layouts {
		for _, b := range layouts {
			tile, err := GreatestCommonDomain(a, b)
			if err != nil {
				t.Fatal(err)
			}
			reverse, err := GreatestCommonDomain(b, a)
			if err != nil || reverse.String() != tile.String() {
				t.Fatalf("asymmetric %v/%v: %v/%v, %v", a, b, tile, reverse, err)
			}
			// Euclid here is an independent size oracle, not the production gcd.
			x, y := a.Size(), b.Size()
			for y != 0 {
				x, y = y, x%y
			}
			if x%tile.Size() != 0 {
				t.Fatalf("%v/%v common size %d does not divide gcd %d", a, b, tile.Size(), x)
			}
			if a.Shape().String() == b.Shape().String() && tile.Size() != a.Size() {
				t.Fatalf("identical shape %v lost common factors: %v", a.Shape(), tile)
			}
			for _, l := range []Layout{a, b} {
				d, err := LogicalDivide(l, tile)
				if err != nil {
					t.Fatalf("divide %v by %v: %v", l, tile, err)
				}
				first, err := d.Mode(0)
				if err != nil || first.Size() != tile.Size() || d.Size() != l.Size() {
					t.Fatalf("invalid split %v/%v: %v, %v", l, tile, d, err)
				}
				for i := 0; i < tile.Size(); i++ {
					j, e := tile.Index(i)
					if e != nil {
						t.Fatal(e)
					}
					want, e := l.Index(j)
					if e != nil {
						t.Fatal(e)
					}
					got, e := first.Index(i)
					if e != nil || got != want {
						t.Fatalf("split %v/%v coordinate %d: got %d want %d, %v", l, tile, i, got, want, e)
					}
				}
			}
		}
	}
}

func TestInverseInvalid(t *testing.T) {
	for _, op := range []func(Layout) (Layout, error){RightInverse, Nullspace} {
		if _, err := op(Layout{}); !errors.Is(err, ErrUninitialized) {
			t.Fatalf("zero layout: %v", err)
		}
		f2, err := NewLayout(IntShape(4), F2Stride(1, 2))
		if err != nil {
			t.Fatal(err)
		}
		for _, l := range []Layout{mustParse(t, "4:-1"), mustParse(t, "4:1@0"), f2} {
			if _, err := op(l); !errors.Is(err, ErrUnsupported) {
				t.Fatalf("unsupported %v: %v", l, err)
			}
		}
	}
	if _, err := GreatestCommonDomain(Layout{}, mustParse(t, "4:1")); !errors.Is(err, ErrUninitialized) {
		t.Fatalf("zero common domain: %v", err)
	}
}

func ExampleRightInverse() {
	l, _ := ParseLayout("(4,2):(2,1)")
	r, err := RightInverse(l)
	fmt.Println(r, err)
	// Output: (2,4):(4,1) <nil>
}

func ExampleNullspace() {
	l, _ := ParseLayout("(2,3,4):(0,1,0)")
	n, err := Nullspace(l)
	fmt.Println(n, err)
	// Output: (2,4):(1,6) <nil>
}

func ExampleGreatestCommonDomain() {
	a, _ := ParseLayout("(6,35):(1,6)")
	b, _ := ParseLayout("(15,14):(1,15)")
	t, err := GreatestCommonDomain(a, b)
	fmt.Println(t, err)
	// Output: (3,7):(1,30) <nil>
}
