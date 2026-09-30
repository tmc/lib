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
