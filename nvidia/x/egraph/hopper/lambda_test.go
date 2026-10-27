package hopper

import (
	"fmt"
	"testing"
)

func TestLambda(t *testing.T) {
	tests := []struct {
		src   string
		args  []float32
		want  float32
		cuda  string
		arity int
	}{
		{"$0", []float32{3}, 3, "($0)", 1},
		{"(add $0 1.5)", []float32{2}, 3.5, "(($0) + 1.5f)", 1},
		{"(mul $1 (sub $0 2))", []float32{5, 4}, 12, "(($1) * (($0) - 2.0f))", 2},
		{"(max $0 (neg $0))", []float32{-7}, 7, "fmaxf(($0), (-($0)))", 1},
		{"(sigmoid 0)", nil, 0.5, "eg_sigmoid(0.0f)", 0},
		{"(silu $0)", []float32{0}, 0, "eg_silu(($0))", 1},
		{"(bf16 $0)", []float32{1 + 1.0/256}, 1, "eg_bf16_to_f32(eg_f32_to_bf16(($0)))", 1},
		{"(e4m3 $0)", []float32{0.1}, 0.1015625, "eg_e4m3_to_f32(eg_f32x2_to_e4m3x2(($0), 0.0f) & 0xffu)", 1},
		{"(rsqrt (add $0 1e-5))", []float32{4 - 1e-5}, 0.5, "rsqrtf((($0) + 1e-05f))", 1},
	}
	for _, tt := range tests {
		l, err := ParseLambda(tt.src)
		if err != nil {
			t.Fatalf("ParseLambda(%q): %v", tt.src, err)
		}
		if got := l.Eval(tt.args...); got != tt.want {
			t.Errorf("%s(%v) = %g, want %g", tt.src, tt.args, got, tt.want)
		}
		args := []string{"$0", "$1"}
		got, err := l.CUDA(args...)
		if err != nil || got != tt.cuda {
			t.Errorf("%s: CUDA = %q, %v; want %q", tt.src, got, err, tt.cuda)
		}
		if l.Arity() != tt.arity {
			t.Errorf("%s: arity %d, want %d", tt.src, l.Arity(), tt.arity)
		}
	}
}

func TestLambdaErrors(t *testing.T) {
	for _, src := range []string{"", "(", "(add $0)", "(add $0 $1 $2)", "(foo $0)", "$x", "(add $0 1) 2", "inf", ")"} {
		if _, err := ParseLambda(src); err == nil {
			t.Errorf("ParseLambda(%q) succeeded", src)
		}
	}
}

func TestLambdaCompose(t *testing.T) {
	outer := MustLambda("(mul $0 $1)")
	got := outer.Compose(0, MustLambda("(exp $0)"))
	if got.String() != "(mul (exp $2) $1)" || got.Arity() != 3 {
		t.Errorf("Compose = %s arity %d", got, got.Arity())
	}
	if v := got.Eval(99, 2, 0); v != 2 {
		t.Errorf("Eval = %g, want 2", v)
	}
}

func ExampleLambda() {
	gate := MustLambda("(mul $0 (sigmoid $1))") // rms_norm_gate's output factor
	c, _ := gate.CUDA("acc", "g")
	fmt.Println(gate.Eval(2, 0))
	fmt.Println(c)
	// Output:
	// 1
	// ((acc) * eg_sigmoid((g)))
}
