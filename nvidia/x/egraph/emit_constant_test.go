package egraph

import (
	"errors"
	"testing"
)

func TestEmitCUDARefusesNumericPayloadOnOperator(t *testing.T) {
	for _, op := range []string{"unimplemented", "fused_swiglu_gemv", "add", "silu"} {
		t.Run(op, func(t *testing.T) {
			g := New()
			root := g.Add(ENode{Op: op, Data: float32(1)})
			expr, _, err := ExtractBest(g, root, AstSizeCost{})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := EmitCUDA(expr, EmitOptions{KernelName: "numeric_payload"}); !errors.Is(err, ErrEmissionUnsupported) {
				t.Fatalf("EmitCUDA(%q) = %v, want unsupported operator", op, err)
			}
			if _, err := (Source{Expression: expr}).ReferenceGraph(); !errors.Is(err, ErrEmissionUnsupported) {
				t.Fatalf("ReferenceGraph(%q) = %v, want unsupported operator", op, err)
			}
		})
	}
}

func TestExpressionConstantRequiresConstantOperator(t *testing.T) {
	for _, tt := range []struct {
		name string
		expr *Expr
		want float32
		ok   bool
	}{
		{"constant", &Expr{Op: "constant", Data: float32(1)}, 1, true},
		{"const", &Expr{Op: "const", Data: int32(2)}, 2, true},
		{"literal", &Expr{Op: "0.25", Data: float32(.25)}, .25, true},
		{"text", &Expr{Op: "0.25", Data: "0.25"}, .25, true},
		{"bare", &Expr{Op: "0.25"}, .25, true},
		{"inconsistent", &Expr{Op: "0.25", Data: float32(.5)}, 0, false},
		{"operand", &Expr{Op: "constant", Data: float32(1), Children: []*Expr{{Op: "1"}}}, 0, false},
		{"nil", nil, 0, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := expressionConstant(tt.expr)
			if got != tt.want || ok != tt.ok {
				t.Fatalf("expressionConstant = %v, %v; want %v, %v", got, ok, tt.want, tt.ok)
			}
		})
	}
}
