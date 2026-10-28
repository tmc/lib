package discover

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// Expr returns p's result as an e-graph pattern S-expression. Input k is
// the pattern variable ?a, ?b, ...; shared results are repeated. Ops
// carry their instruction immediates in the name: shl_8, lop3_ca,
// prmt_6240; immediate operands are leaves imm_00ff00ff.
func (p Program) Expr() string {
	var expr func(s Operand) string
	expr = func(s Operand) string {
		if s.IsImm {
			return fmt.Sprintf("imm_%08x", s.Imm)
		}
		if s.Reg < p.NumInputs {
			return "?" + string(rune('a'+s.Reg))
		}
		in := p.Insts[s.Reg-p.NumInputs]
		var op string
		switch in.Op {
		case OpNot:
			op = "not"
		case OpAnd:
			op = "and"
		case OpOr:
			op = "or"
		case OpXor:
			op = "xor"
		case OpShl:
			op = fmt.Sprintf("shl_%d", in.Imm)
		case OpShr:
			op = fmt.Sprintf("shr_%d", in.Imm)
		case OpShf:
			op = fmt.Sprintf("shf_%d", in.Imm)
		case OpLop3:
			op = fmt.Sprintf("lop3_%02x", in.Imm)
		case OpPrmt:
			op = fmt.Sprintf("prmt_%04x", in.Imm)
		}
		args := []string{op}
		for k := range in.Op.arity() {
			args = append(args, expr(in.Src[k]))
		}
		return "(" + strings.Join(args, " ") + ")"
	}
	return expr(R(p.NumInputs + len(p.Insts) - 1))
}

// Generalize lifts a verified rewrite of baseline into found to a symbolic
// rule: concrete input registers become pattern variables ?a, ?b, ...
// shared by both sides.
func Generalize(baseline, found Program) (lhs, rhs string) {
	return baseline.Expr(), found.Expr()
}

// Rule returns idiom's generalized rule as an [egraph.Rewrite] named
// "discover/" + idiom.Name, after re-proving the idiom's program
// equivalent to its baseline.
func Rule(idiom DiscoveredKernelIdiom) (*egraph.Rewrite, error) {
	if err := ProveEquivalent(idiom.Baseline, idiom.Program); err != nil {
		return nil, fmt.Errorf("rule %s: %w", idiom.Name, err)
	}
	r, err := egraph.NewRewrite("discover/"+idiom.Name, idiom.RuleLHS, idiom.RuleRHS)
	if err != nil {
		return nil, fmt.Errorf("rule %s: %w", idiom.Name, err)
	}
	return r, nil
}
