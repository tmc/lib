package main

import (
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
)

var halfBitsAnalyzer = &analysis.Analyzer{
	Name: "halfbits",
	Doc: `check that Float16BitsScalar is given bits, not a number

Float16BitsScalar takes the bit pattern of an IEEE 754 half-precision value.
Float16BitsScalar(uint16(x)) with x a float converts the value instead, so
1.0 becomes the bits 0x0001, about 6e-8, and nothing fails. halfbits reports
an argument that is a conversion to an integer from a float, or from an
untyped float constant such as 1.0, directly or through a variable assigned
once. A conversion of the result of a function whose name mentions bits or
half, which presumably computes the pattern, is not reported.`,
	Run: runHalfBits,
}

func runHalfBits(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		defs := assignments(info, body)
		for _, call := range calls(body) {
			fn := callee(info, call)
			if fn == nil || !isBinding(fn.Pkg()) || fn.Name() != "Float16BitsScalar" || len(call.Args) != 1 {
				continue
			}
			arg := call.Args[0]
			e := arg
			if v := objOf(info, e); v != nil {
				if d := defs[v]; len(d) == 1 && d[0] != nil {
					e = d[0]
				}
			}
			if x := floatConverted(info, e); x != nil {
				pass.Reportf(arg.Pos(), "%s converts the float %s to an integer; Float16BitsScalar takes the fp16 bit pattern",
					render(pass.Fset, arg), render(pass.Fset, x))
			}
		}
	})
	return nil, nil
}

// floatConverted returns the float operand of a chain of conversions e, or
// nil if no operand in the chain is a float or the innermost one is a call of
// a function named for bits or half.
func floatConverted(info *types.Info, e ast.Expr) ast.Expr {
	var float ast.Expr
	for {
		e = ast.Unparen(e)
		call, ok := e.(*ast.CallExpr)
		if !ok {
			break
		}
		if tv, ok := info.Types[call.Fun]; !ok || !tv.IsType() || len(call.Args) != 1 {
			if fn := callee(info, call); fn != nil {
				name := strings.ToLower(fn.Name())
				if strings.Contains(name, "bits") || strings.Contains(name, "half") {
					return nil
				}
			}
			break
		}
		x := call.Args[0]
		if float == nil && isFloat(info, x) {
			float = x
		}
		e = x
	}
	return float
}

// isFloat reports whether e has a float type or is an untyped float
// constant.
func isFloat(info *types.Info, e ast.Expr) bool {
	tv, ok := info.Types[ast.Unparen(e)]
	if !ok {
		return false
	}
	if b, ok := tv.Type.Underlying().(*types.Basic); ok && b.Info()&types.IsFloat != 0 {
		return true
	}
	if tv.Value == nil {
		return false
	}
	// An untyped constant operand of a conversion is recorded with the type
	// it is converted to, so 1.0 in uint16(1.0) reads as a uint16. Its
	// literals and named constants say what it was.
	switch e := ast.Unparen(e).(type) {
	case *ast.BasicLit:
		return e.Kind == token.FLOAT
	case *ast.Ident, *ast.SelectorExpr:
		id, ok := e.(*ast.Ident)
		if !ok {
			id = e.(*ast.SelectorExpr).Sel
		}
		c, ok := info.Uses[id].(*types.Const)
		if !ok {
			return false
		}
		b, ok := c.Type().(*types.Basic)
		return ok && b.Info()&types.IsFloat != 0
	case *ast.UnaryExpr:
		return isFloat(info, e.X)
	case *ast.BinaryExpr:
		return isFloat(info, e.X) || isFloat(info, e.Y)
	}
	return false
}
