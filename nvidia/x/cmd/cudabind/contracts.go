package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
	"strings"
)

// launchContract is a bounded, nonnegative integer comparison preserved in a
// kernel's libclang raw comment. It is not inferred from the function body.
type launchContract struct {
	text        string
	left, right ast.Expr
	op          token.Token
}

func contractLine(line string) (string, bool) {
	line = strings.TrimSpace(line)
	line = strings.TrimSpace(strings.TrimLeft(line, "/*"))
	if !strings.HasPrefix(line, "cudabind:") {
		return "", false
	}
	return strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(line, "cudabind:"), "*/")), true
}

func hasLaunchContract(raw string) bool {
	for line := range strings.Lines(raw) {
		if _, ok := contractLine(line); ok {
			return true
		}
	}
	return false
}

func parseLaunchContracts(raw string, params []param) ([]launchContract, error) {
	byName := make(map[string]param)
	for _, p := range params {
		byName[p.cName] = p
	}
	var out []launchContract
	for line := range strings.Lines(raw) {
		text, ok := contractLine(line)
		if !ok {
			continue
		}
		if len(text) > 1024 || len(out) >= 32 {
			return nil, fmt.Errorf("launch contract exceeds bounded syntax")
		}
		e, err := parser.ParseExpr(text)
		if err != nil {
			return nil, fmt.Errorf("launch contract %q: %w", text, err)
		}
		b, ok := e.(*ast.BinaryExpr)
		if !ok || b.Op != token.GEQ && b.Op != token.EQL {
			return nil, fmt.Errorf("launch contract %q: require >= or ==", text)
		}
		for _, expr := range []ast.Expr{b.X, b.Y} {
			if err := validateContractExpr(expr, byName, 0); err != nil {
				return nil, fmt.Errorf("launch contract %q: %w", text, err)
			}
		}
		out = append(out, launchContract{text, b.X, b.Y, b.Op})
	}
	return out, nil
}

func validateContractExpr(e ast.Expr, params map[string]param, depth int) error {
	if depth > 16 {
		return fmt.Errorf("expression exceeds depth limit")
	}
	switch e := e.(type) {
	case *ast.ParenExpr:
		return validateContractExpr(e.X, params, depth+1)
	case *ast.BasicLit:
		if e.Kind != token.INT {
			break
		}
		if _, err := strconv.ParseUint(e.Value, 0, 64); err != nil {
			return fmt.Errorf("invalid integer %s", e.Value)
		}
		return nil
	case *ast.Ident:
		if e.Name == "sharedBytes" {
			if _, ok := params[e.Name]; ok {
				return fmt.Errorf("parameter sharedBytes conflicts with the launch shared-memory count")
			}
			return nil
		}
		p, ok := params[e.Name]
		if !ok {
			return fmt.Errorf("missing parameter %s", e.Name)
		}
		if p.slice || !integerType(p.typ) {
			return fmt.Errorf("parameter %s is not an integer count", e.Name)
		}
		return nil
	case *ast.SelectorExpr:
		id, ok := e.X.(*ast.Ident)
		if ok && (id.Name == "grid" || id.Name == "block") && (e.Sel.Name == "x" || e.Sel.Name == "y" || e.Sel.Name == "z") {
			return nil
		}
	case *ast.CallExpr:
		id, ok := e.Fun.(*ast.Ident)
		if ok && id.Name == "len" && len(e.Args) == 1 {
			arg, ok := e.Args[0].(*ast.Ident)
			if !ok {
				break
			}
			p, ok := params[arg.Name]
			if !ok {
				return fmt.Errorf("missing parameter %s", arg.Name)
			}
			if !p.slice {
				return fmt.Errorf("parameter %s is not a slice", arg.Name)
			}
			return nil
		}
	case *ast.BinaryExpr:
		if e.Op != token.ADD && e.Op != token.MUL {
			break
		}
		if err := validateContractExpr(e.X, params, depth+1); err != nil {
			return err
		}
		return validateContractExpr(e.Y, params, depth+1)
	}
	return fmt.Errorf("unsupported expression")
}

func integerType(s string) bool {
	switch s {
	case "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64":
		return true
	}
	return false
}
