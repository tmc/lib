package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"strconv"
)

func (g *gen) launchHelpers(t string) {
	g.p("func %sRequired(name string, actual, required uint64, equal bool) error {", t)
	g.p("if actual < required || equal && actual != required { return fmt.Errorf(\"%%s: actual %%d; this launch requires %%d\", name, actual, required) }; return nil")
	g.p("}")
	g.p("// SetLaunchChecks enables or explicitly disables generated preconditions. Checks are enabled by default.")
	g.p("func (m *%s) SetLaunchChecks(enabled bool) error { if m == nil { return fmt.Errorf(\"nil module\") }; m.unchecked.Store(!enabled); return nil }", t)
	g.p("func %sCount(name string, n int64) (uint64, error) {", t)
	g.p("if n < 0 { return 0, fmt.Errorf(\"%%s has count %%d; requires at least 0\", name, n) }; return uint64(n), nil")
	g.p("}")
	g.p("func %sArithmetic(name string, a, b uint64, multiply bool) (uint64, error) {", t)
	g.p("if multiply { if b != 0 && a > ^uint64(0)/b { return 0, fmt.Errorf(\"%%s count product overflows: %%d * %%d\", name, a, b) }; return a*b, nil }")
	g.p("if a > ^uint64(0)-b { return 0, fmt.Errorf(\"%%s count sum overflows: %%d + %%d\", name, a, b) }; return a+b, nil")
	g.p("}")
	g.p("func %sExtent(name string, count int, pointer driver.DevicePtr, alignment int) error {", t)
	g.p("if count < 1 { return fmt.Errorf(\"%%s has %%d elements; this launch requires at least 1\", name, count) }")
	g.p("if pointer == 0 { return fmt.Errorf(\"%%s has pointer 0; %%d elements require nonzero storage\", name, count) }")
	g.p("return cudakernel.CheckPointerAlignment(name, pointer, alignment)")
	g.p("}")
	g.p("func %sGeometry(grid, block [3]uint32, shared uint32, deviceAttribute func(driver.DeviceAttribute) (int, error), kernelAttribute func(driver.FunctionAttribute) (int, error)) error {", t)
	g.p("attrs := [...]driver.DeviceAttribute{driver.DevAttrMaxBlockDimX, driver.DevAttrMaxBlockDimY, driver.DevAttrMaxBlockDimZ}")
	g.p("var threads uint64 = 1")
	g.p("for i, d := range block {")
	g.p("if grid[i] == 0 || d == 0 { return fmt.Errorf(\"grid/block dimension %%d has %%d/%%d; requires at least 1/1\", i, grid[i], d) }")
	g.p("limit, err := deviceAttribute(attrs[i]); if err != nil { return fmt.Errorf(\"block limit: %%w\", err) }")
	g.p("if limit < 1 || uint64(d) > uint64(limit) { return fmt.Errorf(\"block dimension %%d has %%d; maximum %%d\", i, d, limit) }")
	g.p("threads, err = %sArithmetic(\"block\", threads, uint64(d), true); if err != nil { return err }", t)
	g.p("}")
	g.p("for _, limitSource := range []func() (int, error){func() (int,error) {return deviceAttribute(driver.DevAttrMaxThreadsPerBlock)}, func() (int,error) {return kernelAttribute(driver.FuncAttributeMaxThreadsPerBlock)}} {")
	g.p("limit, err := limitSource(); if err != nil { return fmt.Errorf(\"thread limit: %%w\", err) }; if limit < 1 || threads > uint64(limit) { return fmt.Errorf(\"block has %%d threads; maximum %%d\", threads, limit) }")
	g.p("}")
	g.p("limit, err := deviceAttribute(driver.DevAttrMaxSharedMemoryPerBlockOptin); if err != nil { return fmt.Errorf(\"shared memory limit: %%w\", err) }")
	g.p("static, err := kernelAttribute(driver.FuncAttributeSharedSizeBytes); if err != nil { return fmt.Errorf(\"static shared memory: %%w\", err) }")
	g.p("if limit < 0 || static < 0 || uint64(shared)+uint64(static) > uint64(limit) { return fmt.Errorf(\"sharedBytes has %%d dynamic and %%d static bytes; maximum %%d\", shared, static, limit) }")
	g.p("dynamic, err := kernelAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes); if err != nil { return fmt.Errorf(\"dynamic shared memory limit: %%w\", err) }")
	g.p("if dynamic < 0 || uint64(shared) > uint64(dynamic) { return fmt.Errorf(\"sharedBytes has %%d bytes; configured maximum %%d\", shared, dynamic) }; return nil")
	g.p("}")
}

func (g *gen) kernelChecks(t string, k *kernel, index int, shared string) {
	g.p("if m == nil { return fmt.Errorf(\"%s: nil module\") }", k.name)
	g.p("if !m.unchecked.Load() {")
	// Declared extents run first, yielding the caller's required count rather
	// than the generic nonempty-storage refusal.
	next := 0
	for _, c := range k.contracts {
		a := g.contractExpr(t, c.left, k.params, &next, shared)
		b := g.contractExpr(t, c.right, k.params, &next, shared)
		g.p("if err := %sRequired(%q, %s, %s, %v); err != nil { return err }", t, c.text, a, b, c.op == token.EQL)
	}
	for _, p := range k.params {
		if p.slice {
			g.p("if err := %sExtent(%q, %s.Len(), %s.Ptr(), %d); err != nil { return err }", t, p.cName, p.name, p.name, p.alignment)
		}
	}
	g.p("if m.dev == nil || m.kernels[%d] == nil { return fmt.Errorf(\"%s: closed module\") }", index, k.name)
	g.p("if err := %sGeometry(grid, block, %s, m.dev.Attribute, m.kernels[%d].Attribute); err != nil { return err }", t, shared, index)
	g.p("}")
}

func (g *gen) contractExpr(t string, expr ast.Expr, params []param, next *int, shared string) string {
	name := fmt.Sprintf("cudabindContract%d", *next)
	*next++
	for _, p := range params {
		if p.name == name {
			return g.contractExpr(t, expr, params, next, shared)
		}
	}
	switch e := expr.(type) {
	case *ast.ParenExpr:
		return g.contractExpr(t, e.X, params, next, shared)
	case *ast.BasicLit:
		n, _ := strconv.ParseUint(e.Value, 0, 64)
		g.p("%s := uint64(%d)", name, n)
	case *ast.Ident:
		if e.Name == "sharedBytes" {
			g.p("%s := uint64(%s)", name, shared)
			break
		}
		for _, p := range params {
			if p.cName == e.Name {
				if len(p.typ) > 0 && p.typ[0] == 'u' {
					g.p("%s := uint64(%s)", name, p.name)
				} else {
					g.p("%s, err := %sCount(%q, int64(%s)); if err != nil { return err }", name, t, p.cName, p.name)
				}
			}
		}
	case *ast.SelectorExpr:
		g.p("%s := uint64(%s[%d])", name, e.X.(*ast.Ident).Name, map[string]int{"x": 0, "y": 1, "z": 2}[e.Sel.Name])
	case *ast.CallExpr:
		for _, p := range params {
			if p.cName == e.Args[0].(*ast.Ident).Name {
				g.p("%s := uint64(%s.Len())", name, p.name)
			}
		}
	case *ast.BinaryExpr:
		a := g.contractExpr(t, e.X, params, next, shared)
		b := g.contractExpr(t, e.Y, params, next, shared)
		g.p("%s, err := %sArithmetic(%q, %s, %s, %v); if err != nil { return err }", name, t, "launch contract", a, b, e.Op == token.MUL)
	}
	return name
}
