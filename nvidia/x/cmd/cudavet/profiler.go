package main

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/cfg"
)

var profilerExitAnalyzer = &analysis.Analyzer{
	Name: "profilerexit",
	Doc: `check that a deferred driver.ProfilerStop is not skipped by os.Exit

os.Exit and log.Fatal end the process without running deferred calls, so a
deferred ProfilerStop never flushes an attached profiler and a capture of the
program comes back empty. profilerexit reports an os.Exit or log.Fatal that a
function can reach after deferring ProfilerStop, directly or through a wrapper
such as dev.WithContext, with no explicit ProfilerStop on the way.

A program that never calls ProfilerStop is not reported: returning from main
skips the flush as surely as os.Exit does, so every such program would be.`,
	Run: runProfilerExit,
}

func runProfilerExit(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	// stops reports whether n refers to driver.ProfilerStop, called or
	// passed as a value, as in dev.WithContext(driver.ProfilerStop).
	stops := func(n ast.Node) bool {
		found := false
		ast.Inspect(n, func(n ast.Node) bool {
			if id, ok := n.(*ast.Ident); ok {
				if fn, ok := info.Uses[id].(*types.Func); ok && fn.Name() == "ProfilerStop" && inPkg(fn.Pkg(), "driver") {
					found = true
				}
			}
			return !found
		})
		return found
	}
	reported := map[token.Pos]bool{}
	// skip reports each exit reachable from node i of block b before an
	// explicit ProfilerStop.
	var skip func(b *cfg.Block, i int, seen map[*cfg.Block]bool)
	skip = func(b *cfg.Block, i int, seen map[*cfg.Block]bool) {
		for _, n := range b.Nodes[i:] {
			if _, ok := n.(*ast.DeferStmt); !ok && stops(n) {
				return
			}
			for _, c := range calls(n) {
				if exits(info, c) {
					if !reported[c.Pos()] {
						reported[c.Pos()] = true
						fn := callee(info, c)
						pass.Reportf(c.Pos(), "%s.%s skips the deferred driver.ProfilerStop and an attached profiler is never flushed; return instead, or call ProfilerStop first", fn.Pkg().Name(), fn.Name())
					}
					return
				}
			}
		}
		for _, s := range b.Succs {
			if !seen[s] {
				seen[s] = true
				skip(s, 0, seen)
			}
		}
	}
	eachFunc(pass, func(body *ast.BlockStmt) {
		g := cfg.New(body, func(call *ast.CallExpr) bool { return !noReturn(info, call) })
		for _, b := range g.Blocks {
			for i, n := range b.Nodes {
				if d, ok := n.(*ast.DeferStmt); ok && stops(d) {
					skip(b, i+1, map[*cfg.Block]bool{})
				}
			}
		}
	})
	return nil, nil
}
