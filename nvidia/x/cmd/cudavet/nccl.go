package main

import (
	"go/ast"
	"go/token"
	"go/types"
	"regexp"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/cfg"
)

var groupBalanceAnalyzer = &analysis.Analyzer{
	Name: "groupbalance",
	Doc: `check that every nccl.GroupStart is closed on every path

Operations between GroupStart and GroupEnd are only queued; a group left open
on some path, such as an error return, holds them back and hangs the next
collective. groupbalance reports a GroupStart from which some path leaves the
function without calling GroupEnd or GroupSimulateEnd, which also ends the
group.`,
	Run: runGroupBalance,
}

func runGroupBalance(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	// isGroup reports whether a CFG node, a simple statement or a condition,
	// calls nccl.name. A deferred GroupEnd closes the group on every path.
	isGroup := func(n ast.Node, name string) bool {
		found := false
		inspectShallow(n, func(n ast.Node) bool {
			if c, ok := n.(*ast.CallExpr); ok && calleeIn(info, c, "nccl") == name {
				found = true
			}
			return !found
		})
		return found
	}
	for _, file := range srcFiles(pass) {
		ast.Inspect(file, func(n ast.Node) bool {
			var body *ast.BlockStmt
			switch n := n.(type) {
			case *ast.FuncDecl:
				body = n.Body
			case *ast.FuncLit:
				body = n.Body
			}
			if body == nil {
				return true
			}
			g := cfg.New(body, func(call *ast.CallExpr) bool { return !noReturn(info, call) })
			for _, b := range g.Blocks {
				for i, node := range b.Nodes {
					if !isGroup(node, "GroupStart") {
						continue
					}
					// A GroupStart that failed opened no group.
					seen := map[*cfg.Block]bool{}
					for _, s := range b.Succs {
						if failed(info, s, node) {
							seen[s] = true
						}
					}
					// A group opened under if cond is closed under the same
					// cond; take cond to hold for the rest of the path.
					guard := ""
					if stmt, ok := b.Stmt.(*ast.IfStmt); ok && b.Kind == cfg.KindIfThen {
						guard = render(pass.Fset, stmt.Cond)
					}
					p := &groupPath{info: info, fset: pass.Fset, isGroup: isGroup, guard: guard, seen: seen}
					if p.open(b, i+1) {
						pass.Reportf(node.Pos(), "GroupStart is not matched by GroupEnd on every path out of the function; an open group hangs the next collective")
					}
				}
			}
			return true
		})
	}
	return nil, nil
}

// failed reports whether b is the branch taken when the GroupStart in node
// returned an error: the body of an if err != nil on the error node assigns.
func failed(info *types.Info, b *cfg.Block, node ast.Node) bool {
	stmt, ok := b.Stmt.(*ast.IfStmt)
	if !ok || b.Kind != cfg.KindIfThen {
		return false
	}
	assign, ok := node.(*ast.AssignStmt)
	if !ok || len(assign.Lhs) == 0 {
		return false
	}
	cond, ok := ast.Unparen(stmt.Cond).(*ast.BinaryExpr)
	if !ok || cond.Op != token.NEQ || !isNil(info, cond.Y) {
		return false
	}
	v := objOf(info, cond.X)
	return v != nil && v == objOf(info, assign.Lhs[len(assign.Lhs)-1])
}

// A groupPath searches the paths from a GroupStart.
type groupPath struct {
	info    *types.Info
	fset    *token.FileSet
	isGroup func(ast.Node, string) bool
	guard   string // condition the GroupStart was under, or ""
	seen    map[*cfg.Block]bool
}

// open reports whether some path from node i of block b reaches the end of
// the function without a GroupEnd or GroupSimulateEnd.
func (p *groupPath) open(b *cfg.Block, i int) bool {
	for _, n := range b.Nodes[i:] {
		if p.isGroup(n, "GroupEnd") || p.isGroup(n, "GroupSimulateEnd") {
			return false
		}
	}
	if len(b.Succs) == 0 {
		return b.Live && !endsInPanic(p.info, b)
	}
	succs := b.Succs
	if len(succs) == 2 && len(b.Nodes) > 0 && p.guard != "" {
		if cond, ok := b.Nodes[len(b.Nodes)-1].(ast.Expr); ok && render(p.fset, cond) == p.guard {
			succs = succs[:1] // the guard still holds
		}
	}
	for _, s := range succs {
		if p.seen[s] {
			continue
		}
		p.seen[s] = true
		if p.open(s, 0) {
			return true
		}
	}
	return false
}

// endsInPanic reports whether a block without successors stops in a call
// that does not return, rather than at a return or the end of the function.
func endsInPanic(info *types.Info, b *cfg.Block) bool {
	if len(b.Nodes) == 0 {
		return false
	}
	stmt, ok := b.Nodes[len(b.Nodes)-1].(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := stmt.X.(*ast.CallExpr)
	return ok && noReturn(info, call)
}

var collectiveDivergeAnalyzer = &analysis.Analyzer{
	Name: "collectivediverge",
	Doc: `check that collectives are not called on only some ranks

Every rank of a communicator must call each collective. A collective under a
condition on the rank, such as if rank == 0 { AllReduce }, with no matching
call on the other branch, waits forever for the ranks that skipped it.
collectivediverge reports that shape; point-to-point Send and Recv are
expected to depend on the rank and are not reported.`,
	Run: runCollectiveDiverge,
}

// collectives names the nccl calls every rank must make.
var collectives = map[string]bool{
	"AllReduce": true, "AllGather": true, "Broadcast": true, "Bcast": true,
	"Reduce": true, "ReduceScatter": true, "AlltoAll": true, "AllToAll": true,
}

var rankRE = regexp.MustCompile(`(?i)rank`)

func runCollectiveDiverge(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	collectiveIn := func(n ast.Node) map[string]*ast.CallExpr {
		out := make(map[string]*ast.CallExpr)
		if n == nil {
			return out
		}
		for _, c := range calls(n) {
			if name := calleeIn(info, c, "nccl"); collectives[name] {
				if _, ok := out[name]; !ok {
					out[name] = c
				}
			}
		}
		return out
	}
	eachFunc(pass, func(body *ast.BlockStmt) {
		inspectShallow(body, func(n ast.Node) bool {
			stmt, ok := n.(*ast.IfStmt)
			if !ok || !rankDependent(info, stmt.Cond) {
				return true
			}
			then, els := collectiveIn(stmt.Body), collectiveIn(stmt.Else)
			for name, c := range then {
				if _, ok := els[name]; !ok {
					pass.Reportf(c.Pos(), "%s is called only when %s; the ranks that skip it leave the others waiting", name, types.ExprString(stmt.Cond))
				}
			}
			for name, c := range els {
				if _, ok := then[name]; !ok {
					pass.Reportf(c.Pos(), "%s is called only when !(%s); the ranks that skip it leave the others waiting", name, types.ExprString(stmt.Cond))
				}
			}
			return true
		})
	})
	return nil, nil
}

// rankDependent reports whether cond mentions a rank: a variable or field
// whose name contains rank, or a call such as Comm.UserRank.
func rankDependent(info *types.Info, cond ast.Expr) bool {
	found := false
	ast.Inspect(cond, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && rankRE.MatchString(id.Name) {
			if _, isVar := info.Uses[id].(*types.Var); isVar {
				found = true
			}
			if fn, ok := info.Uses[id].(*types.Func); ok && inPkg(fn.Pkg(), "nccl") {
				found = true
			}
		}
		return !found
	})
	return found
}
