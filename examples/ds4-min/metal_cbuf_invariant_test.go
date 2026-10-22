//go:build darwin && arm64

package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// The command-buffer accounting rests on a source-level invariant: every Metal
// command buffer the inference runtime creates is created by
// (*metalPinnedOps).commandBuffer and recorded by (*metalPinnedOps).record.
// Before centralization the package held 28 creation sites and 2 recorders, so
// cbufs/tok read 86.00 against a gputrace count of 87.0 -- a metric with zero
// variance and a missing term. A count is only worth its narrow error bars if
// nothing escapes it.
//
// This is a source test, not a runtime one, and deliberately so: it costs no
// GPU, no model load and no quiet host, so it can gate every edit. The runtime
// half of the gate is BenchmarkMetalSpanProbe agreeing with gputrace.
func TestCommandBufferCreationIsCentralized(t *testing.T) {
	const (
		create = "Queue().CommandBuffer()"
		helper = "metal_ops_darwin.go"
		// metalTinyBackend is a separate toy backend, not part of the
		// inference runtime whose buffers cbufs/tok counts.
		tiny = "metal_tiny_darwin.go"
	)

	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var offenders []string
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") || f == helper || f == tiny {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(src), "\n") {
			if strings.Contains(line, create) {
				offenders = append(offenders, fileLine(f, i+1, line))
			}
		}
	}
	if len(offenders) > 0 {
		t.Errorf("command buffers created outside %s:\n\t%s\n"+
			"route them through (*metalPinnedOps).run so their spans are counted",
			helper, strings.Join(offenders, "\n\t"))
	}
}

// The recorder must stay single, for the same reason.
func TestRegionRecordingIsCentralized(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var sites []string
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(src), "\n") {
			if strings.Contains(line, "regions.add(") {
				sites = append(sites, fileLine(f, i+1, line))
			}
		}
	}
	if len(sites) != 1 {
		t.Errorf("got %d region recording sites, want exactly 1:\n\t%s",
			len(sites), strings.Join(sites, "\n\t"))
	}
}

// Stage 1 merged all 43 layer evaluations into a single Metal command buffer per
// token (dropping cbufs/tok from 44 to 2). metal_runtime_darwin.go may open a
// command buffer only at the sites inventoried below:
//  1. evalPosition: lazily opens the per-token command buffer, once per layer
//     at most and only when none is open.
//  2. evalLayer: opens a fallback closing command buffer when the router falls
//     off the fast path into the deep regime.
//  3. evalLayer: with SSD expert streaming only, opens the routed-FFN command
//     buffer. The host must read the routed expert IDs to map those experts
//     before the FFN is encoded, so this boundary costs at most one buffer per
//     layer in streaming mode and none in the default schedule.
//
// Every site after the first must commit and wait on the buffer it replaces in
// the same block, so at most one batch is ever open. A site outside the
// inventory, or one whose guard changes, is an unmerged or eager trailing
// command buffer and breaks the budget.
func TestRuntimeBatchAllocationIsBounded(t *testing.T) {
	const target = "metal_runtime_darwin.go"
	type site struct{ fn, guard, loop string }
	want := []site{
		{"evalPosition", "batch == nil", "range r.layers"},
		{"evalLayer", "i >= 2 && !batchedCompressed", ""},
		{"evalLayer", "r.streaming != nil", ""},
	}
	f, err := parser.ParseFile(token.NewFileSet(), target, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var got []site
	var stack []ast.Node
	ast.Inspect(f, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		stack = append(stack, n)
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if id, ok := call.Fun.(*ast.Ident); !ok || id.Name != "newMetalBatch" {
			return true
		}
		var s site
		var guard *ast.IfStmt
		var block *ast.BlockStmt
		for i := len(stack) - 2; i >= 0; i-- {
			switch x := stack[i].(type) {
			case *ast.FuncDecl:
				if s.fn == "" {
					s.fn = x.Name.Name
				}
			case *ast.IfStmt:
				if guard == nil && stack[i+1] == x.Body {
					guard = x
					s.guard = types.ExprString(x.Cond)
				}
			case *ast.BlockStmt:
				if block == nil {
					block = x
				}
			case *ast.ForStmt:
				s.loop = strings.TrimPrefix(s.loop+"; for", "; ")
			case *ast.RangeStmt:
				s.loop = strings.TrimPrefix(s.loop+"; range "+types.ExprString(x.X), "; ")
			case *ast.FuncLit:
				if s.fn == "" {
					s.fn = "func literal"
				}
			}
		}
		got = append(got, s)
		if guard == nil || block == nil {
			t.Errorf("%s: unguarded newMetalBatch in %s", target, s.fn)
			return true
		}
		if bin, ok := guard.Cond.(*ast.BinaryExpr); ok && bin.Op == token.EQL && types.ExprString(bin.Y) == "nil" {
			return true // opens only when none is open
		}
		committed, waited := map[string]bool{}, map[string]bool{}
		for _, stmt := range block.List {
			if stmt.Pos() > call.Pos() {
				break
			}
			ast.Inspect(stmt, func(n ast.Node) bool {
				c, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				sel, ok := c.Fun.(*ast.SelectorExpr)
				if !ok {
					return true
				}
				switch sel.Sel.Name {
				case "Commit":
					committed[types.ExprString(sel.X)] = true
				case "Wait":
					waited[types.ExprString(sel.X)] = true
				}
				return true
			})
		}
		closed := false
		for name := range committed {
			closed = closed || waited[name]
		}
		if !closed {
			t.Errorf("%s: newMetalBatch under %q in %s opens a second batch without committing and waiting on the open one", target, s.guard, s.fn)
		}
		return true
	})
	if !slices.Equal(got, want) {
		t.Errorf("newMetalBatch sites in %s:\n\tgot  %+v\n\twant %+v\n"+
			"a site outside the inventory means an unmerged or eager trailing command buffer has been introduced",
			target, got, want)
	}
}

func fileLine(file string, line int, text string) string {
	return file + ":" + itoa(line) + ": " + strings.TrimSpace(text)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// Creating a command buffer and filing its span are two separate acts, and for
// most of this package's life only the first happened. The creation counter saw
// every buffer; the span log saw the two that went through a metalBatch Wait or
// rope-attention-mixed. The workload benchmarks require the two counts to agree
// -- a3c3ae7c's reportWorkload calls b.Fatalf when they differ -- so a 512-token
// prefill that created 44*512+4 buffers and recorded 43*512+1 spans made
// BenchmarkMetalPrefill and BenchmarkMetalDecode unrunnable, on every arm, with
// no number printed at all. The missing per-token span was the embedding fetch,
// which reaches the GPU through dispatch.
//
// The count agreeing is a property of the whole file, not of any one call, so
// no unit test can hold it: the way it breaks is that someone adds an
// eighteenth immediate op and commits and waits inline, exactly as the other
// seventeen did. This asserts the shape in the source instead. It needs no GPU
// and no model, which is the point -- the defect it guards was invisible for
// weeks precisely because seeing it required a quiet host and an 80 GiB load.
func TestEveryImmediateBufferFilesASpan(t *testing.T) {
	// metal_batch_darwin.go is exempt: newMetalBatch creates the buffer and
	// (*metalBatch).Wait files the span, so creation and recording are
	// deliberately in different functions there.
	files := []string{"metal_ops_darwin.go", "metal_head_darwin.go"}
	var offenders []string
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(src), "\n")
		start, name := 0, ""
		flush := func(end int) {
			if name == "" {
				return
			}
			// The factory itself names the method it defines; it hands the
			// buffer to a caller that records, and records nothing itself.
			if strings.Contains(name, ") commandBuffer(") {
				return
			}
			body := strings.Join(lines[start:end], "\n")
			if !strings.Contains(body, "commandBuffer()") {
				return
			}
			if strings.Contains(body, "submitAndWait(") || strings.Contains(body, ".record(") {
				return
			}
			offenders = append(offenders, fileLine(f, start+1, name))
		}
		for i, line := range lines {
			if strings.HasPrefix(line, "func ") {
				flush(i)
				start, name = i, line
			}
		}
		flush(len(lines))
	}
	if len(offenders) > 0 {
		t.Errorf("%d function(s) create a command buffer without filing its span:\n\t%s\n"+
			"commit and wait through (*metalPinnedOps).submitAndWait, which records, "+
			"rather than calling cb.Commit and libmetal.WaitCommandBuffer inline; "+
			"a buffer the counter sees and the log does not makes every workload "+
			"benchmark fail its span-count check",
			len(offenders), strings.Join(offenders, "\n\t"))
	}
}
