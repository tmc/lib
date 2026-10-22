package main

import (
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestCUDAQ4BatchShape(t *testing.T) {
	for _, tc := range []struct {
		name                        string
		cols, rows                  uint64
		pairs, slots, input, output int
		invalid                     bool
	}{
		{"paired", 4096, 2048, 12, 6, 2 * 4096 / 256 * 292, 12 * 2048, false},
		{"down", 2048, 4096, 12, 1, 12 * 2048 / 256 * 292, 12 * 4096, false},
		{"q8-1-input", 4096, 2048, 12, 6, 2 * 4096 / 32 * 36, 12 * 2048, true},
		{"partial-token", 4096, 2048, 11, 6, 2 * 4096 / 256 * 292, 11 * 2048, true},
		{"output-short", 4096, 2048, 12, 6, 2 * 4096 / 256 * 292, 12*2048 - 1, true},
		{"no-slots", 4096, 2048, 12, 0, 0, 0, true},
		{"negative-input", 4096, 2048, 12, 6, -1, 12 * 2048, true},
		{"zero-pairs", 4096, 2048, 0, 6, 0, 0, true},
		{"unaligned", 4097, 2048, 12, 6, 0, 0, true},
		{"grid-x-overflow", 256, math.MaxInt32 + 1, 6, 6, 292, 0, true},
		{"integer-overflow", math.MaxUint64, math.MaxUint64, math.MaxInt, 1, math.MaxInt, math.MaxInt, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := q4BatchShape(tc.cols, tc.rows, tc.pairs, tc.slots, tc.input, tc.output)
			if (err != nil) != tc.invalid {
				t.Fatalf("error=%v invalid=%v", err, tc.invalid)
			}
		})
	}
}

func TestCUDAQ4BatchCapacity(t *testing.T) {
	good := q4BatchCapacity{norm: 8192, input: 2 * 4096 / 256 * 292, gate: 12 * 2048, up: 12 * 2048, mid: 12 * 2048, quantMid: 12 * 2048 / 256 * 292, down: 12 * 4096, ids: 12, weights: 12, pairs: 12}
	if err := checkQ4BatchCapacity(2, good, false); err != nil {
		t.Fatal(err)
	}
	if err := checkQ4BatchCapacity(2, good, true); err == nil || !strings.Contains(err.Error(), "profile scratch") {
		t.Fatalf("missing collector buffer: %v", err)
	}
	good.profile = 12 * 2048
	if err := checkQ4BatchCapacity(2, good, true); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name     string
		truncate func(*q4BatchCapacity)
	}{
		{"norm", func(c *q4BatchCapacity) { c.norm-- }},
		{"input", func(c *q4BatchCapacity) { c.input-- }},
		{"gate", func(c *q4BatchCapacity) { c.gate-- }},
		{"up", func(c *q4BatchCapacity) { c.up-- }},
		{"mid", func(c *q4BatchCapacity) { c.mid-- }},
		{"quantized mid", func(c *q4BatchCapacity) { c.quantMid-- }},
		{"down", func(c *q4BatchCapacity) { c.down-- }},
		{"ids", func(c *q4BatchCapacity) { c.ids-- }},
		{"weights", func(c *q4BatchCapacity) { c.weights-- }},
		{"pairs", func(c *q4BatchCapacity) { c.pairs-- }},
		{"profile", func(c *q4BatchCapacity) { c.profile-- }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := good
			tc.truncate(&c)
			if err := checkQ4BatchCapacity(2, c, true); err == nil || !strings.Contains(err.Error(), tc.name+" scratch") {
				t.Fatalf("truncation: %v", err)
			}
		})
	}
	for _, n := range []int{0, -1, math.MaxInt} {
		if err := checkQ4BatchCapacity(n, good, true); err == nil {
			t.Fatalf("accepted token count %d", n)
		}
	}
}

// This compiles the actual CUDA fragments with a host thread/barrier shim.
// It tests byte layout, integer arithmetic and CSR addressing, not CUDA
// compilation, device barriers, launch geometry or model qualification.
func TestCUDAQ4BatchHostOracle(t *testing.T) {
	path := os.Getenv("DS4_Q4_ORACLE_DIR")
	if path == "" {
		t.Skip("DS4_Q4_ORACLE_DIR not set")
	}
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	pinned := func(file, hash string) string {
		s := read(filepath.Join(path, file))
		if fmt.Sprintf("%x", sha256.Sum256([]byte(s))) != hash {
			t.Fatalf("%s: unregistered source", file)
		}
		return s
	}
	extract := func(s, name string) string {
		t.Helper()
		i := strings.Index(s, name+"(")
		if i < 0 {
			t.Fatalf("missing function %s", name)
		}
		start := strings.LastIndex(s[:i], "\n") + 1
		brace := i + strings.Index(s[i:], "{")
		depth := 0
		for end := brace; end < len(s); end++ {
			switch s[end] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return s[start : end+1]
				}
			}
		}
		t.Fatalf("unterminated %s", name)
		return ""
	}
	cu := pinned("ds4_cuda.cu", "c420f92fecf07af2113bad5b9d74801c5e1d10020f1094120d9f31ba430fc0b5")
	c := pinned("ds4.c", "62e458f2c416148c0882f7247c7b16f9820054b0c2ae41374a9209541fe7a4c7")
	oracle := extract(c, "ds4_quantize_row_q8_K") + "\n" + extract(cu, "dev_q4_K_get_scale_min") + "\n" + extract(cu, "dev_dot_q4_32") + "\n" + extract(cu, "dev_dot_q4_K_q8_K_block")
	oracle = strings.Replace(oracle, "return y->d * xd", "oracle_total = isum; oracle_minimum = summs; return y->d * xd", 1)
	dot := extract(read("cuda_experts_q4.cu"), "ds4_q4_block_q8_k")
	dot = strings.Replace(dot, "float d = ds4_half_to_float", "host_total = total; host_minimum = minimum; float d = ds4_half_to_float", 1)
	production := extract(read("cuda_experts.cu"), "ds4_quant_q8_k") + "\n" + dot + "\n" + read("cuda_experts_q4_batch.cu")
	harness := read("testdata/q4_batch_host.cc")
	for _, cols := range []int{512, 2048, 4096} {
		for _, tc := range []struct{ name, old, new, failure string }{
			{name: "baseline"},
			{"token-address", "unsigned(pair) / slots", "unsigned(pair) % slots", "gate/up output"},
			{"output-order", "static_cast<unsigned long long>(pair) * rows + row", "static_cast<unsigned long long>(at) * rows + row", "gate/up output"},
			{"scale-high-half", "((scales[g - 4] >> 6) << 4)", "((scales[g] >> 6) << 4)", "integer partial"},
			{"minimum-sign", "d * float(total) - dm * float(minimum)", "d * float(total) + dm * float(minimum)", "dot output"},
			{"down-slots", "cols, rows, experts, 1, npairs", "cols, rows, experts, 6, npairs", "down output"},
			{"q8-k-sums", "sum += q->qs[tid * 16 + j];", "sum -= q->qs[tid * 16 + j];", "quantizer bytes"},
		} {
			t.Run(fmt.Sprintf("%d/%s", cols, tc.name), func(t *testing.T) {
				s := production
				if tc.old != "" {
					if strings.Count(s, tc.old) != 1 {
						t.Fatal("mutation site not unique")
					}
					s = strings.Replace(s, tc.old, tc.new, 1)
				}
				s = strings.Replace(strings.Replace(harness, "// ORACLE", oracle, 1), "// PRODUCTION", s, 1)
				dir := t.TempDir()
				source, binary := filepath.Join(dir, "control.cc"), filepath.Join(dir, "control")
				if err := os.WriteFile(source, []byte(s), 0600); err != nil {
					t.Fatal(err)
				}
				if out, err := exec.Command("c++", "-std=c++17", "-O0", "-ffp-contract=off", "-pthread", source, "-o", binary).CombinedOutput(); err != nil {
					t.Fatalf("compile: %v\n%s", err, out)
				}
				out, err := exec.Command(binary, fmt.Sprint(cols)).CombinedOutput()
				if tc.old == "" {
					if err != nil {
						t.Fatalf("baseline: %v\n%s", err, out)
					}
					t.Log(strings.TrimSpace(string(out)))
				} else if err == nil || !strings.Contains(string(out), tc.failure) {
					t.Fatalf("mutation must fail %q: %v\n%s", tc.failure, err, out)
				}
			})
		}
	}
}

func TestCUDAQ4BatchProductionContract(t *testing.T) {
	read := func(name string) string {
		t.Helper()
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	session, runtime := read("cuda_session_linux.go"), read("cuda_runtime_linux.go")
	isQ4Condition := func(expr ast.Expr) bool {
		binary, ok := expr.(*ast.BinaryExpr)
		if !ok || binary.Op != token.EQL {
			return false
		}
		kind, ok := binary.Y.(*ast.Ident)
		if !ok || kind.Name != "typeQ4K" {
			return false
		}
		typeID, ok := binary.X.(*ast.SelectorExpr)
		if !ok || typeID.Sel.Name != "typeID" {
			return false
		}
		gate, ok := typeID.X.(*ast.SelectorExpr)
		if !ok || gate.Sel.Name != "expertGate" {
			return false
		}
		layer, ok := gate.X.(*ast.Ident)
		return ok && layer.Name == "l"
	}
	check := func(source string) error {
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, "session.go", source, 0)
		if err != nil {
			return err
		}
		var dispatch, allocation bool
		var dispatches int
		ast.Inspect(f, func(n ast.Node) bool {
			fn, ok := n.(*ast.FuncDecl)
			if !ok {
				return true
			}
			if fn.Name.Name == "evalBatchExperts" {
				for _, stmt := range fn.Body.List {
					branch, ok := stmt.(*ast.IfStmt)
					if !ok {
						continue
					}
					if !isQ4Condition(branch.Cond) {
						continue
					}
					dispatches++
					body := source[fset.Position(branch.Body.Pos()).Offset:fset.Position(branch.Body.End()).Offset]
					dispatch = strings.Contains(body, "return r.evalBatchQ4Experts(rows, index)") && !strings.Contains(body, "evalQ4Experts(")
				}
			}
			if fn.Name.Name == "newCUDABatchResources" {
				ast.Inspect(fn.Body, func(n ast.Node) bool {
					branch, ok := n.(*ast.IfStmt)
					if !ok {
						return true
					}
					cond := source[fset.Position(branch.Cond.Pos()).Offset:fset.Position(branch.Cond.End()).Offset]
					if cond != "cudaModelNeedsQ4(r.model)" {
						return true
					}
					body := source[fset.Position(branch.Body.Pos()).Offset:fset.Position(branch.Body.End()).Offset]
					allocation = strings.Contains(body, "b.dense.q4Input =") && strings.Contains(body, "b.dense.q4Experts =") && strings.Contains(body, "b.dense.q4Mid =")
					return false
				})
			}
			return false
		})
		if dispatches > 1 {
			return fmt.Errorf("ambiguous per-layer batched dispatch")
		}
		if dispatches != 1 || !dispatch {
			return fmt.Errorf("missing per-layer batched dispatch")
		}
		if !allocation {
			return fmt.Errorf("missing Q4-only scratch admission")
		}
		return nil
	}
	for _, tc := range []struct{ name, old, new, failure string }{
		{name: "baseline"},
		{"preceding-guard", "func (r *cudaRuntime) evalBatchExperts(rows []cudaTokenState, index int) error {", "func (r *cudaRuntime) evalBatchExperts(rows []cudaTokenState, index int) error {\n\tif len(rows) == 0 { return nil }", ""},
		{"old-scalar-path", "return r.evalBatchQ4Experts(rows, index)", "for i := range rows { if err := r.evalQ4Experts(&rows[i], index); err != nil { return err } }; return nil", "batched dispatch"},
		{"wrong-layer-type", "if l.expertGate.typeID == typeQ4K {\n\t\treturn r.evalBatchQ4Experts(rows, index)", "if l.expertGate.typeID == typeIQ2XXS {\n\t\treturn r.evalBatchQ4Experts(rows, index)", "batched dispatch"},
		{"q2-scratch-regression", "if cudaModelNeedsQ4(r.model) {\n\t\tb.dense.q4Input", "if true {\n\t\tb.dense.q4Input", "scratch admission"},
		{"duplicate-dispatch", "if l.expertGate.typeID == typeQ4K {\n\t\treturn r.evalBatchQ4Experts(rows, index)", "if l.expertGate.typeID == typeQ4K { return r.evalBatchQ4Experts(rows, index) }; if l.expertGate.typeID == typeQ4K {\n\t\treturn r.evalBatchQ4Experts(rows, index)", "ambiguous per-layer batched dispatch"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := session
			if tc.old != "" {
				if strings.Count(s, tc.old) != 1 {
					t.Fatal("mutation site not unique")
				}
				s = strings.Replace(s, tc.old, tc.new, 1)
			}
			err := check(s)
			if tc.failure == "" && err != nil {
				t.Fatal(err)
			}
			if tc.failure != "" && (err == nil || !strings.Contains(err.Error(), tc.failure)) {
				t.Fatalf("mutation: %v", err)
			}
		})
	}
	for _, name := range []string{"cuda_experts_q4_batch.cu", "ds4_q4_k_pair_selected_q8_k_gather", "ds4_q4_k_selected_q8_k_gather"} {
		if !strings.Contains(runtime, fmt.Sprintf("%q", name)) {
			t.Fatalf("unregistered %s", name)
		}
	}
	orchestrator := read("cuda_experts_q4_batch_linux.go")
	for _, call := range []string{"encodeQuantQ8K(input, q, cudaWidth)", "encodeMoEGroupCSR(", "encodeQ4KPairGather(", "encodeSwiGLUSlots(", "d.profileMid.Slice(0, mid.Len()).CopyDeviceToDeviceAsync(mid", "encodeQuantQ8K(mid, qmid, 2048)", "encodeQ4KDownGather("} {
		if !strings.Contains(orchestrator, call) {
			t.Fatalf("missing production stage %s", call)
		}
	}
}
