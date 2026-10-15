package cudapipeline_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func parseControlKernel(t *testing.T, body string) *cudaptx.Kernel {
	t.Helper()
	text := ".version 8.8\n.target sm_90\n.address_size 64\n.visible .entry fixture() {\n" + body + "\n}\n"
	m, err := cudaptx.Parse(strings.NewReader(text))
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Kernels) != 1 {
		t.Fatalf("kernels = %d, want 1", len(m.Kernels))
	}
	return m.Kernels[0]
}

func TestPipelineControlFlowContract(t *testing.T) {
	const init = "mbarrier.init.shared::cta.b64 [bar], 1;\n"
	for _, tt := range []struct {
		name string
		body string
		op   string
	}{
		{"call", "call helper;\nret;", "call"},
		{"uniform call", "call.uni helper;\nret;", "call.uni"},
		{"early return", "ret;\nmbarrier.arrive.shared::cta.b64 %state, [bar];\nret;", "ret"},
		{"predicated final return", "@%p ret;", "ret"},
		{"predicated early return", "@!%p ret;\nret;", "ret"},
		{"wait label", "bra wait_skip;\nwait_skip:\nret;", "bra"},
		{"spin label", "bra spin_skip;\nspin_skip:\nret;", "bra"},
		{"indirect branch", "brx.idx %index, targets;\nret;", "brx.idx"},
		{"early exit", "exit;\nret;", "exit"},
		{"predicated final exit", "@%p exit;", "exit"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			k := parseControlKernel(t, init+tt.body)
			violations, err := cudapipeline.ValidateKernel(k)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, v := range violations {
				if v.Violation == cudapipeline.ErrUnsupportedControlFlow && v.Opcode == tt.op && v.Line > 0 {
					found = true
				}
			}
			if !found {
				t.Fatalf("missing unsupported control-flow diagnostic for %q: %v", tt.op, violations)
			}
		})
	}
}

func TestPipelineEmittedWaitLoopUnverified(t *testing.T) {
	// The parser drops label definitions. Its instruction list cannot prove
	// this genuine emitted loop's target/backedge, even though guards survive.
	for _, name := range []string{"wait", "spin", "custom"} {
		t.Run(name, func(t *testing.T) {
			loop := cudapipeline.EmitWaitLoop("bar", 0, name)
			body := "mbarrier.init.shared::cta.b64 [bar], 1;\nmbarrier.arrive.shared::cta.b64 %state, [bar];\n" + loop + "\nret;"
			k := parseControlKernel(t, body)
			var branch *cudaptx.Instruction
			for i := range k.Instructions {
				inst := &k.Instructions[i]
				if inst.Opcode == "bra" {
					branch = inst
				}
				if strings.HasSuffix(inst.Opcode, ":") {
					t.Fatal("parser now retains labels; reassess the supported loop contract")
				}
			}
			if branch == nil || branch.Predicate != "@!%p" || branch.Operands != name+"_loop" {
				t.Fatalf("emitted backedge guard/operand not retained: %+v", branch)
			}
			violations, err := cudapipeline.ValidateKernel(k)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, v := range violations {
				found = found || v.Violation == cudapipeline.ErrUnsupportedControlFlow
			}
			if !found {
				t.Fatal("instruction-only analysis claimed an unproven emitted backedge")
			}
		})
	}
}

func TestPipelineStraightLineControlFlow(t *testing.T) {
	for _, end := range []string{"ret;", "ret.uni;", "exit;"} {
		t.Run(end, func(t *testing.T) {
			body := fmt.Sprintf("mbarrier.init.shared::cta.b64 [bar], 1;\nmbarrier.arrive.shared::cta.b64 %%state, [bar];\nmbarrier.try_wait.parity.shared::cta.b64 %%p, [bar], 0;\n%s", end)
			violations, err := cudapipeline.ValidateKernel(parseControlKernel(t, body))
			if err != nil || len(violations) != 0 {
				t.Fatalf("straight-line barrier protocol: %v, %v", violations, err)
			}
		})
	}
	// No barrier protocol is analyzed in this kernel. The pipeline checker is
	// not a general validator of arbitrary PTX control flow.
	violations, err := cudapipeline.ValidateKernel(parseControlKernel(t, "call helper;\nret;"))
	if err != nil || len(violations) != 0 {
		t.Fatalf("kernel without a barrier protocol: %v, %v", violations, err)
	}
}
