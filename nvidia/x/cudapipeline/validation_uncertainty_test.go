package cudapipeline_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func TestWaitParityUncertainty(t *testing.T) {
	for _, opcode := range []string{"mbarrier.try_wait.parity.shared::cta.b64", "mbarrier.test_wait.parity.shared::cta.b64"} {
		for _, tt := range []struct {
			name, operand, violation string
			prefix                   []cudaptx.Instruction
			phase                    int
		}{
			{name: "known zero", operand: "0", phase: 1},
			{name: "known wrong", operand: "1", violation: cudapipeline.ErrPhaseParity, phase: 1},
			{name: "unknown register", operand: "%parity", violation: "unverified wait parity"},
			{name: "unknown expression", operand: "%parity+1", violation: "unverified wait parity"},
			{name: "propagated zero", operand: "%parity", prefix: []cudaptx.Instruction{{Opcode: "mov.u32", Operands: "%parity, 0"}}, phase: 1},
			{name: "propagated wrong", operand: "%parity", prefix: []cudaptx.Instruction{{Opcode: "mov.u32", Operands: "%parity, 1"}}, violation: cudapipeline.ErrPhaseParity, phase: 1},
		} {
			t.Run(opcode+"/"+tt.name, func(t *testing.T) {
				instructions := append([]cudaptx.Instruction(nil), tt.prefix...)
				instructions = append(instructions,
					cudaptx.Instruction{Opcode: "mbarrier.init.shared::cta.b64", Operands: "[bar], 1"},
					cudaptx.Instruction{Opcode: "mbarrier.arrive.expect_tx.shared::cta.b64", Operands: "_, [bar], 16"},
					cudaptx.Instruction{Opcode: "mbarrier.complete_tx.shared::cta.b64", Operands: "[bar], 16"},
					cudaptx.Instruction{Line: 10, Opcode: opcode, Operands: "%p, [bar], " + tt.operand},
				)
				tracker := cudapipeline.NewBarrierTracker()
				violations, err := tracker.ValidateKernel(&cudaptx.Kernel{Body: true, Instructions: instructions})
				if err != nil {
					t.Fatal(err)
				}
				if tt.violation == "" {
					if len(violations) != 0 {
						t.Fatalf("unexpected violations: %v", violations)
					}
				} else if len(violations) != 1 || violations[0].Violation != tt.violation || violations[0].Line != 10 || violations[0].Barrier != "bar" {
					t.Fatalf("violations = %v, want %q on bar at line 10", violations, tt.violation)
				}
				barrier := tracker.Barrier("bar")
				if barrier == nil || barrier.Phase != tt.phase {
					t.Fatalf("barrier = %+v, want phase %d", barrier, tt.phase)
				}
				if tt.phase == 0 {
					if !barrier.Arrived || barrier.ExpectedTxBytes != 16 || barrier.CompletedTxBytes != 16 {
						t.Fatalf("unverified wait discarded pending state: %+v", barrier)
					}
					for _, event := range barrier.Events {
						if event.Kind == cudapipeline.EventWait {
							t.Fatal("unverified parity recorded a completed wait")
						}
					}
				}
			})
		}
	}
}
