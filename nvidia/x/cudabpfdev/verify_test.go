package cudabpfdev

import (
	"context"
	"encoding/binary"
	"errors"
	"strings"
	"testing"
)

func FuzzVerifierBound(f *testing.F) {
	f.Add([]byte{byte(Exit), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, uint32(0x89))
	f.Fuzz(func(t *testing.T, data []byte, mask uint32) {
		if len(data) == 0 || len(data)%12 != 0 || len(data) > 64*12 {
			return
		}
		p := program()
		for off := 0; off < len(data); off += 12 {
			x := data[off : off+12]
			p.Code = append(p.Code, Instruction{Op: Op(x[0]), Dst: x[1], A: x[2], B: x[3], Imm: binary.LittleEndian.Uint32(x[4:]), Target: int(int32(binary.LittleEndian.Uint32(x[8:])))})
		}
		a, err := Verify(p, budget())
		if err != nil {
			return
		}
		r, err := Evaluate(context.Background(), p, budget(), mask)
		if err != nil {
			t.Fatalf("verified program failed host execution: %v", err)
		}
		if r.Steps > a.MaxSteps {
			t.Fatalf("steps=%d bound=%d", r.Steps, a.MaxSteps)
		}
	})
}

func budget() Budget {
	return Budget{MaxInstructions: 64, MaxSteps: 1000, MaxRegisters: 64, MaxSharedBytes: 1024, MaxSlots: 16}
}
func program(code ...Instruction) Program {
	return Program{Version: 1, Registers: 4, Slots: 4, Code: code}
}

func TestVerifierContract(t *testing.T) {
	for _, tt := range []struct {
		name   string
		p      Program
		reason string
	}{
		{"valid", program(Instruction{Op: Const, Dst: 1, Imm: 3}, Instruction{Op: MapAdd, A: 0, B: 1}, Instruction{Op: Exit}), ""},
		{"varying-branch", program(Instruction{Op: Lane, Dst: 1}, Instruction{Op: JumpNonzero, A: 1, Target: 3}, Instruction{Op: Const, Dst: 2, Imm: 1}, Instruction{Op: Exit}), "lane-varying branch"},
		{"varying-key", program(Instruction{Op: Lane, Dst: 1}, Instruction{Op: MapAdd, A: 1}, Instruction{Op: Exit}), "lane-varying map"},
		{"varying-value", program(Instruction{Op: Lane, Dst: 1}, Instruction{Op: MapAdd, B: 1}, Instruction{Op: Exit}), "lane-varying map"},
		{"ballot-branch", program(Instruction{Op: Lane, Dst: 1}, Instruction{Op: Ballot, Dst: 2, A: 1}, Instruction{Op: JumpNonzero, A: 2, Target: 4}, Instruction{Op: Const, Dst: 3, Imm: 1}, Instruction{Op: Exit}), ""},
		{"leader-key", program(Instruction{Op: Lane, Dst: 1}, Instruction{Op: Leader, Dst: 2, A: 1}, Instruction{Op: MapAdd, A: 2}, Instruction{Op: Exit}), ""},
		{"unbounded", program(Instruction{Op: JumpNonzero, Target: 0}, Instruction{Op: Exit}), "unbounded backward"},
		{"bounded", program(Instruction{Op: Add, Dst: 1, A: 1}, Instruction{Op: Repeat, Target: 0, Imm: 3}, Instruction{Op: Exit}), ""},
		{"zero-loop", program(Instruction{Op: Const}, Instruction{Op: Repeat, Target: 0}, Instruction{Op: Exit}), "invalid bounded loop"},
		{"irreducible", program(Instruction{Op: JumpNonzero, Target: 2}, Instruction{Op: Const}, Instruction{Op: Const}, Instruction{Op: Repeat, Target: 1, Imm: 2}, Instruction{Op: Exit}), "irreducible"},
		{"unsupported-barrier", program(Instruction{Op: Op(255)}, Instruction{Op: Exit}), "unsupported opcode"},
		{"unused-fields", program(Instruction{Op: Exit, Imm: 1}), "unused instruction field"},
		{"unreachable", program(Instruction{Op: Exit}, Instruction{Op: Exit}), "unreachable"},
		{"fall-off", program(Instruction{Op: Const}), "falls off"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Verify(tt.p, budget())
			if tt.reason == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			var located *VerifyError
			if !errors.As(err, &located) || !strings.Contains(err.Error(), tt.reason) {
				t.Fatalf("error=%v want %s", err, tt.reason)
			}
		})
	}
}

func TestBytecodeVersionAndBudgets(t *testing.T) {
	p := program(Instruction{Op: Const}, Instruction{Op: Repeat, Imm: 3, Target: 0}, Instruction{Op: Exit})
	a, err := Verify(p, budget())
	if err != nil || a.MaxSteps != 7 || a.Registers != 5 {
		t.Fatalf("analysis=%+v error=%v", a, err)
	}
	for _, mutate := range []func(*Program, *Budget){
		func(p *Program, b *Budget) { p.Version = 2 }, func(p *Program, b *Budget) { b.MaxRegisters = 4 },
		func(p *Program, b *Budget) { b.MaxSteps = 6 }, func(p *Program, b *Budget) { p.StagingBytes = 1025 },
		func(p *Program, b *Budget) { p.Slots = 17 }, func(p *Program, b *Budget) { p.Registers = 0 },
	} {
		q := p
		b := budget()
		mutate(&q, &b)
		if _, err := Verify(q, b); err == nil {
			t.Fatalf("accepted %+v %+v", q, b)
		}
	}
}

func TestUniformityThroughLoopAndJoin(t *testing.T) {
	// The first loop iteration starts with uniform r1. The backedge carries
	// the varying value from instruction 3; a single linear pass misses it.
	p := program(Instruction{Op: JumpNonzero, A: 1, Target: 2}, Instruction{Op: Const, Dst: 2}, Instruction{Op: Const, Dst: 3}, Instruction{Op: Lane, Dst: 1}, Instruction{Op: Repeat, Imm: 2, Target: 0}, Instruction{Op: Exit})
	if _, err := Verify(p, budget()); err == nil || !strings.Contains(err.Error(), "lane-varying branch") {
		t.Fatalf("loop-carried variation accepted: %v", err)
	}
	// One predecessor kills r1's variation, the other retains it.
	p = program(Instruction{Op: Lane, Dst: 1}, Instruction{Op: JumpNonzero, Target: 3}, Instruction{Op: Const, Dst: 1}, Instruction{Op: MapAdd, A: 1}, Instruction{Op: Exit})
	if _, err := Verify(p, budget()); err == nil {
		t.Fatal("join lost varying predecessor")
	}
}

func TestNestedLoopBoundOverflow(t *testing.T) {
	p := program(Instruction{Op: Const}, Instruction{Op: Repeat, Imm: ^uint32(0), Target: 0}, Instruction{Op: Repeat, Imm: ^uint32(0), Target: 0}, Instruction{Op: Repeat, Imm: ^uint32(0), Target: 0}, Instruction{Op: Exit})
	b := budget()
	b.MaxSteps = ^uint64(0)
	if _, err := Verify(p, b); err == nil {
		t.Fatal("accepted overflowing dynamic bound")
	}
}
