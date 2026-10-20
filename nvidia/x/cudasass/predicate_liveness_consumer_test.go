package cudasass

import (
	"cmp"
	"maps"
	"strings"
	"testing"
)

// TestDefsStayDeadAcrossGuardedWrite checks the liveness gate of
// CheckMutation against a later guarded write. Renaming a definition onto R4
// ahead of @P0 IADD R4 clobbers the R4 a successor reads whenever P0 is
// false, so it must be refused; ahead of @PT IADD R4 the old R4 is always
// overwritten, so the rename is safe as far as liveness goes.
func TestDefsStayDeadAcrossGuardedWrite(t *testing.T) {
	r4 := Register{Class: GPR, Num: 4}
	rep := Instruction{Address: 0x0, Opcode: "IADD", Operands: "R4, R2, UR6"}
	tests := []struct {
		guard  string
		refuse bool
	}{
		{"@P0", true},
		{"@!P0", true},
		{"@UP1", true},
		{"@!PT", true},
		{"@PT", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(cmp.Or(tt.guard, "unguarded"), func(t *testing.T) {
			b := BasicBlock{Instructions: []Instruction{
				{Address: 0x0, Opcode: "IADD", Operands: "R5, R2, UR6"},
				{Address: 0x10, Predicate: tt.guard, Opcode: "IADD", Operands: "R4, R3, UR6"},
			}}
			liveOut := LiveSet{r4: true}
			err := defsStayDead(b, 0, rep, liveOut, SM121a)
			if tt.refuse {
				if err == nil {
					t.Fatalf("rename onto R4 ahead of %s IADD R4 accepted; R4 is live across the guarded write", tt.guard)
				}
				if !strings.Contains(err.Error(), "live after it") {
					t.Errorf("refusal does not name liveness: %v", err)
				}
			} else if err != nil {
				t.Errorf("rename onto R4 ahead of %q IADD R4 refused: %v", tt.guard, err)
			}
			if !maps.Equal(liveOut, LiveSet{r4: true}) {
				t.Errorf("liveOut changed to %v", liveOut.Registers())
			}
		})
	}
}

// TestCheckMutationRefusesRenameAheadOfGuardedWrite is the same rename
// through CheckMutation on re-encoded instructions. CheckMutation refuses it,
// but its edge gate does so before liveness is consulted: the rename adds a
// WAW edge onto the later write whatever that write's guard. That is why the
// liveness gate is tested directly above.
func TestCheckMutationRefusesRenameAheadOfGuardedWrite(t *testing.T) {
	for _, guard := range []string{"@P0", "@PT"} {
		fn := &Function{Name: "k", Instructions: []Instruction{
			{Address: 0x0, Opcode: "IADD", Operands: "R5, R2, UR6"},
			{Address: 0x10, Predicate: guard, Opcode: "IADD", Operands: "R4, R3, UR6"},
		}}
		for i := range fn.Instructions {
			fn.Instructions[i].Encoding = Encoding{Hi: 0x000fe20000007000}
			e, err := ReEncode(fn.Instructions[i], SM121a)
			if err != nil {
				t.Fatal(err)
			}
			fn.Instructions[i].Encoding = e
		}
		m := Mutation{Index: 0, Replacement: Instruction{Opcode: "IADD", Operands: "R4, R2, UR6"}}
		_, err := CheckMutation(fn, SM121a, m, LiveSet{{Class: GPR, Num: 4}: true})
		if err == nil || !strings.Contains(err.Error(), "WAW 0->1 on R4") {
			t.Errorf("%s follower: err = %v, want a refusal naming WAW 0->1 on R4", guard, err)
		}
	}
}

// TestLivenessMatchesMayDefOracle compares Liveness with an independent
// backward pass that kills a destination only on an unguarded or @PT/@UPT
// write. Each block puts a NOP first, so the set after it is the set live
// before the instruction under test; an implementation that kills on a
// guarded write loses the old value there.
func TestLivenessMatchesMayDefOracle(t *testing.T) {
	oracle := func(b BasicBlock, liveOut LiveSet) []LiveSet {
		out := make([]LiveSet, len(b.Instructions))
		live := maps.Clone(liveOut)
		for i := len(b.Instructions) - 1; i >= 0; i-- {
			out[i] = maps.Clone(live)
			inst := b.Instructions[i]
			defs, uses, _ := DefsUses(inst)
			switch strings.TrimSpace(inst.Predicate) {
			case "", "@PT", "@UPT":
				for _, d := range defs {
					delete(live, d)
				}
			}
			for _, u := range uses {
				live[u] = true
			}
		}
		return out
	}
	r4, r5 := Register{Class: GPR, Num: 4}, Register{Class: GPR, Num: 5}
	tests := []struct {
		name    string
		insts   []Instruction
		liveOut LiveSet
	}{
		{"guarded write", []Instruction{
			{Opcode: "NOP"},
			{Predicate: "@P0", Opcode: "IADD", Operands: "R4, R2, UR6"},
		}, LiveSet{r4: true}},
		{"guarded then unguarded", []Instruction{
			{Opcode: "NOP"},
			{Opcode: "IADD", Operands: "R4, R2, UR6"},
			{Opcode: "NOP"},
			{Predicate: "@!P1", Opcode: "IADD", Operands: "R4, R3, UR6"},
		}, LiveSet{r4: true}},
		{"two guarded writes", []Instruction{
			{Opcode: "NOP"},
			{Predicate: "@P0", Opcode: "IADD", Operands: "R4, R2, UR6"},
			{Predicate: "@UP2", Opcode: "IADD", Operands: "R5, R4, UR6"},
		}, LiveSet{r5: true}},
		{"never executes", []Instruction{
			{Opcode: "NOP"},
			{Predicate: "@!PT", Opcode: "IADD", Operands: "R4, R2, UR6"},
		}, LiveSet{r4: true}},
		{"unguarded", []Instruction{
			{Opcode: "NOP"},
			{Predicate: "@PT", Opcode: "IADD", Operands: "R4, R2, UR6"},
		}, LiveSet{r4: true}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := BasicBlock{Instructions: tt.insts}
			for i := range b.Instructions {
				b.Instructions[i].Address = uint64(i) * 0x10
			}
			got, err := Liveness(b, tt.liveOut, SM121a)
			if err != nil {
				t.Fatal(err)
			}
			want := oracle(b, tt.liveOut)
			for i := range want {
				if !maps.Equal(got[i], want[i]) {
					t.Errorf("after %d: Liveness %v, oracle %v", i, got[i].Registers(), want[i].Registers())
				}
			}
		})
	}
}
