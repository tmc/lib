package cudasass_test

import (
	"maps"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func gpr(n int) cudasass.Register   { return cudasass.Register{Class: cudasass.GPR, Num: n} }
func pred(n int) cudasass.Register  { return cudasass.Register{Class: cudasass.Pred, Num: n} }
func upred(n int) cudasass.Register { return cudasass.Register{Class: cudasass.UPred, Num: n} }

// TestPredicatedWriteDoesNotKill checks that only a write that always executes
// kills its destination. Each block is a NOP followed by the instruction under
// test, so live[0] is the set live immediately before that instruction.
func TestPredicatedWriteDoesNotKill(t *testing.T) {
	r4 := gpr(4)
	tests := []struct {
		name      string
		predicate string
		opcode    string
		operands  string
		liveOut   []cudasass.Register
		live      []cudasass.Register // must be live before the instruction
		dead      []cudasass.Register // must not be live before it
	}{
		{"unguarded kills", "", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, nil, []cudasass.Register{r4}},
		{"PT kills", "@PT", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, nil, []cudasass.Register{r4}},
		{"UPT kills", "@UPT", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, nil, []cudasass.Register{r4}},
		{"P0 preserves", "@P0", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, []cudasass.Register{r4, pred(0)}, nil},
		{"not P0 preserves", "@!P0", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, []cudasass.Register{r4, pred(0)}, nil},
		{"P6 preserves", "@P6", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, []cudasass.Register{r4, pred(6)}, nil},
		{"UP0 preserves", "@UP0", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, []cudasass.Register{r4, upred(0)}, nil},
		{"not UP3 preserves", "@!UP3", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, []cudasass.Register{r4, upred(3)}, nil},
		{"not PT never executes", "@!PT", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, []cudasass.Register{r4}, nil},
		{"not UPT never executes", "@!UPT", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, []cudasass.Register{r4}, nil},
		{"malformed guard preserves", "@Q3", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, []cudasass.Register{r4}, nil},
		{"spaced guard preserves", "@ PT", "IADD", "R4, R2, UR6", []cudasass.Register{r4}, []cudasass.Register{r4}, nil},
		{"guarded dead destination", "@P0", "IADD", "R4, R2, UR6", nil, []cudasass.Register{pred(0), gpr(2)}, []cudasass.Register{r4}},
		{"unguarded dead destination", "", "IADD", "R4, R2, UR6", nil, []cudasass.Register{gpr(2)}, []cudasass.Register{r4}},
		{"guarded read before write", "@P0", "IADD", "R4, R4, UR6", nil, []cudasass.Register{r4}, nil},
		{"unguarded read before write", "", "IADD", "R4, R4, UR6", nil, []cudasass.Register{r4}, nil},
		{"guarded predicate def preserves", "@P0", "ISETP.GE.AND", "P1, PT, R0, R1, PT", []cudasass.Register{pred(1)}, []cudasass.Register{pred(1)}, nil},
		{"unguarded predicate def kills", "", "ISETP.GE.AND", "P1, PT, R0, R1, PT", []cudasass.Register{pred(1)}, nil, []cudasass.Register{pred(1)}},
		{"guard redefines itself", "@P0", "ISETP.GE.AND", "P0, PT, R0, R1, PT", []cudasass.Register{pred(0)}, []cudasass.Register{pred(0)}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := cudasass.BasicBlock{Instructions: []cudasass.Instruction{
				{Address: 0x0, Opcode: "NOP"},
				{Address: 0x10, Predicate: tt.predicate, Opcode: tt.opcode, Operands: tt.operands},
			}}
			liveOut := cudasass.LiveSet{}
			for _, r := range tt.liveOut {
				liveOut[r] = true
			}
			saved := maps.Clone(liveOut)
			live, err := cudasass.Liveness(b, liveOut, cudasass.SM121a)
			if err != nil {
				t.Fatal(err)
			}
			if !maps.Equal(liveOut, saved) {
				t.Errorf("liveOut changed to %v, want %v", liveOut.Registers(), saved.Registers())
			}
			if !maps.Equal(live[1], saved) {
				t.Errorf("live after last instruction = %v, want liveOut %v", live[1].Registers(), saved.Registers())
			}
			for _, r := range tt.live {
				if !live[0].Contains(r) {
					t.Errorf("%s not live before %s %s %s: %v", r, tt.predicate, tt.opcode, tt.operands, live[0].Registers())
				}
			}
			for _, r := range tt.dead {
				if live[0].Contains(r) {
					t.Errorf("%s live before %s %s %s: %v", r, tt.predicate, tt.opcode, tt.operands, live[0].Registers())
				}
			}
		})
	}
}

// TestPredicatedWriteAfterUnconditionalKill checks that a guarded write keeps
// a register live back to the unconditional write that last killed it, and no
// further.
func TestPredicatedWriteAfterUnconditionalKill(t *testing.T) {
	b := cudasass.BasicBlock{Instructions: []cudasass.Instruction{
		{Address: 0x0, Opcode: "NOP"},
		{Address: 0x10, Opcode: "IADD", Operands: "R4, R2, UR6"},
		{Address: 0x20, Predicate: "@P0", Opcode: "IADD", Operands: "R4, R3, UR6"},
	}}
	live, err := cudasass.Liveness(b, cudasass.LiveSet{gpr(4): true}, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if !live[1].Contains(gpr(4)) {
		t.Errorf("R4 not live before the guarded write: %v", live[1].Registers())
	}
	if live[0].Contains(gpr(4)) {
		t.Errorf("R4 live before the unconditional write: %v", live[0].Registers())
	}
}

// TestPredicatedLivenessFromParsedText checks the guards as the parser
// records them from nvdisasm text.
func TestPredicatedLivenessFromParsedText(t *testing.T) {
	b := block(t, `.text.k:
        /*0000*/  NOP ;                    `+ctl+`
        /*0010*/  @P0 IADD R4, R2, UR6 ;   `+ctl+`
        /*0020*/  NOP ;                    `+ctl+`
        /*0030*/  @!PT IADD R5, R2, UR6 ;  `+ctl+`
        /*0040*/  NOP ;                    `+ctl+`
        /*0050*/  @PT IADD R6, R2, UR6 ;   `+ctl+`
`)
	if got := []string{b.Instructions[1].Predicate, b.Instructions[3].Predicate, b.Instructions[5].Predicate}; !slices.Equal(got, []string{"@P0", "@!PT", "@PT"}) {
		t.Fatalf("parsed predicates = %q", got)
	}
	liveOut := cudasass.LiveSet{gpr(4): true, gpr(5): true, gpr(6): true}
	live, err := cudasass.Liveness(b, liveOut, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if !live[0].Contains(gpr(4)) {
		t.Errorf("R4 not live before @P0 write: %v", live[0].Registers())
	}
	if !live[2].Contains(gpr(5)) {
		t.Errorf("R5 not live before @!PT write: %v", live[2].Registers())
	}
	if live[4].Contains(gpr(6)) {
		t.Errorf("R6 live before @PT write: %v", live[4].Registers())
	}
}

// TestPredicatedWriteStaysMayDef checks that the liveness rule leaves
// DefsUses and Dependencies alone: a guarded write is still reported as a
// definition and still orders against other accesses to its destination.
func TestPredicatedWriteStaysMayDef(t *testing.T) {
	guarded := cudasass.Instruction{Address: 0x10, Predicate: "@P0", Opcode: "IADD", Operands: "R4, R2, UR6"}
	defs, uses, ok := cudasass.DefsUses(guarded)
	if !ok {
		t.Fatal("IADD outside the covered subset")
	}
	if !slices.Equal(defs, []cudasass.Register{gpr(4)}) {
		t.Errorf("defs = %v, want [R4]", defs)
	}
	if !slices.Contains(uses, pred(0)) {
		t.Errorf("uses = %v, want P0 among them", uses)
	}
	b := cudasass.BasicBlock{Instructions: []cudasass.Instruction{
		{Address: 0x0, Opcode: "IADD", Operands: "R4, R3, UR6"},
		guarded,
		{Address: 0x20, Opcode: "IADD", Operands: "R7, R4, UR6"},
	}}
	edges, err := cudasass.Dependencies(b, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(edgeKinds(edges, 0, 1), cudasass.WAW) {
		t.Errorf("no WAW edge 0->1: %v", edges)
	}
	if !slices.Contains(edgeKinds(edges, 1, 2), cudasass.RAW) {
		t.Errorf("no RAW edge 1->2: %v", edges)
	}
}
