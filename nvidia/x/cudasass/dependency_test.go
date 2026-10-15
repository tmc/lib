package cudasass_test

import (
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// block builds a one-block program from SASS text for the dependency tests.
// The encodings are synthetic but structurally real: each carries a control
// code so that DecodeControl succeeds.
func block(t *testing.T, sass string) cudasass.BasicBlock {
	t.Helper()
	d, err := cudasass.ParseString(sass)
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Functions) != 1 {
		t.Fatalf("got %d functions, want 1", len(d.Functions))
	}
	blocks := cudasass.BasicBlocks(d.Functions[0])
	if len(blocks) == 0 {
		t.Fatal("no basic blocks")
	}
	return blocks[0]
}

// ctl is an encoding carrying a plain control code: stall 1, no barriers.
const ctl = `/* 0x0000000000000000 */
                                     /* 0x000fe20000007000 */`

func edgeKinds(edges []cudasass.Edge, from, to int) []cudasass.DepKind {
	var out []cudasass.DepKind
	for _, e := range edges {
		if e.From == from && e.To == to {
			out = append(out, e.Kind)
		}
	}
	return out
}

func hasKind(edges []cudasass.Edge, from, to int, k cudasass.DepKind) bool {
	for _, got := range edgeKinds(edges, from, to) {
		if got == k {
			return true
		}
	}
	return false
}

func TestRAWEdgeDetected(t *testing.T) {
	b := block(t, `.text.k:
        /*0000*/  IADD R4, R2, UR6 ;   `+ctl+`
        /*0010*/  IADD R5, R4, UR7 ;   `+ctl+`
`)
	edges, err := cudasass.Dependencies(b, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if !hasKind(edges, 0, 1, cudasass.RAW) {
		t.Errorf("no RAW edge on R4: %v", edges)
	}
}

func TestWAREdgeDetected(t *testing.T) {
	b := block(t, `.text.k:
        /*0000*/  IADD R4, R2, UR6 ;   `+ctl+`
        /*0010*/  IADD R2, R5, UR7 ;   `+ctl+`
`)
	edges, err := cudasass.Dependencies(b, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if !hasKind(edges, 0, 1, cudasass.WAR) {
		t.Errorf("no WAR edge on R2: %v", edges)
	}
}

func TestWAWEdgeDetected(t *testing.T) {
	b := block(t, `.text.k:
        /*0000*/  IADD R4, R2, UR6 ;   `+ctl+`
        /*0010*/  IADD R4, R5, UR7 ;   `+ctl+`
`)
	edges, err := cudasass.Dependencies(b, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if !hasKind(edges, 0, 1, cudasass.WAW) {
		t.Errorf("no WAW edge on R4: %v", edges)
	}
}

// TestMemoryOrderingEdgePreserved checks that two stores are ordered even
// though they share no register.
func TestMemoryOrderingEdgePreserved(t *testing.T) {
	b := block(t, `.text.k:
        /*0000*/  STG.E [R2.64], R4 ;   `+ctl+`
        /*0010*/  STG.E [R6.64], R8 ;   `+ctl+`
`)
	edges, err := cudasass.Dependencies(b, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if !hasKind(edges, 0, 1, cudasass.Memory) {
		t.Errorf("no memory ordering edge between two stores: %v", edges)
	}
	if got := cudasass.CanSwap(b, 0, 1, cudasass.SM121a, cudasass.NumericalPolicy{}); got.Legal {
		t.Errorf("swapping two stores was allowed: %v", got)
	}
}

// TestReorderAcrossFenceRejected checks that a memory operation may not move
// across a fence.
func TestReorderAcrossFenceRejected(t *testing.T) {
	b := block(t, `.text.k:
        /*0000*/  MEMBAR.SC.GPU ;      `+ctl+`
        /*0010*/  STG.E [R2.64], R4 ;  `+ctl+`
`)
	edges, err := cudasass.Dependencies(b, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if !hasKind(edges, 0, 1, cudasass.Fence) {
		t.Errorf("no fence edge: %v", edges)
	}
	if got := cudasass.CanSwap(b, 0, 1, cudasass.SM121a, cudasass.NumericalPolicy{}); got.Legal {
		t.Errorf("a store moved across a fence: %v", got)
	}
}

// TestIllegalSwapViolatingLivenessRejected checks that a swap breaking a RAW
// dependency is refused, and names the register.
func TestIllegalSwapViolatingLivenessRejected(t *testing.T) {
	b := block(t, `.text.k:
        /*0000*/  IADD R4, R2, UR6 ;   `+ctl+`
        /*0010*/  IADD R5, R4, UR7 ;   `+ctl+`
`)
	got := cudasass.CanSwap(b, 0, 1, cudasass.SM121a, cudasass.NumericalPolicy{})
	if got.Legal {
		t.Fatalf("swap allowed: %v", got)
	}
	if !strings.Contains(got.String(), "RAW") || !strings.Contains(got.String(), "R4") {
		t.Errorf("verdict %q does not name the RAW dependency on R4", got)
	}
}

// TestFloatAccumulationReorderRejectedUnderPolicy checks that floating-point
// reassociation is refused unless the kernel's policy permits it, and that the
// default policy does not.
func TestFloatAccumulationReorderRejectedUnderPolicy(t *testing.T) {
	b := block(t, `.text.k:
        /*0000*/  FADD R4, R2, R6 ;    `+ctl+`
        /*0010*/  FADD R5, R3, R7 ;    `+ctl+`
`)
	if got := cudasass.CanSwap(b, 0, 1, cudasass.SM121a, cudasass.NumericalPolicy{}); got.Legal {
		t.Errorf("float reassociation allowed under the default policy: %v", got)
	}
	permissive := cudasass.NumericalPolicy{AllowFloatReassociation: true}
	if got := cudasass.CanSwap(b, 0, 1, cudasass.SM121a, permissive); !got.Legal {
		t.Errorf("independent FADDs refused under a permissive policy: %v", got)
	}
}

// TestSwapDroppingBarrierIndexRejected checks that a swap is refused when one
// instruction waits on a scoreboard barrier the other sets.
func TestSwapDroppingBarrierIndexRejected(t *testing.T) {
	// The first instruction sets read barrier 0; the second waits on it.
	const setsBarrier = `/* 0x0000000000000000 */
                                     /* 0x000e220000007000 */`
	const waitsBarrier = `/* 0x0000000000000000 */
                                     /* 0x001fe20000007000 */`
	b := block(t, `.text.k:
        /*0000*/  IADD R4, R2, UR6 ;   `+setsBarrier+`
        /*0010*/  IADD R8, R9, UR7 ;   `+waitsBarrier+`
`)
	edges, err := cudasass.Dependencies(b, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if !hasKind(edges, 0, 1, cudasass.Barrier) {
		t.Fatalf("no barrier edge; control codes: %v", edges)
	}
	got := cudasass.CanSwap(b, 0, 1, cudasass.SM121a, cudasass.NumericalPolicy{})
	if got.Legal {
		t.Errorf("a swap dropped a barrier wait: %v", got)
	}
}

// TestControlCodeRecomputationFailureRejects checks that a swap of two
// instructions whose stall counts differ is refused, because no sm_121a
// latency table is available to recompute them.
func TestControlCodeRecomputationFailureRejects(t *testing.T) {
	const stall1 = `/* 0x0000000000000000 */
                                     /* 0x000fe20000007000 */`
	const stall2 = `/* 0x0000000000000000 */
                                     /* 0x000fe40000007000 */`
	b := block(t, `.text.k:
        /*0000*/  IADD R4, R2, UR6 ;   `+stall1+`
        /*0010*/  IADD R8, R9, UR7 ;   `+stall2+`
`)
	a, c := b.Instructions[0], b.Instructions[1]
	ca, err := cudasass.DecodeControl(a.Encoding, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	cc, err := cudasass.DecodeControl(c.Encoding, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if ca.Stall == cc.Stall {
		t.Fatalf("fixture stalls are equal (%d); the test does not exercise the refusal", ca.Stall)
	}
	if _, _, err := cudasass.RecomputeControl(a, c, cudasass.SM121a); err == nil {
		t.Error("RecomputeControl succeeded on differing stall counts")
	}
	got := cudasass.CanSwap(b, 0, 1, cudasass.SM121a, cudasass.NumericalPolicy{})
	if got.Legal {
		t.Errorf("swap allowed despite uncomputable control codes: %v", got)
	}
}

// TestUnclassifiedOpcodeRefusesRatherThanGuesses checks that an opcode outside
// the covered subset makes the analysis refuse, not return a partial graph.
func TestUnclassifiedOpcodeRefusesRatherThanGuesses(t *testing.T) {
	b := block(t, `.text.k:
        /*0000*/  HMMA.16816.F32 R4, R8, R10, R4 ;  `+ctl+`
        /*0010*/  IADD R5, R4, UR7 ;                `+ctl+`
`)
	if _, err := cudasass.Dependencies(b, cudasass.SM121a); err == nil {
		t.Error("Dependencies accepted an unclassified opcode")
	}
	got := cudasass.CanSwap(b, 0, 1, cudasass.SM121a, cudasass.NumericalPolicy{})
	if got.Legal {
		t.Error("CanSwap allowed a swap it could not analyse")
	}
	if !got.Undecided {
		t.Errorf("verdict %v is not marked undecided", got)
	}
}

// TestRegisterAndPredicateLiveness checks the backward pass over a block,
// including that a guarding predicate counts as a use.
func TestRegisterAndPredicateLiveness(t *testing.T) {
	b := block(t, `.text.k:
        /*0000*/  IADD R4, R2, UR6 ;       `+ctl+`
        /*0010*/  @P0 IADD R5, R4, UR7 ;   `+ctl+`
        /*0020*/  EXIT ;                   `+ctl+`
`)
	live, err := cudasass.Liveness(b, nil, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if len(live) != 3 {
		t.Fatalf("got %d live sets, want 3", len(live))
	}
	// After instruction 0, R4 is live because instruction 1 reads it, and P0
	// is live because it guards instruction 1.
	if !live[0].Contains(cudasass.Register{Class: cudasass.GPR, Num: 4}) {
		t.Errorf("R4 not live after instruction 0: %v", live[0].Registers())
	}
	if !live[0].Contains(cudasass.Register{Class: cudasass.Pred, Num: 0}) {
		t.Errorf("P0 not live after instruction 0: %v", live[0].Registers())
	}
	// R5 is written by instruction 1 and read by nobody, so it is dead.
	if live[1].Contains(cudasass.Register{Class: cudasass.GPR, Num: 5}) {
		t.Errorf("R5 live after instruction 1: %v", live[1].Registers())
	}
	// RZ is a sink and is never live.
	if live[0].Contains(cudasass.Register{Class: cudasass.GPR, Num: cudasass.RZNum}) {
		t.Error("RZ reported live")
	}
}

// TestPeakLiveGPRNotReportedAsAllocatedRegisters checks that the two are kept
// apart: peak live GPR is a property of a life-range analysis, and the number
// of registers allocated to a kernel is a different number that this package
// does not compute.
func TestPeakLiveGPRNotReportedAsAllocatedRegisters(t *testing.T) {
	b := block(t, `.text.k:
        /*0000*/  IADD R4, R2, UR6 ;   `+ctl+`
        /*0010*/  IADD R5, R4, UR7 ;   `+ctl+`
`)
	live, err := cudasass.Liveness(b, nil, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	peak := 0
	for _, s := range live {
		n := 0
		for _, r := range s.Registers() {
			if r.Class == cudasass.GPR {
				n++
			}
		}
		if n > peak {
			peak = n
		}
	}
	// The block names R2, R4, R5 and UR6/UR7, so an allocated-register count
	// would be at least 6 (R0..R5). Peak simultaneous liveness is smaller.
	if peak >= 6 {
		t.Errorf("peak live GPR = %d, which is an allocated-register count, not a liveness count", peak)
	}
	// The pressure figures nvdisasm reports are a separate channel and are
	// absent here, not zero.
	if got := b.Instructions[0].LiveGPR; got != 0 {
		t.Errorf("LiveGPR = %d from a disassembly with no life-range column; want 0 meaning absent", got)
	}
}

// TestBasicBlockReconstructionAgainstNvdisasm compares the blocks this package
// reconstructs with the ones nvdisasm reports for the same CUBIN.
//
// nvdisasm is the authority here: the fixture is its own -bbcfg output, so a
// disagreement is this package's defect. The comparison is over block leader
// addresses. It is restricted to the addresses nvdisasm's graph names, because
// -c disassembly of a section also prints the NOP padding after the last
// instruction and the control-flow graph does not; that padding is a property
// of the section, not of the program, and neither view is wrong.
func TestBasicBlockReconstructionAgainstNvdisasm(t *testing.T) {
	dot, err := os.ReadFile("testdata/roundtrip/sm121a-roundtrip.bbcfg.dot")
	if err != nil {
		t.Fatal(err)
	}
	d := readRoundTripFixture(t)
	for _, fn := range d.Functions {
		t.Run(fn.Name, func(t *testing.T) {
			want, last := nvdisasmLeaders(t, string(dot), fn.Name)
			if len(want) == 0 {
				t.Fatalf("no blocks for %s in the CFG fixture", fn.Name)
			}
			var got []uint64
			for _, b := range cudasass.BasicBlocks(fn) {
				if b.Start <= last {
					got = append(got, b.Start)
				}
			}
			if len(got) != len(want) {
				t.Fatalf("reconstructed %d blocks, nvdisasm reports %d\n got %#x\nwant %#x", len(got), len(want), got, want)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("block %d starts at %#x, nvdisasm says %#x", i, got[i], want[i])
				}
			}
		})
	}
}

// nvdisasmLeaders returns the block leader addresses nvdisasm's -bbcfg output
// gives for one function, and the address of its last instruction. Each node's
// label opens with the offset of its first instruction, printed as four hex
// digits followed by a colon.
func nvdisasmLeaders(t *testing.T, dot, fn string) (leaders []uint64, last uint64) {
	t.Helper()
	cluster := `subgraph "cluster_` + fn + `" {`
	i := strings.Index(dot, cluster)
	if i < 0 {
		t.Fatalf("no cluster for %s", fn)
	}
	rest := dot[i+len(cluster):]
	if j := strings.Index(rest, `subgraph "cluster_`); j >= 0 {
		rest = rest[:j]
	}
	offset := regexp.MustCompile(`([0-9a-f]{4}):\\ `)
	for _, label := range strings.Split(rest, "[label=")[1:] {
		ms := offset.FindAllStringSubmatch(label, -1)
		if len(ms) == 0 {
			continue
		}
		first, err := strconv.ParseUint(ms[0][1], 16, 64)
		if err != nil {
			t.Fatal(err)
		}
		leaders = append(leaders, first)
		l, err := strconv.ParseUint(ms[len(ms)-1][1], 16, 64)
		if err != nil {
			t.Fatal(err)
		}
		if l > last {
			last = l
		}
	}
	sort.Slice(leaders, func(a, b int) bool { return leaders[a] < leaders[b] })
	return leaders, last
}

// TestStallPercentagesMeasuredOrAbsent holds the boundary that a stall
// percentage is measured or absent, never synthesized. This package runs no
// sampler, so every instruction in the fixture reports absence, and the
// absence must be the sentinel rather than a plausible zero. The control
// code's static stall count is a different quantity and must not be presented
// as one.
func TestStallPercentagesMeasuredOrAbsent(t *testing.T) {
	fn := roundTripFunction(t)
	for _, inst := range fn.Instructions {
		profiles, ok := cudasass.InstructionStalls(inst)
		if ok || profiles != nil {
			t.Fatalf("%#x %s reports %d stall profiles; nothing here measured one",
				inst.Address, inst.Opcode, len(profiles))
		}
	}

	for _, tc := range []struct {
		name    string
		profile cudasass.StallProfile
		wantErr bool
	}{
		{
			name:    "measured",
			profile: cudasass.StallProfile{Reason: "long_scoreboard", Percent: 31.5, Measured: true, Source: "ncu --set full"},
		},
		{
			name:    "absent",
			profile: cudasass.StallProfile{Reason: "long_scoreboard", Percent: cudasass.StallSampleUnavailable},
		},
		{
			name:    "absence synthesized as zero",
			profile: cudasass.StallProfile{Reason: "long_scoreboard", Percent: 0},
			wantErr: true,
		},
		{
			name:    "measurement with no sampler",
			profile: cudasass.StallProfile{Reason: "long_scoreboard", Percent: 31.5, Measured: true},
			wantErr: true,
		},
		{
			name:    "absence naming a sampler",
			profile: cudasass.StallProfile{Reason: "long_scoreboard", Percent: cudasass.StallSampleUnavailable, Source: "ncu"},
			wantErr: true,
		},
		{
			name:    "measured out of range",
			profile: cudasass.StallProfile{Reason: "long_scoreboard", Percent: 140, Measured: true, Source: "ncu"},
			wantErr: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.profile.Valid()
			if tc.wantErr != (err != nil) {
				t.Fatalf("Valid() = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}

	if cudasass.StallSampleUnavailable == 0 {
		t.Error("the unavailable sentinel is zero, so an unmeasured instruction is indistinguishable from one sampled at 0%")
	}
}
