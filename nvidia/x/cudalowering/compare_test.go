package cudalowering

import (
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// TestCompare runs the default table over each build of testdata/lowering.cu.
// Every want entry names something ptxas did: the sm_90 build is the control
// for the two builds beside it, and differs from them only in a compiler flag
// or an architecture.
func TestCompare(t *testing.T) {
	tests := []struct {
		name string
		ptx  string
		sass string
		want []string
	}{{
		// The one divergence in an ordinary build: atomicDrop and atomicKeep
		// compile to the same atom.global.add.f32 in PTX, and ptxas substitutes
		// the cheaper reduction for the one whose result nothing reads.
		name: "default build",
		ptx:  "lowering.sm90.ptx",
		sass: "lowering.sm90.sass",
		want: []string{
			"_Z10atomicDropPfPKfi: erased: 1 atom.global, no ATOMG",
		},
	}, {
		// Same PTX, same architecture, register budget cut to 24. The PTX has no
		// local memory access at all, so both of these are the allocator's.
		name: "register budget forces spills",
		ptx:  "lowering.sm90.ptx",
		sass: "lowering.sm90.rr24.sass",
		want: []string{
			"_Z7spillerPfPKfi: spill: 5 LDL, no ld.local",
			"_Z7spillerPfPKfi: spill: 5 STL, no st.local",
			"_Z10atomicDropPfPKfi: erased: 1 atom.global, no ATOMG",
		},
	}, {
		// The same kernels for Blackwell. ptxas pads fmaChain's eight DFMA with
		// 28 NOPs, which the sm_90 build above does not do: a throughput
		// prediction read off the PTX instruction mix is wrong by that much.
		name: "blackwell pads the double pipe",
		ptx:  "lowering.sm121a.ptx",
		sass: "lowering.sm121a.sass",
		want: []string{
			"_Z8fmaChainPdPKdS1_i: padding: 28 NOP ahead of the last real instruction (64 instructions in all)",
			"_Z10atomicDropPfPKfi: erased: 1 atom.global, no ATOMG",
		},
	}, {
		// The SASS of a different module. Comparing the wrong pair of artifacts
		// has to fail; a comparison that pairs nothing and reports nothing would
		// be a gate that cannot fire.
		name: "unrelated disassembly",
		ptx:  "lowering.sm90.ptx",
		sass: "other.sm90.sass",
		want: []string{
			"_Z8fmaChainPdPKdS1_i: missing sass: kernel is in the ptx and not in the disassembly",
			"_Z7spillerPfPKfi: missing sass: kernel is in the ptx and not in the disassembly",
			"_Z13shuffleReducePfPKfi: missing sass: kernel is in the ptx and not in the disassembly",
			"_Z10atomicKeepPfPKfi: missing sass: kernel is in the ptx and not in the disassembly",
			"_Z10atomicDropPfPKfi: missing sass: kernel is in the ptx and not in the disassembly",
			"_Z9unrelatedPii: missing ptx: function is in the disassembly and not in the ptx",
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := Compare(readPTX(t, tt.ptx), readSASS(t, tt.sass), nil)
			if err != nil {
				t.Fatal(err)
			}
			if got := divergences(r); !equalStrings(got, tt.want) {
				t.Errorf("divergences:\ngot:\n\t%s\nwant:\n\t%s",
					strings.Join(got, "\n\t"), strings.Join(tt.want, "\n\t"))
			}
			if err := r.Err(); (err != nil) != (len(tt.want) > 0) {
				t.Errorf("Err() = %v, want non-nil = %v", err, len(tt.want) > 0)
			}
		})
	}
}

// TestCompareStats checks the counts a quiet comparison still reports.
func TestCompareStats(t *testing.T) {
	r, err := Compare(readPTX(t, "lowering.sm90.ptx"), readSASS(t, "lowering.sm90.sass"), nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []Stats{
		{Kernel: "_Z8fmaChainPdPKdS1_i", PTXInstructions: 29, SASSInstructions: 40, NOPs: 12},
		{Kernel: "_Z7spillerPfPKfi", PTXInstructions: 87, SASSInstructions: 96, NOPs: 9},
		{Kernel: "_Z13shuffleReducePfPKfi", PTXInstructions: 50, SASSInstructions: 48, NOPs: 15},
		{Kernel: "_Z10atomicKeepPfPKfi", PTXInstructions: 18, SASSInstructions: 32, NOPs: 13},
		{Kernel: "_Z10atomicDropPfPKfi", PTXInstructions: 16, SASSInstructions: 24, NOPs: 8},
	}
	if len(r.Stats) != len(want) {
		t.Fatalf("got %d stats, want %d", len(r.Stats), len(want))
	}
	for i, w := range want {
		if r.Stats[i] != w {
			t.Errorf("stats[%d] = %+v, want %+v", i, r.Stats[i], w)
		}
	}
}

// TestSpillPressure checks that a spill divergence carries the register
// pressure that explains it. Pressure is supporting detail only: the spill is
// counted from STL and LDL opcodes, which is a different measurement from the
// live register counts joined on here.
func TestSpillPressure(t *testing.T) {
	d := readSASS(t, "lowering.sm90.rr24.sass")
	f, err := os.Open("testdata/lowering.sm90.rr24.life.sass")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	pressure, err := cudasass.ParseLifeRangeCount(f)
	if err != nil {
		t.Fatal(err)
	}
	if err := cudasass.JoinPressure(d, pressure); err != nil {
		t.Fatal(err)
	}

	r, err := Compare(readPTX(t, "lowering.sm90.ptx"), d, nil)
	if err != nil {
		t.Fatal(err)
	}
	var spills int
	for _, div := range r.Divergences {
		if div.Kind != Spill {
			continue
		}
		spills++
		if !strings.HasPrefix(div.Detail, "peak ") || !strings.HasSuffix(div.Detail, " live gpr") {
			t.Errorf("spill detail = %q, want a peak live register count", div.Detail)
		}
	}
	if spills != 2 {
		t.Fatalf("got %d spill divergences, want 2", spills)
	}

	// Without the join the same comparison reports the same spills and no
	// pressure, rather than inventing a number.
	r, err = Compare(readPTX(t, "lowering.sm90.ptx"), readSASS(t, "lowering.sm90.rr24.sass"), nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, div := range r.Divergences {
		if div.Kind == Spill && div.Detail != "" {
			t.Errorf("spill detail = %q with no life ranges read, want empty", div.Detail)
		}
	}
}

// TestCompareOptions checks that the caller's table decides what is reported.
func TestCompareOptions(t *testing.T) {
	m := readPTX(t, "lowering.sm90.ptx")
	d := readSASS(t, "lowering.sm90.sass")

	// An exclusive entry for an opcode the kernels do use, lowered from one they
	// do not: every kernel computes addresses with IMAD and none asks for tex.
	// This is the Unsourced label the spill fixture exercises under its own name.
	r, err := Compare(m, d, &Options{
		Lowerings: []Lowering{{PTX: "tex", SASS: []string{"IMAD"}, Exclusive: true}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Divergences) != 5 {
		t.Fatalf("got %d divergences, want one per kernel:\n%s", len(r.Divergences), r)
	}
	for _, div := range r.Divergences {
		if div.Kind != Unsourced {
			t.Errorf("kind = %v, want %v", div.Kind, Unsourced)
		}
	}

	// The same entry without Exclusive reports nothing: an opcode ptxas emits on
	// its own account cannot be attributed to a PTX opcode that is missing.
	r, err = Compare(m, d, &Options{
		Lowerings: []Lowering{{PTX: "tex", SASS: []string{"IMAD"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Divergences) != 0 {
		t.Errorf("non-exclusive lowering reported:\n%s", r)
	}

	// A negative threshold disables the padding check on the build that trips it.
	r, err = Compare(readPTX(t, "lowering.sm121a.ptx"), readSASS(t, "lowering.sm121a.sass"),
		&Options{Lowerings: []Lowering{}, MaxNOPFraction: -1})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Divergences) != 0 {
		t.Errorf("padding reported with the check disabled:\n%s", r)
	}
}

func TestCompareErrors(t *testing.T) {
	if _, err := Compare(nil, readSASS(t, "lowering.sm90.sass"), nil); err == nil {
		t.Error("Compare with nil module: got nil error")
	}
	if _, err := Compare(readPTX(t, "lowering.sm90.ptx"), nil, nil); err == nil {
		t.Error("Compare with nil disassembly: got nil error")
	}
	if err := (&Report{}).Err(); err != nil {
		t.Errorf("empty report Err() = %v, want nil", err)
	}
}

func TestKindString(t *testing.T) {
	for _, tt := range []struct {
		kind Kind
		want string
	}{
		{Erased, "erased"},
		{Unsourced, "unsourced"},
		{Spill, "spill"},
		{Padding, "padding"},
		{MissingSASS, "missing sass"},
		{MissingPTX, "missing ptx"},
		{Kind(0), "kind(0)"},
	} {
		if got := tt.kind.String(); got != tt.want {
			t.Errorf("Kind(%d).String() = %q, want %q", int(tt.kind), got, tt.want)
		}
	}
}
