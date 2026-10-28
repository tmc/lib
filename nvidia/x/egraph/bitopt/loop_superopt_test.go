package bitopt

import (
	"math"
	"strings"
	"testing"
)

// nvfp4Epilogue returns the naive lowering of an NVFP4 quantization
// epilogue: groups·8 FP32 accumulators, then one scale, each accumulator
// scaled, converted to an E2M1 code on its own and shifted into place in a
// left-deep OR chain, one 32-bit word per group of eight. Each group is one
// tcgen05.ld .x8.
func nvfp4Epilogue(groups int) EpilogueDAG {
	n := 8*groups + 1
	d := EpilogueDAG{NumInputs: n}
	add := func(x DAGNode) int {
		d.Nodes = append(d.Nodes, x)
		return n + len(d.Nodes) - 1
	}
	zero := add(DAGNode{Op: DAGConst})
	scale := n - 1
	for g := range groups {
		word := -1
		var grp []int
		for e := range 8 {
			in := 8*g + e
			grp = append(grp, in)
			m := add(DAGNode{Op: DAGFMul, A: in, B: scale})
			t := add(DAGNode{Op: DAGCvtE2M1x2, A: zero, B: m})
			if e > 0 {
				t = add(DAGNode{Op: DAGShl, A: t, Imm: uint32(4 * e)})
			}
			if word < 0 {
				word = t
			} else {
				word = add(DAGNode{Op: DAGOr, A: word, B: t})
			}
		}
		d.Outputs = append(d.Outputs, word)
		d.LoadGroups = append(d.LoadGroups, grp)
	}
	return d
}

// referenceNVFP4 packs eight scaled codes per word, code e at bits 4e.
func referenceNVFP4(in []uint32, groups int) []uint32 {
	s := math.Float32frombits(in[len(in)-1])
	out := make([]uint32, groups)
	for g := range groups {
		for e := range 8 {
			x := math.Float32frombits(in[8*g+e])
			out[g] |= e2m1(x*s) << (4 * e)
		}
	}
	return out
}

func TestE2M1(t *testing.T) {
	for _, tc := range []struct {
		x    float32
		want uint32
	}{
		{0, 0}, {0.25, 0}, {0.2500001, 1}, {0.5, 1}, {0.75, 2}, {1, 2}, {1.25, 2},
		{1.5, 3}, {1.75, 4}, {2, 4}, {2.5, 4}, {3, 5}, {3.5, 6}, {4, 6}, {5, 6},
		{5.0001, 7}, {6, 7}, {1e30, 7}, {float32(math.Inf(1)), 7},
		{-1, 0xa}, {-6, 0xf}, {float32(math.Copysign(0, -1)), 0x8}, {-0.1, 0x8},
	} {
		if got := e2m1(tc.x); got != tc.want {
			t.Errorf("e2m1(%v) = %#x, want %#x", tc.x, got, tc.want)
		}
	}
}

func TestNVFP4EpilogueReference(t *testing.T) {
	d := nvfp4Epilogue(2)
	if err := d.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, v := range TestVectors(d.NumInputs, 200, 7) {
		got, want := d.Eval(v), referenceNVFP4(v, 2)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("naive DAG on %#x = %#x, want %#x", v, got, want)
			}
		}
	}
}

func TestSynthesizeEpilogueDAGNVFP4(t *testing.T) {
	target := nvfp4Epilogue(4)
	res, err := SynthesizeEpilogueDAG(target, 2)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("instructions %d -> %d, peak live %d -> %d", res.OriginalInsts, res.SuperoptInsts, res.OriginalPeakLive, res.PeakLiveRegs)
	if res.OriginalInsts != 120 {
		t.Errorf("OriginalInsts = %d, want 120", res.OriginalInsts)
	}
	// The spec's bar: at least 25% fewer instructions and 30% fewer live
	// registers.
	if 4*res.SuperoptInsts > 3*res.OriginalInsts {
		t.Errorf("SuperoptInsts = %d, want at most 75%% of %d", res.SuperoptInsts, res.OriginalInsts)
	}
	if 10*res.PeakLiveRegs > 7*res.OriginalPeakLive {
		t.Errorf("PeakLiveRegs = %d, want at most 70%% of %d", res.PeakLiveRegs, res.OriginalPeakLive)
	}
	vectors := TestVectors(target.NumInputs, 10000, 11)
	if err := CheckEquivalent(target, res.DAG, vectors); err != nil {
		t.Fatal(err)
	}
	for _, v := range vectors[:2000] {
		got, want := res.DAG.Eval(v), referenceNVFP4(v, 4)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("optimized DAG on %#x = %#x, reference %#x", v, got, want)
			}
		}
	}
	n := strings.Count(res.PTX, "cvt.rn.satfinite.e2m1x2.f32")
	if n != 16 {
		t.Errorf("PTX has %d e2m1x2 conversions, want 16 (one per pair)", n)
	}
	if err := VerifyWaitDataflow(res.PTX); err != nil {
		t.Errorf("emitted PTX: %v", err)
	}
	if n := strings.Count(res.PTX, "tcgen05.ld.sync.aligned.32x32b.x8.b32"); n != 4 {
		t.Errorf("PTX has %d x8 loads, want 4", n)
	}
	if _, _, err := CoalesceWaits(res.PTX); err != nil {
		t.Errorf("CoalesceWaits on emitted PTX: %v", err)
	}
	if peak := res.DAG.peak(res.Schedule, res.DAG.LoadGroups); peak != res.PeakLiveRegs {
		t.Errorf("schedule peak %d, reported %d", peak, res.PeakLiveRegs)
	}
}

// TestCheckEquivalentCatchesSwappedCvt checks that the whole-DAG check
// catches a wrong fusion: a cvt with its operands swapped puts the two
// codes in the wrong nibbles.
func TestCheckEquivalentCatchesSwappedCvt(t *testing.T) {
	target := nvfp4Epilogue(1)
	res, err := SynthesizeEpilogueDAG(target, 2)
	if err != nil {
		t.Fatal(err)
	}
	mut := res.DAG
	mut.Nodes = append([]DAGNode(nil), mut.Nodes...)
	swapped := false
	for i, x := range mut.Nodes {
		if x.Op == DAGCvtE2M1x2 {
			mut.Nodes[i].A, mut.Nodes[i].B = x.B, x.A
			swapped = true
			break
		}
	}
	if !swapped {
		t.Fatal("no cvt in the result")
	}
	if err := CheckEquivalent(target, mut, TestVectors(target.NumInputs, 100, 5)); err == nil {
		t.Fatal("CheckEquivalent accepted a DAG with a cvt's operands swapped")
	}
}

func TestSynthesizeEpilogueDAGSmall(t *testing.T) {
	for _, tc := range []struct {
		name string
		d    EpilogueDAG
		want int // instructions after
	}{
		{"duplicate subexpression", EpilogueDAG{
			NumInputs: 2,
			Nodes: []DAGNode{
				{Op: DAGXor, A: 0, B: 1},
				{Op: DAGXor, A: 1, B: 0},
				{Op: DAGAnd, A: 2, B: 3},
			},
			Outputs: []int{4},
		}, 1},
		{"shift pair", EpilogueDAG{
			NumInputs: 1,
			Nodes: []DAGNode{
				{Op: DAGShl, A: 0, Imm: 3},
				{Op: DAGShl, A: 1, Imm: 5},
			},
			Outputs: []int{2},
		}, 1},
		// (a & 0xff) | b<<24 is shl then prmt: prmt has no zero byte to
		// fill the middle with, so it borrows the shifted word's.
		{"or of disjoint bytes", EpilogueDAG{
			NumInputs: 2,
			Nodes: []DAGNode{
				{Op: DAGConst, Imm: 0xff},
				{Op: DAGAnd, A: 0, B: 2},
				{Op: DAGShl, A: 1, Imm: 24},
				{Op: DAGOr, A: 3, B: 4},
			},
			Outputs: []int{5},
		}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			res, err := SynthesizeEpilogueDAG(tc.d, 1)
			if err != nil {
				t.Fatal(err)
			}
			if res.SuperoptInsts != tc.want {
				t.Errorf("SuperoptInsts = %d, want %d: %+v", res.SuperoptInsts, tc.want, res.DAG.Nodes)
			}
		})
	}
}

func TestEpilogueDAGValidate(t *testing.T) {
	for _, d := range []EpilogueDAG{
		{NumInputs: 1, Nodes: []DAGNode{{Op: DAGAnd, A: 0, B: 1}}},
		{NumInputs: 1, Nodes: []DAGNode{{Op: DAGShl, A: 0, Imm: 32}}},
		{NumInputs: 1, Outputs: []int{1}},
		{NumInputs: 2, LoadGroups: [][]int{{0}, {0}}},
		{NumInputs: 1, Nodes: []DAGNode{{Op: numDAGOps}}},
	} {
		if err := d.Validate(); err == nil {
			t.Errorf("Validate(%+v) = nil", d)
		}
	}
}

const waitsPTX = `{
tcgen05.ld.sync.aligned.32x32b.x8.b32 {%v0, %v1}, [%taddr+0];
tcgen05.wait::ld.sync.aligned;
tcgen05.ld.sync.aligned.32x32b.x8.b32 {%v2, %v3}, [%taddr+8];
tcgen05.wait::ld.sync.aligned;
tcgen05.wait::ld.sync.aligned;
mul.rn.f32 %v4, %v0, %v9;
mul.rn.f32 %v5, %v2, %v9;
}`

func TestCoalesceWaits(t *testing.T) {
	got, removed, err := CoalesceWaits(waitsPTX)
	if err != nil {
		t.Fatal(err)
	}
	if removed != 2 {
		t.Errorf("removed %d waits, want 2", removed)
	}
	want := `{
tcgen05.ld.sync.aligned.32x32b.x8.b32 {%v0, %v1}, [%taddr+0];
tcgen05.ld.sync.aligned.32x32b.x8.b32 {%v2, %v3}, [%taddr+8];
tcgen05.wait::ld.sync.aligned;
mul.rn.f32 %v4, %v0, %v9;
mul.rn.f32 %v5, %v2, %v9;
}`
	if got != want {
		t.Errorf("CoalesceWaits =\n%s\nwant\n%s", got, want)
	}
}

func TestCoalesceWaitsKeepsPipelinedWait(t *testing.T) {
	// The first results are read before the text waits for the second
	// load, so the first wait must not move past the second load.
	in := `tcgen05.ld.sync.aligned.32x32b.x2.b32 {%v0, %v1}, [%taddr+0];
tcgen05.wait::ld.sync.aligned;
tcgen05.ld.sync.aligned.32x32b.x2.b32 {%v2, %v3}, [%taddr+8];
mul.rn.f32 %v4, %v0, %v9;
tcgen05.wait::ld.sync.aligned;
mul.rn.f32 %v5, %v2, %v9;`
	got, removed, err := CoalesceWaits(in)
	if err != nil {
		t.Fatal(err)
	}
	if got != in || removed != 0 {
		t.Errorf("CoalesceWaits changed a pipelined schedule (%d removed):\n%s", removed, got)
	}
}

func TestVerifyWaitDataflow(t *testing.T) {
	for _, tc := range []struct {
		name string
		ptx  string
		ok   bool
	}{
		{"ok", waitsPTX, true},
		// The spec's mutant: the wait moved past the first reader.
		{"wait after reader", `tcgen05.ld.sync.aligned.32x32b.x2.b32 {%v0, %v1}, [%taddr];
mul.rn.f32 %v4, %v0, %v9;
tcgen05.wait::ld.sync.aligned;`, false},
		{"overwrite before wait", `tcgen05.ld.sync.aligned.32x32b.x2.b32 {%v0, %v1}, [%taddr];
mov.b32 %v1, 0;
tcgen05.wait::ld.sync.aligned;`, false},
		{"pending at branch", `tcgen05.ld.sync.aligned.32x32b.x2.b32 {%v0, %v1}, [%taddr];
bra L;
tcgen05.wait::ld.sync.aligned;`, false},
		{"pending at label", `tcgen05.ld.sync.aligned.32x32b.x2.b32 {%v0, %v1}, [%taddr];
L:
tcgen05.wait::ld.sync.aligned;`, false},
		{"unrelated register", `tcgen05.ld.sync.aligned.32x32b.x2.b32 {%v0, %v1}, [%taddr];
mul.rn.f32 %v4, %v10, %v9;
tcgen05.wait::ld.sync.aligned;
mul.rn.f32 %v5, %v0, %v9;`, true},
		{"predicated reader", `tcgen05.ld.sync.aligned.32x32b.x2.b32 {%v0, %v1}, [%taddr];
@%p1 mul.rn.f32 %v4, %v1, %v9;
tcgen05.wait::ld.sync.aligned;`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := VerifyWaitDataflow(tc.ptx)
			if (err == nil) != tc.ok {
				t.Fatalf("VerifyWaitDataflow = %v, want ok=%v", err, tc.ok)
			}
		})
	}
}
