package cudasass_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// testLatencies is the latency table the synthetic blocks are checked
// against. It is an assumption of the tests, not a measurement.
var testLatencies = cudasass.Latencies{
	Result: map[string]uint8{"FFMA2": 4, "FFMA": 4, "FADD": 4, "IMAD": 4},
}

// bundles builds bundles from "OPCODE operands | control" lines, where
// control is space-separated key=value fields: s (stall), wr, rd, wait
// (hex mask), reuse (hex), and the bare word y for the yield bit.
func bundles(t *testing.T, lines ...string) []cudasass.Bundle {
	t.Helper()
	var insts []cudasass.Instruction
	for i, line := range lines {
		text, ctl, _ := strings.Cut(line, "|")
		op, operands, _ := strings.Cut(strings.TrimSpace(text), " ")
		c := cudasass.ControlCode{WriteBarrier: cudasass.NoBarrier, ReadBarrier: cudasass.NoBarrier}
		for _, f := range strings.Fields(ctl) {
			k, v, _ := strings.Cut(f, "=")
			n := parseUint(t, v)
			switch k {
			case "s":
				c.Stall = uint8(n)
			case "wr":
				c.WriteBarrier = uint8(n)
			case "rd":
				c.ReadBarrier = uint8(n)
			case "wait":
				c.WaitMask = uint8(n)
			case "reuse":
				c.Reuse = uint8(n)
			case "y":
				c.Yield = true
			default:
				t.Fatalf("bad control field %q", f)
			}
		}
		enc, err := cudasass.EncodeControl(cudasass.Encoding{Lo: 0x1000 + uint64(i), Hi: 0xabc}, c, cudasass.SM121a)
		if err != nil {
			t.Fatal(err)
		}
		insts = append(insts, cudasass.Instruction{Address: uint64(16 * i), Opcode: op, Operands: strings.TrimSpace(operands), Encoding: enc})
	}
	b, err := cudasass.Bundles(cudasass.BasicBlock{Start: 0, End: uint64(16 * len(insts)), Instructions: insts}, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func parseUint(t *testing.T, s string) uint64 {
	t.Helper()
	if s == "" {
		return 0
	}
	var n uint64
	base := uint64(10)
	if h, ok := strings.CutPrefix(s, "0x"); ok {
		s, base = h, 16
	}
	for _, r := range s {
		d := uint64(strings.IndexRune("0123456789abcdef", r))
		if d >= base {
			t.Fatalf("bad number %q", s)
		}
		n = n*base + d
	}
	return n
}

func TestVerifyHazardSafety(t *testing.T) {
	for _, tc := range []struct {
		name  string
		block []string
		kind  cudasass.DepKind
		ok    bool
	}{
		{"ffma2 chain at latency", []string{
			"FFMA2 R4, R1, R2, R3 | s=4",
			"FFMA2 R5, R4, R6, R7 | s=4",
		}, 0, true},
		// The spec's mutant: stall 1 is below the 4-cycle latency.
		{"ffma2 chain under latency", []string{
			"FFMA2 R4, R1, R2, R3 | s=1",
			"FFMA2 R5, R4, R6, R7 | s=4",
		}, cudasass.RAW, false},
		{"latency spread over independent work", []string{
			"FFMA2 R4, R1, R2, R3 | s=2",
			"FFMA2 R9, R1, R2, R3 | s=2",
			"FFMA2 R5, R4, R6, R7 | s=4",
		}, 0, true},
		{"lds.128 waited", []string{
			"LDS.128 R8, [R2] | s=1 wr=0",
			"FFMA R12, R10, R1, R1 | s=4 wait=0x1",
		}, 0, true},
		// The spec's second mutant: the wait for the LDS.128 is dropped,
		// and the reader touches the third register of the quad.
		{"lds.128 wait dropped", []string{
			"LDS.128 R8, [R2] | s=1 wr=0",
			"FFMA R12, R10, R1, R1 | s=4",
		}, cudasass.RAW, false},
		{"wait on the wrong barrier", []string{
			"LDS.128 R8, [R2] | s=1 wr=0",
			"FFMA R12, R10, R1, R1 | s=4 wait=0x2",
		}, cudasass.RAW, false},
		{"overwrite of variable result unwaited", []string{
			"LDS R8, [R2] | s=1 wr=0",
			"FFMA R8, R1, R1, R1 | s=4",
		}, cudasass.WAW, false},
		{"source under read barrier overwritten", []string{
			"STS [R2], R8 | s=1 rd=1",
			"FFMA R8, R1, R1, R1 | s=4",
		}, cudasass.WAR, false},
		{"source under read barrier waited", []string{
			"STS [R2], R8 | s=1 rd=1",
			"FFMA R8, R1, R1, R1 | s=4 wait=0x2",
		}, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := cudasass.VerifyHazardSafety(bundles(t, tc.block...), testLatencies)
			if tc.ok {
				if err != nil {
					t.Fatalf("VerifyHazardSafety = %v, want nil", err)
				}
				return
			}
			var he *cudasass.HazardError
			if !errors.As(err, &he) || he.Kind != tc.kind {
				t.Fatalf("VerifyHazardSafety = %v, want %v hazard", err, tc.kind)
			}
		})
	}
}

func TestVerifyHazardSafetyUnknownLatency(t *testing.T) {
	b := bundles(t, "FFMA R4, R1, R2, R3 | s=4", "FFMA R5, R4, R6, R7 | s=4")
	if err := cudasass.VerifyHazardSafety(b, cudasass.Latencies{}); err == nil {
		t.Fatal("VerifyHazardSafety with no latency entry = nil, want error")
	}
}

// checkOptimized checks the invariants every optimizer output must hold.
func checkOptimized(t *testing.T, in, out []cudasass.Bundle, lat cudasass.Latencies) {
	t.Helper()
	if len(in) != len(out) {
		t.Fatalf("optimizer returned %d bundles for %d", len(out), len(in))
	}
	if err := cudasass.VerifyHazardSafety(out, lat); err != nil {
		t.Fatalf("optimized block fails VerifyHazardSafety: %v", err)
	}
	for i := range in {
		if out[i].Inst.Encoding.Lo != in[i].Inst.Encoding.Lo {
			t.Errorf("instruction %d: Lo changed from %#x to %#x", i, in[i].Inst.Encoding.Lo, out[i].Inst.Encoding.Lo)
		}
		if out[i].Inst.Opcode != in[i].Inst.Opcode || out[i].Inst.Operands != in[i].Inst.Operands {
			t.Errorf("instruction %d changed text", i)
		}
		got, err := cudasass.DecodeControl(out[i].Inst.Encoding, cudasass.SM121a)
		if err != nil || got != out[i].Ctrl {
			t.Errorf("instruction %d: encoding carries %v (%v), bundle says %v", i, got, err, out[i].Ctrl)
		}
		if out[i].Ctrl.Stall > in[i].Ctrl.Stall {
			t.Errorf("instruction %d: stall raised from %d to %d", i, in[i].Ctrl.Stall, out[i].Ctrl.Stall)
		}
	}
}

func TestOptimizeBasicBlockControlPaddedEpilogue(t *testing.T) {
	// A padded FFMA2 epilogue: every stall is 11, the latency is 4. The
	// last result is live out, so the block must still end 4 cycles after
	// it issues.
	in := bundles(t,
		"FFMA2 R4, R1, R2, R3 | s=11",
		"FFMA2 R6, R1, R2, R3 | s=11",
		"FFMA2 R8, R4, R6, R3 | s=11",
		"FFMA2 R10, R8, R2, R3 | s=11",
	)
	out, st, err := cudasass.OptimizeBasicBlockControl(in, testLatencies)
	if err != nil {
		t.Fatal(err)
	}
	checkOptimized(t, in, out, testLatencies)
	var stalls []uint8
	for _, b := range out {
		stalls = append(stalls, b.Ctrl.Stall)
	}
	// R4 is read by the third: 1 + 3 = 4. R6 by the third: 3. R8 by the
	// fourth: 4. R10 is live out and keeps 4.
	if want := []uint8{1, 4, 4, 4}; !bytes.Equal(stalls, want) {
		t.Errorf("stalls = %v, want %v", stalls, want)
	}
	if st.StallCyclesSaved != 44-13 {
		t.Errorf("StallCyclesSaved = %d, want %d", st.StallCyclesSaved, 44-13)
	}
	if st.InstructionsScanned != 4 {
		t.Errorf("InstructionsScanned = %d", st.InstructionsScanned)
	}
}

func TestOptimizeBasicBlockControlLiveIn(t *testing.T) {
	// R1 is read by the second instruction and defined before the block,
	// by a producer this block cannot see. The 6 cycles between entry and
	// the read may be what that producer needs, so they stay.
	in := bundles(t,
		"FFMA R4, R2, R2, R2 | s=6",
		"FFMA R5, R1, R2, R3 | s=4",
	)
	out, _, err := cudasass.OptimizeBasicBlockControl(in, testLatencies)
	if err != nil {
		t.Fatal(err)
	}
	checkOptimized(t, in, out, testLatencies)
	if out[0].Ctrl.Stall != 4 {
		// min(6, the largest latency in the table, 4) = 4.
		t.Errorf("live-in distance stall = %d, want 4", out[0].Ctrl.Stall)
	}
}

func TestOptimizeBasicBlockControlRefusesUnsafeInput(t *testing.T) {
	in := bundles(t,
		"FFMA2 R4, R1, R2, R3 | s=1",
		"FFMA2 R5, R4, R6, R7 | s=4",
	)
	if _, _, err := cudasass.OptimizeBasicBlockControl(in, testLatencies); err == nil {
		t.Fatal("OptimizeBasicBlockControl accepted a block that fails VerifyHazardSafety")
	}
}

func TestOptimizeBasicBlockControlSplitScoreboards(t *testing.T) {
	// Both loads set barrier 0. The first load's reader waits on it, which
	// also waits for the second load, whose reader comes later.
	in := bundles(t,
		"LDS R8, [R2] | s=1 wr=0",
		"LDS R9, [R3] | s=1 wr=0",
		"FFMA R10, R8, R1, R1 | s=4 wait=0x1",
		"FFMA R11, R10, R1, R1 | s=4",
		"FFMA R12, R9, R1, R1 | s=4",
	)
	out, st, err := cudasass.OptimizeBasicBlockControl(in, testLatencies)
	if err != nil {
		t.Fatal(err)
	}
	checkOptimized(t, in, out, testLatencies)
	if st.SplitScoreboards != 1 {
		t.Fatalf("SplitScoreboards = %d, want 1", st.SplitScoreboards)
	}
	nb := out[1].Ctrl.WriteBarrier
	if nb == 0 || nb == cudasass.NoBarrier {
		t.Fatalf("second load still on barrier %d", nb)
	}
	if !out[4].Ctrl.Waits(int(nb)) {
		t.Errorf("reader of the second load waits %#x, want barrier %d", out[4].Ctrl.WaitMask, nb)
	}
	// Moving the barrier without the new wait is what the verifier exists
	// to refuse.
	bad := append([]cudasass.Bundle(nil), out...)
	bad[4].Ctrl.WaitMask = 0
	if err := cudasass.VerifyHazardSafety(bad, testLatencies); err == nil {
		t.Error("split without the new wait passes VerifyHazardSafety")
	}
}

func TestOptimizeBasicBlockControlNoSplitWithoutReader(t *testing.T) {
	// The second load has no reader in the block: a successor may wait on
	// barrier 0 for it, so it must stay there.
	in := bundles(t,
		"LDS R8, [R2] | s=1 wr=0",
		"LDS R9, [R3] | s=1 wr=0",
		"FFMA R10, R8, R1, R1 | s=4 wait=0x1",
	)
	out, st, err := cudasass.OptimizeBasicBlockControl(in, testLatencies)
	if err != nil {
		t.Fatal(err)
	}
	if st.SplitScoreboards != 0 || out[1].Ctrl.WriteBarrier != 0 {
		t.Errorf("load with no reader moved to barrier %d", out[1].Ctrl.WriteBarrier)
	}
}

func TestOptimizeBasicBlockControlReuse(t *testing.T) {
	for _, tc := range []struct {
		name  string
		block []string
		want  uint8
	}{
		{"shared slot 0", []string{"FFMA R4, R1, R2, R3 | s=4 y", "FFMA R5, R1, R6, R7 | s=4 y"}, 0x1},
		{"shared slots 0 and 2", []string{"FFMA R4, R1, R2, R3 | s=4 y", "FFMA R5, R1, R6, R3 | s=4 y"}, 0x5},
		{"different slot", []string{"FFMA R4, R1, R2, R3 | s=4 y", "FFMA R5, R2, R1, R7 | s=4 y"}, 0},
		{"yield clear", []string{"FFMA R4, R1, R2, R3 | s=4", "FFMA R5, R1, R6, R7 | s=4"}, 0},
		{"overwritten by the first", []string{"FFMA R1, R1, R2, R3 | s=4 y", "FFMA R5, R1, R6, R7 | s=4 y"}, 0},
		{"next waits", []string{"FFMA R4, R1, R2, R3 | s=4 y", "FFMA R5, R1, R6, R7 | s=4 y wait=0x1"}, 0},
		{"negated operand", []string{"FFMA R4, -R1, R2, R3 | s=4 y", "FFMA R5, -R1, R6, R7 | s=4 y"}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := bundles(t, tc.block...)
			out, st, err := cudasass.OptimizeBasicBlockControl(in, testLatencies)
			if err != nil {
				t.Fatal(err)
			}
			checkOptimized(t, in, out, testLatencies)
			if out[0].Ctrl.Reuse != tc.want {
				t.Errorf("reuse = %#x, want %#x", out[0].Ctrl.Reuse, tc.want)
			}
			if n := st.ReuseHintsAdded; n != popcount(tc.want) {
				t.Errorf("ReuseHintsAdded = %d, want %d", n, popcount(tc.want))
			}
		})
	}
}

func popcount(x uint8) int {
	n := 0
	for ; x != 0; x &= x - 1 {
		n++
	}
	return n
}

func TestOptimizeBasicBlockControlStickyTensor(t *testing.T) {
	in := bundles(t,
		"HMMA.16816.F32.BF16 R4, R20, R24, R4 | s=2",
		"HMMA.16816.F32.BF16 R8, R20, R28, R8 | s=2",
		"HMMA.16816.F32.BF16 R12, R20, R32, R12 | s=2",
		"FFMA R40, R41, R42, R43 | s=4",
	)
	out, st, err := cudasass.OptimizeBasicBlockControl(in, testLatencies)
	if err != nil {
		t.Fatal(err)
	}
	checkOptimized(t, in, out, testLatencies)
	if st.StickyTensorBursts != 1 {
		t.Errorf("StickyTensorBursts = %d, want 1", st.StickyTensorBursts)
	}
	for i, want := range []bool{true, true, false, false} {
		if out[i].Ctrl.Yield != want {
			t.Errorf("instruction %d yield bit = %v, want %v", i, out[i].Ctrl.Yield, want)
		}
	}
}

// fixtureFunctions parses a SASS fixture.
func fixtureFunctions(t *testing.T, path string) []*cudasass.Function {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	d, err := cudasass.Parse(f)
	if err != nil {
		t.Fatal(err)
	}
	return d.Functions
}

// TestOptimizeFunctionControlFixtures runs the optimizer over every ptxas
// fixture and checks each changed block against the model and each plan for
// control-only changes.
func TestOptimizeFunctionControlFixtures(t *testing.T) {
	lat := cudasass.ObservedLatencies()
	for _, path := range []string{
		"testdata/roundtrip/sm121a-roundtrip.sass",
		"testdata/sm121a-encoding.sass",
		"testdata/sm121a-loop-13.3.sass",
		"testdata/sm121a-loop-13.4.sass",
		"testdata/sm121a-reconvergence.sass",
	} {
		for _, fn := range fixtureFunctions(t, path) {
			plan, err := cudasass.OptimizeFunctionControl(fn, cudasass.SM121a, cudasass.ControlOptions{Latencies: lat})
			if err != nil {
				t.Fatalf("%s %s: %v", path, fn.Name, err)
			}
			if err := cudasass.CheckControlOnly(plan.Plans, cudasass.SM121a); err != nil {
				t.Errorf("%s %s: %v", path, fn.Name, err)
			}
			if plan.Stats.StallCyclesSaved < 0 {
				t.Errorf("%s %s: StallCyclesSaved = %d", path, fn.Name, plan.Stats.StallCyclesSaved)
			}
			for _, b := range cudasass.BasicBlocks(fn) {
				in, err := cudasass.Bundles(b, cudasass.SM121a)
				if err != nil {
					continue
				}
				out, _, err := cudasass.OptimizeBasicBlockControl(in, lat)
				if err != nil {
					continue
				}
				checkOptimized(t, in, out, lat)
			}
		}
	}
}

// TestOptimizeFunctionControlPatchesCubin applies the optimizer's plans to
// the fixture CUBIN and reads them back from the patched bytes.
func TestOptimizeFunctionControlPatchesCubin(t *testing.T) {
	im, err := cudasass.OpenCubin(readRoundTripCubin(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, fn := range fixtureFunctions(t, "testdata/roundtrip/sm121a-roundtrip.sass") {
		plan, err := cudasass.OptimizeFunctionControl(fn, cudasass.SM121a, cudasass.ControlOptions{Latencies: cudasass.ObservedLatencies()})
		if err != nil {
			t.Fatal(err)
		}
		if len(plan.Plans) == 0 {
			t.Fatalf("%s: no control changes", fn.Name)
		}
		p, err := im.Apply(fn.Name, plan.Plans)
		if err != nil {
			t.Fatalf("%s: Apply: %v", fn.Name, err)
		}
		for i, a := range p.Patches {
			lo := binary.LittleEndian.Uint64(p.Bytes[a.FileOffset:])
			hi := binary.LittleEndian.Uint64(p.Bytes[a.FileOffset+8:])
			got := cudasass.Encoding{Lo: lo, Hi: hi}
			if got.Lo != plan.Plans[i].Before.Lo {
				t.Errorf("%s %#x: Lo %#x, want %#x", fn.Name, a.Address, got.Lo, plan.Plans[i].Before.Lo)
			}
			c, err := cudasass.DecodeControl(got, cudasass.SM121a)
			if err != nil || c != plan.Plans[i].Control {
				t.Errorf("%s %#x: patched control %v (%v), want %v", fn.Name, a.Address, c, err, plan.Plans[i].Control)
			}
		}
		// The fixture carries a capmerc mirror of each kernel's text, so
		// the patched image is not offered to a device.
		if err := p.Loadable(); !errors.Is(err, cudasass.ErrMirroredText) {
			t.Errorf("%s: Loadable = %v, want ErrMirroredText", fn.Name, err)
		}
	}
}

func TestOptimizeFunctionControlSamples(t *testing.T) {
	fns := fixtureFunctions(t, "testdata/roundtrip/sm121a-roundtrip.sass")
	var addk *cudasass.Function
	for _, fn := range fns {
		if fn.Name == "addk" {
			addk = fn
		}
	}
	all, err := cudasass.OptimizeFunctionControl(addk, cudasass.SM121a, cudasass.ControlOptions{Latencies: cudasass.ObservedLatencies()})
	if err != nil {
		t.Fatal(err)
	}
	none, err := cudasass.OptimizeFunctionControl(addk, cudasass.SM121a, cudasass.ControlOptions{
		Latencies: cudasass.ObservedLatencies(),
		Samples:   map[uint64]uint64{0x0: 3}, // only the entry block is hot
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(all.Plans) == 0 || len(none.Plans) >= len(all.Plans) {
		t.Errorf("plans with samples on the entry block = %d, without samples %d", len(none.Plans), len(all.Plans))
	}
}

func TestCheckControlOnly(t *testing.T) {
	before := cudasass.Encoding{Lo: 0x0000000607007c35, Hi: 0x041fe2000f8e0000}
	c, _ := cudasass.DecodeControl(before, cudasass.SM121a)
	c.Stall = 3
	after, _ := cudasass.EncodeControl(before, c, cudasass.SM121a)
	ok := cudasass.MutationPlan{Address: 0x50, Before: before, After: after, Control: c}
	if err := cudasass.CheckControlOnly([]cudasass.MutationPlan{ok}, cudasass.SM121a); err != nil {
		t.Fatalf("control-only plan rejected: %v", err)
	}
	bad := ok
	bad.After.Lo ^= 1 << 16 // destination register
	if err := cudasass.CheckControlOnly([]cudasass.MutationPlan{bad}, cudasass.SM121a); !errors.Is(err, cudasass.ErrControlChanged) {
		t.Fatalf("plan changing Lo: %v, want ErrControlChanged", err)
	}
}
