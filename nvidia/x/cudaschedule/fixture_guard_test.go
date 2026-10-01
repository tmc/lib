package cudaschedule_test

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"text/scanner"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// checkFixtureHandoff accepts the checked-in fixture's initialization shape.
// It is a bounded structural check, not a PTX interpreter or race oracle.
func checkFixtureHandoff(name, source string, clean bool) error {
	fail := func(cause string) error { return fmt.Errorf("%s: %s", name, cause) }
	m, err := cudaptx.ParseString(source)
	if err != nil {
		return fail("parse fixture: " + err.Error())
	}
	if len(m.Kernels) != 1 {
		return fail("expected one fixture kernel")
	}
	insts := m.Kernels[0].Instructions
	operands := func(inst cudaptx.Instruction) string { return strings.Join(strings.Fields(inst.Operands), "") }
	find := func(op, args string, start int) int {
		for i := start; i < len(insts); i++ {
			if insts[i].Opcode == op && operands(insts[i]) == args {
				return i
			}
		}
		return -1
	}
	sentinel := find("st.volatile.shared.u32", "[handoff],%r3", 0)
	if sentinel < 3 || sentinel+1 >= len(insts) {
		return fail("missing sentinel initialization")
	}
	value := insts[sentinel-1]
	if value.Opcode != "mov.u32" || operands(value) != "%r3,0" || value.Predicate != "" || insts[sentinel].Predicate != "" {
		return fail("sentinel definition is not zero")
	}
	branch, guard, bar := insts[sentinel-2], insts[sentinel-3], insts[sentinel+1]
	if guard.Opcode != "setp.eq.u32" || operands(guard) != "%p1,%r0,0" || guard.Predicate != "" || branch.Opcode != "bra" || operands(branch) != "INIT_DONE" || branch.Predicate != "@!%p1" {
		return fail("unexpected initialization branch")
	}
	if bar.Opcode != "bar.sync" || operands(bar) != "0" || bar.Predicate != "" {
		return fail("init barrier is not unconditional")
	}
	// Instruction.Line comes from cudaptx. The scanner locates actual label
	// tokens while excluding comments and strings; branch operands are not labels.
	var scan scanner.Scanner
	scan.Init(strings.NewReader(source))
	scan.Mode = scanner.ScanIdents | scanner.ScanStrings | scanner.ScanComments | scanner.SkipComments
	labelLine, labels := 0, 0
	for token := scan.Scan(); token != scanner.EOF; token = scan.Scan() {
		if token != scanner.Ident || scan.TokenText() != "INIT_DONE" {
			continue
		}
		line := scan.Position.Line
		if scan.Scan() == ':' {
			labelLine = line
			labels++
		}
	}
	if labels != 1 || labelLine <= insts[sentinel].Line || labelLine >= bar.Line {
		return fail("INIT_DONE skips the init barrier")
	}
	writer := find("st.volatile.shared.u32", "[handoff],%r1", sentinel+1)
	reader := find("ld.volatile.shared.u32", "%r2,[handoff]", sentinel+1)
	if writer < 0 || reader <= writer {
		return fail("missing writer/reader handoff")
	}
	if writer < 4 || insts[writer].Predicate != "@%p1" || insts[writer-1].Opcode != "setp.eq.u32" || operands(insts[writer-1]) != "%p1,%r0,0" || insts[writer-1].Predicate != "" {
		return fail("writer store is not guarded by tid == 0")
	}
	if insts[writer-4].Opcode != "setp.lt.u32" || operands(insts[writer-4]) != "%p0,%r0,32" || insts[writer-4].Predicate != "" || insts[writer-3].Opcode != "bra" || operands(insts[writer-3]) != "READER" || insts[writer-3].Predicate != "@!%p0" {
		return fail("writer and reader dispatch is not warp uniform")
	}
	if clean && (insts[reader-1].Opcode != "bar.sync" || operands(insts[reader-1]) != "0" || insts[reader-1].Predicate != "") {
		return fail("clean reader barrier is not unconditional")
	}
	if !clean {
		for _, inst := range insts[writer+1 : reader] {
			for _, prefix := range []string{"bar.", "barrier.", "mbarrier.", "membar.", "fence."} {
				if strings.HasPrefix(inst.Opcode, prefix) {
					return fail("race arm contains ordering instruction " + inst.Opcode)
				}
			}
			for _, part := range strings.Split(inst.Opcode, ".") {
				if part == "release" || part == "acquire" || part == "acq_rel" {
					return fail("race arm contains ordering instruction " + inst.Opcode)
				}
			}
		}
	}
	return nil
}

func TestFixtureHandoffGuardsRejectControlAndValueMutants(t *testing.T) {
	for _, name := range []string{"schedule_race.ptx", "schedule_race_clean.ptx"} {
		data, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		source := string(data)
		clean := strings.Contains(name, "clean")
		if err := checkFixtureHandoff(name, source, clean); err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct{ name, old, new, want string }{
			{"barrier skipped", "INIT_DONE:\n    bar.sync 0;", "bar.sync 0;\nINIT_DONE:", "INIT_DONE skips"},
			{"comment label decoy", "INIT_DONE:\n    bar.sync 0;", "// INIT_DONE:\n    bar.sync 0;\nINIT_DONE:", "INIT_DONE skips"},
			{"sentinel overwritten", "mov.u32 %r3, 0;", "mov.u32 %r3, 0;\n    mov.u32 %r3, 1;", "sentinel definition"},
			{"sentinel arithmetic", "mov.u32 %r3, 0;", "mov.u32 %r3, 0;\n    add.u32 %r3, %r3, 1;", "sentinel definition"},
			{"predicated init barrier", "INIT_DONE:\n    bar.sync 0;", "INIT_DONE:\n    @%p1 bar.sync 0;", "init barrier is not unconditional"},
			{"wrong branch predicate", "@!%p1 bra INIT_DONE;", "@%p1 bra INIT_DONE;", "initialization branch"},
			{"divergent writer warp", "setp.lt.u32 %p0, %r0, 32;", "setp.eq.u32 %p0, %r0, 0;", "not warp uniform"},
			{"writer predicate removed", "@%p1 st.volatile.shared.u32 [handoff], %r1;", "st.volatile.shared.u32 [handoff], %r1;", "not guarded"},
			{"wrong writer", "mov.u32 %r1, 1;\n    setp.eq.u32 %p1, %r0, 0;", "mov.u32 %r1, 1;\n    setp.eq.u32 %p1, %r0, 1;", "not guarded"},
		} {
			t.Run(name+"/"+tc.name, func(t *testing.T) {
				if !strings.Contains(source, tc.old) {
					t.Fatal("mutation target missing")
				}
				mutant := strings.Replace(source, tc.old, tc.new, 1)
				if mutant == source {
					t.Fatal("mutation target missing")
				}
				if _, err := cudaptx.ParseString(mutant); err != nil {
					t.Fatalf("mutant does not parse: %v", err)
				}
				err := checkFixtureHandoff(name, mutant, clean)
				if err == nil || !strings.Contains(err.Error(), name) || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("error=%v want fixture and %q", err, tc.want)
				}
			})
		}
	}
}

func TestRaceFixtureRejectsOrderingSemantics(t *testing.T) {
	const name = "schedule_race.ptx"
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	for _, inst := range []string{
		"st.release.cta.shared.u32 [handoff], %r1;",
		"ld.acquire.cta.shared.u32 %r3, [handoff];",
		"atom.release.cta.shared.add.u32 %r3, [handoff], 0;",
		"atom.acquire.cta.shared.add.u32 %r3, [handoff], 0;",
		"atom.acq_rel.cta.shared.add.u32 %r3, [handoff], 0;",
		"red.release.cta.shared.add.u32 [handoff], 0;",
		"membar.cta;",
	} {
		opcode := strings.TrimSuffix(strings.Fields(inst)[0], ";")
		t.Run(opcode, func(t *testing.T) {
			mutant := strings.Replace(source, "    @%p1 st.volatile.shared.u32 [handoff], %r1;", "    @%p1 st.volatile.shared.u32 [handoff], %r1;\n    "+inst, 1)
			if mutant == source {
				t.Fatal("mutation target missing")
			}
			if _, err := cudaptx.ParseString(mutant); err != nil {
				t.Fatalf("mutant does not parse: %v", err)
			}
			err := checkFixtureHandoff(name, mutant, false)
			if err == nil || !strings.Contains(err.Error(), name) || !strings.Contains(err.Error(), "ordering instruction "+opcode) {
				t.Fatalf("error=%v did not name ordering instruction", err)
			}
		})
	}
}

func TestNativeFixturePairDiffersOnlyByReaderBarrier(t *testing.T) {
	race, err := os.ReadFile(filepath.Join("testdata", "schedule_race.ptx"))
	if err != nil {
		t.Fatal(err)
	}
	clean, err := os.ReadFile(filepath.Join("testdata", "schedule_race_clean.ptx"))
	if err != nil {
		t.Fatal(err)
	}
	const barrier = "JOIN:\n    bar.sync 0;\n"
	if strings.Count(string(clean), barrier) != 1 || strings.Replace(string(clean), barrier, "JOIN:\n", 1) != string(race) {
		t.Fatal("clean fixture differs beyond the reader barrier")
	}
	mutant := strings.Replace(string(clean), barrier, "JOIN:\n    @%p1 bar.sync 0;\n", 1)
	if err := checkFixtureHandoff("clean", mutant, true); err == nil || !strings.Contains(err.Error(), "clean reader barrier") {
		t.Fatalf("conditional clean barrier accepted: %v", err)
	}
}
