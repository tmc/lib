package cudaschedule_test

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func TestNativeScheduleFixtureAnalyzes(t *testing.T) {
	for _, fixture := range []struct {
		name         string
		wantBarriers int
	}{
		{"schedule_race.ptx", 1},
		{"schedule_race_clean.ptx", 2},
	} {
		ptx, err := os.ReadFile(filepath.Join("testdata", fixture.name))
		if err != nil {
			t.Fatal(err)
		}
		sites, err := cudaschedule.Analyze(string(ptx))
		if err != nil {
			t.Fatalf("Analyze %s: %v", fixture.name, err)
		}
		barriers := 0
		for _, site := range sites {
			if site.Kind == cudaschedule.SiteBarrier {
				barriers++
			}
		}
		if barriers != fixture.wantBarriers {
			t.Fatalf("%s barrier sites=%d, want %d (sites=%+v)", fixture.name, barriers, fixture.wantBarriers, sites)
		}
		for seed := uint64(17); seed < 49; seed++ {
			rewrite, err := cudaschedule.Perturb(string(ptx), cudaschedule.Options{Seed: seed, Intensity: 2})
			if err != nil {
				t.Fatal(err)
			}
			if rewrite.Report.SitesInjected > 0 && !strings.Contains(rewrite.PTX, "nanosleep.u32") {
				t.Fatalf("%s seed %d reported injection without nanosleep: %+v", fixture.name, seed, rewrite.Report)
			}
		}
	}
}

func TestNativeRaceFixturesInitializeHandoffAndExposeReversingDraw(t *testing.T) {
	fixtures := []struct {
		name      string
		wantClean bool
	}{
		{name: "schedule_race.ptx"},
		{name: "schedule_race_clean.ptx", wantClean: true},
	}
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("testdata", fixture.name))
			if err != nil {
				t.Fatal(err)
			}
			module, err := cudaptx.ParseString(string(data))
			if err != nil {
				t.Fatal(err)
			}
			if len(module.Kernels) != 1 {
				t.Fatalf("fixture has %d kernels, want one", len(module.Kernels))
			}
			instructions := module.Kernels[0].Instructions
			find := func(predicate func(cudaptx.Instruction) bool, start int) int {
				for i := start; i < len(instructions); i++ {
					if predicate(instructions[i]) {
						return i
					}
				}
				return -1
			}
			initValue := find(func(inst cudaptx.Instruction) bool {
				return inst.Opcode == "mov.u32" && strings.Contains(inst.Operands, "%r3, 0")
			}, 0)
			sentinelStore := find(func(inst cudaptx.Instruction) bool {
				return inst.Opcode == "st.volatile.shared.u32" && strings.Contains(inst.Operands, "[handoff], %r3")
			}, 0)
			initBarrier := find(func(inst cudaptx.Instruction) bool { return inst.Opcode == "bar.sync" }, sentinelStore+1)
			writerValue := find(func(inst cudaptx.Instruction) bool {
				return inst.Opcode == "mov.u32" && strings.Contains(inst.Operands, "%r1, 1")
			}, initBarrier+1)
			writerStore := find(func(inst cudaptx.Instruction) bool {
				return inst.Opcode == "st.volatile.shared.u32" && strings.Contains(inst.Operands, "[handoff], %r1")
			}, writerValue+1)
			readerLoad := find(func(inst cudaptx.Instruction) bool {
				return inst.Opcode == "ld.volatile.shared.u32" && strings.Contains(inst.Operands, "[handoff]")
			}, writerStore+1)
			if initValue < 0 || sentinelStore < 0 || initBarrier < 0 || writerValue < 0 || writerStore < 0 || readerLoad < 0 || !(initValue < sentinelStore && sentinelStore < initBarrier && initBarrier < writerValue && writerValue < writerStore && writerStore < readerLoad) {
				t.Fatalf("sentinel initialization/store/barrier/data access order invalid: value=%d sentinel=%d barrier=%d writer-value=%d writer=%d reader=%d", initValue, sentinelStore, initBarrier, writerValue, writerStore, readerLoad)
			}
			barrierBetween := false
			orderingBetween := false
			for _, inst := range instructions[writerStore+1 : readerLoad] {
				barrierBetween = barrierBetween || inst.Opcode == "bar.sync"
				for _, prefix := range []string{"bar.", "barrier.", "mbarrier.", "membar.", "fence."} {
					orderingBetween = orderingBetween || strings.HasPrefix(inst.Opcode, prefix)
				}
			}
			if barrierBetween != fixture.wantClean {
				t.Fatalf("writer-to-reader barrier=%t, want %t", barrierBetween, fixture.wantClean)
			}
			if orderingBetween != fixture.wantClean {
				t.Fatalf("writer-to-reader ordering instruction=%t, want %t", orderingBetween, fixture.wantClean)
			}
			if !strings.Contains(string(data), "reader work makes the pristine writer likely to publish first") || !strings.Contains(string(data), "writer's shared store can reverse this order") {
				t.Fatal("fixture does not document the pristine bias and reversing perturbation")
			}
			ptx := string(data)
			for _, fragment := range []string{
				"setp.eq.u32 %p1, %r0, 0;",
				"@!%p1 bra INIT_DONE;",
				"setp.eq.u32 %p0, %r0, 0;",
				"@!%p0 bra READER;",
				"mov.u64 %rd2, %globaltimer;",
				"mov.u64 %rd3, %globaltimer;",
				"sub.u64 %rd4, %rd3, %rd2;",
				"setp.lt.u64 %p2, %rd4, 8000;",
				"@%p2 bra READER_DELAY;",
				"st.volatile.global.u64 [%rd0+8], %rd2;",
				"st.volatile.global.u64 [%rd0+16], %rd3;",
			} {
				if !strings.Contains(ptx, fragment) {
					t.Fatalf("fixture does not preserve single-writer/reader-bias fragment %q", fragment)
				}
			}
			if err := checkReaderTimerBlock(fixture.name, ptx); err != nil {
				t.Fatal(err)
			}
		})
	}
	nativeTest, err := os.ReadFile("native_test.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(nativeTest), "if got[0] != 1") {
		t.Fatal("native output check no longer compares the handoff against 1")
	}

	for _, fragment := range []string{
		"checkBaseline(clean)", "checkBaseline(first)", "checkBaseline(second)",
		"start, end := got[1], got[2]", "elapsed := end - start",
		"if end < start", "if elapsed < scheduleReaderDelayThreshold",
		"READER_START_NS: %d", "READER_END_NS: %d", "READER_DELAY_NS: %d",
	} {
		if !strings.Contains(string(nativeTest), fragment) {
			t.Fatalf("native_test.go missing %q", fragment)
		}
	}

	raceData, err := os.ReadFile(filepath.Join("testdata", "schedule_race.ptx"))
	if err != nil {
		t.Fatal(err)
	}
	sites, err := cudaschedule.Analyze(string(raceData))
	if err != nil {
		t.Fatal(err)
	}
	foundWriterDelay := false
	for seed := uint64(17); seed < 49; seed++ {
		rewrite, err := cudaschedule.Perturb(string(raceData), cudaschedule.Options{Seed: seed, Intensity: 2})
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		for _, decision := range rewrite.Report.Decisions {
			if decision.BeforeNs > 0 && strings.Contains(decision.Site.Text, "st.volatile.shared.u32 [handoff], %r1") {
				foundWriterDelay = true
			}
		}
	}
	if !foundWriterDelay {
		t.Fatalf("sweep seeds 17..48 never delay the writer store; sites=%+v", sites)
	}
}

// ALL_TEN_KINDS_PTX contains at least one instruction for each of the 10 site kinds in source order.
const ALL_TEN_KINDS_PTX = `.version 8.7
.target sm_90a
.address_size 64

.visible .entry all_ten_kinds(
    .param .u64 data
)
{
    .reg .pred %p0;
    .reg .b32 %r0;
    .reg .b64 %rd0;

L_loop:
    atom.global.add.u32 %r0, [%rd0], 1;
    red.global.add.u32 [%rd0], %r0;
    bar.sync 0;
    fence.sc.gpu;
    ld.global.acquire.u32 %r0, [%rd0];
    shfl.sync.idx.b32 %r0, %r0, 0, 31;
    cp.async.ca.shared.global [%r0], [%rd0], 4;
    griddepcontrol.launch_dependents;
    tensormap.replace.tile.global_address.global.b1024.b64 [%rd0], %rd0;
    @%p0 bra L_loop;
    ret;
}
`

func TestAnalyze_AllTenKindsClassified(t *testing.T) {
	sites, err := cudaschedule.Analyze(ALL_TEN_KINDS_PTX)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	expectedKinds := []cudaschedule.SiteKind{
		cudaschedule.SiteAtomic,
		cudaschedule.SiteReduction,
		cudaschedule.SiteBarrier,
		cudaschedule.SiteFence,
		cudaschedule.SiteOrderedMemory,
		cudaschedule.SiteWarpCollective,
		cudaschedule.SiteAsyncProxy,
		cudaschedule.SiteGridDependency,
		cudaschedule.SiteTensorMapMutation,
		cudaschedule.SiteBackedge,
	}

	if len(sites) != len(expectedKinds) {
		t.Fatalf("Analyze returned %d sites, want %d", len(sites), len(expectedKinds))
	}

	for i, expected := range expectedKinds {
		site := sites[i]
		if site.Ordinal != i {
			t.Errorf("site %d: Ordinal = %d, want %d", i, site.Ordinal, i)
		}
		if site.Kind != expected {
			t.Errorf("site %d (%s): Kind = %v (%s), want %v (%s)",
				i, site.Head, site.Kind, site.Kind, expected, expected)
		}
		if site.Span.Start >= site.Span.End {
			t.Errorf("site %d (%s): invalid span [%d, %d)", i, site.Head, site.Span.Start, site.Span.End)
		}
		if site.Callable != "all_ten_kinds" {
			t.Errorf("site %d: Callable = %q, want %q", i, site.Callable, "all_ten_kinds")
		}
		if expected == cudaschedule.SiteBackedge {
			if site.BlockIndex == nil {
				t.Errorf("site %d: expected non-nil BlockIndex for Backedge", i)
			}
		}
	}
}

func TestAnalyze_FailClosedOnMalformedBranch(t *testing.T) {
	const malformedBranchPTX = `.version 8.7
.target sm_90
.address_size 64

.visible .entry malformed_loop()
{
    .reg .pred %p0;
    @%p0 bra NonExistentTargetLabel;
    ret;
}
`
	sites, err := cudaschedule.Analyze(malformedBranchPTX)
	if err == nil {
		t.Fatalf("Analyze succeeded on unresolvable branch, want error; returned sites: %+v", sites)
	}
	if !strings.Contains(err.Error(), "NonExistentTargetLabel") {
		t.Errorf("error %q does not name missing target label", err)
	}
}

func TestAnalyze_UnrecognizedOrderingOpcode(t *testing.T) {
	// bar.unknown_sync is a hypothetical unrecognized barrier opcode.
	const unrecognizedPTX = `.version 8.7
.target sm_90
.address_size 64

.visible .entry bad_barrier()
{
    bar.unknown_sync 0;
    ret;
}
`
	_, err := cudaschedule.Analyze(unrecognizedPTX)
	if err == nil {
		t.Fatalf("Analyze succeeded on unrecognized ordering opcode, want error")
	}
	if !strings.Contains(err.Error(), "unrecognized ordering-sensitive opcode") {
		t.Errorf("expected error about unrecognized ordering opcode, got: %v", err)
	}
}

func TestAnalyze_BrxIndexBackedge(t *testing.T) {
	const ptx = `.version 8.7
.target sm_90
.address_size 64
.visible .entry brx_loop(){
L_loop:
    atom.global.add.u32 %r, [%rd], 1;
    brx.idx %r, table;
L_exit:
    ret;
table: .branchtargets L_loop, L_exit;
}
`
	sites, err := cudaschedule.Analyze(ptx)
	if err != nil {
		t.Fatal(err)
	}
	if len(sites) != 2 || sites[1].Kind != cudaschedule.SiteBackedge || sites[1].BlockIndex == nil {
		t.Fatalf("brx.idx sites = %+v, want atom and backedge", sites)
	}
}

func TestAnalyze_BrxUnknownTargetLabelFailsClosed(t *testing.T) {
	const ptx = `.version 8.7
.target sm_90
.address_size 64
.visible .entry brx_bad(){
    brx.idx %r, table;
    ret;
table: .branchtargets L_missing;
}
`
	if _, err := cudaschedule.Analyze(ptx); err == nil || !strings.Contains(err.Error(), `branch table "table" targets unknown label "L_missing"`) {
		t.Fatalf("Analyze unknown branch-table label error = %v", err)
	}
}

func TestAnalyze_BrxForwardOnlyAndForwardDeclaration(t *testing.T) {
	const ptx = `.version 8.7
.target sm_90
.address_size 64
.visible .entry brx_forward(){
    brx.idx %r, table;
L_next:
    ret;
table: .branchtargets L_next;
}
`
	sites, err := cudaschedule.Analyze(ptx)
	if err != nil {
		t.Fatal(err)
	}
	if len(sites) != 0 {
		t.Fatalf("forward-only branch table sites = %+v, want none", sites)
	}
}

func TestAnalyze_RejectsMalformedModules(t *testing.T) {
	tests := []struct {
		name string
		ptx  string
	}{
		{"empty", ""},
		{"not ptx", "this is not ptx {{{"},
		{"missing version", `.target sm_90
.address_size 64
.visible .entry foo(){ ret; }`},
		{"missing target", `.version 8.7
.address_size 64
.visible .entry foo(){ ret; }`},
		{"unterminated", `.version 8.7
.target sm_90
.visible .entry broken(){ atom.global.add.u32 %r,[%rd],1;`},
		{"no callable", `.version 8.7
.target sm_90
.address_size 64`},
		{"extern declarations only", `.version 8.7
.target sm_90
.address_size 64
.extern .func foo();`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if sites, err := cudaschedule.Analyze(tt.ptx); err == nil {
				t.Fatalf("Analyze = %v, %v; want error", sites, err)
			}
		})
	}
}

func TestAnalyze_LdRelaxedIsOrderedMemory(t *testing.T) {
	const ptx = `.version 8.7
.target sm_90
.address_size 64
.visible .entry relaxed(){
    ld.relaxed.global.u32 %r, [%rd];
    ret;
}
`
	sites, err := cudaschedule.Analyze(ptx)
	if err != nil {
		t.Fatal(err)
	}
	if len(sites) != 1 || sites[0].Kind != cudaschedule.SiteOrderedMemory {
		t.Fatalf("ld.relaxed sites = %+v, want one OrderedMemory", sites)
	}
}

func TestAnalyze_PredicatedAtomicPreservesPredicate(t *testing.T) {
	const ptx = `.version 8.7
.target sm_90
.address_size 64
.visible .entry predicated(){
    @%p atom.global.add.u32 %r, [%rd], 1;
    ret;
}
`
	sites, err := cudaschedule.Analyze(ptx)
	if err != nil {
		t.Fatal(err)
	}
	if len(sites) != 1 || sites[0].Kind != cudaschedule.SiteAtomic || sites[0].Predicate != "@%p" {
		t.Fatalf("predicated atom site = %+v, want predicate @%%p", sites)
	}
}

func TestAnalyze_OrderingOpcodeClassesAreClosed(t *testing.T) {
	tests := []struct {
		opcode  string
		stmt    string
		kind    cudaschedule.SiteKind
		wantErr bool
	}{
		{"bar.cta.sync", "bar.cta.sync 0;", cudaschedule.SiteBarrier, false},
		{"bar.cta.arrive", "bar.cta.arrive 0;", cudaschedule.SiteBarrier, false},
		{"bar.cta.red", "bar.cta.red.popc.u32 %r, 0, %p;", cudaschedule.SiteBarrier, false},
		{"multimem.ld_reduce.global.add.u32", "multimem.ld_reduce.global.add.u32 %r, [%rd], %r;", 0, true},
		{"st.async.shared::cta.b32", "st.async.shared::cta.b32 [%rd], %r;", cudaschedule.SiteAsyncProxy, false},
		{"red.async.global.add.u32", "red.async.global.add.u32 [%rd], %r;", cudaschedule.SiteReduction, false},
		{"tensormap.cp_fenceproxy.global.shared::cta", "tensormap.cp_fenceproxy.global.shared::cta;", cudaschedule.SiteAsyncProxy, false},
	}
	for _, tt := range tests {
		t.Run(tt.opcode, func(t *testing.T) {
			ptx := `.version 8.7
.target sm_90
.address_size 64
.visible .entry opcode(){
    ` + tt.stmt + `
    ret;
}
`
			sites, err := cudaschedule.Analyze(ptx)
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), "unrecognized ordering-sensitive opcode") {
					t.Fatalf("Analyze error = %v, want fail-closed ordering error", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(sites) != 1 || sites[0].Kind != tt.kind {
				t.Fatalf("sites = %+v, want one %v", sites, tt.kind)
			}
		})
	}
}

const readerTimerBlock = `mov.u64 %rd2, %globaltimer;
READER_DELAY:
mov.u64 %rd3, %globaltimer;
sub.u64 %rd4, %rd3, %rd2;
setp.lt.u64 %p2, %rd4, 8000;
@%p2 bra READER_DELAY;`

func checkReaderTimerBlock(name, ptx string) error {
	var lines []string
	for _, line := range strings.Split(ptx, "\n") {
		lines = append(lines, strings.TrimSpace(line))
	}
	if !strings.Contains(strings.Join(lines, "\n"), readerTimerBlock) {
		return fmt.Errorf("%s: reader timer block is not contiguous", name)
	}
	return nil
}

func TestReaderTimerBlockRejectsInsertedInstructions(t *testing.T) {
	for _, name := range []string{"schedule_race.ptx", "schedule_race_clean.ptx"} {
		data, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		ptx := string(data)
		for _, line := range strings.Split(readerTimerBlock, "\n")[:5] {
			for _, instruction := range []string{
				"mov.u64 %rd4, 9000;",
				"add.u64 %rd3, %rd2, 9000;",
			} {
				mutant := strings.Replace(ptx, line, line+"\n    "+instruction, 1)
				err := checkReaderTimerBlock(name, mutant)
				if err == nil || !strings.Contains(err.Error(), name) {
					t.Fatalf("%s: inserted %q after %q accepted or wrong diagnostic: %v", name, instruction, line, err)
				}
			}
		}
	}
}
