package cudaschedule_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

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
.branchtargets table, L_loop, L_exit;
L_loop:
    atom.global.add.u32 %r, [%rd], 1;
    brx.idx %r, table;
L_exit:
    ret;
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
.branchtargets table, L_missing;
    brx.idx %r, table;
    ret;
}
`
	if _, err := cudaschedule.Analyze(ptx); err == nil || !strings.Contains(err.Error(), `branch table "table" targets unknown label "L_missing"`) {
		t.Fatalf("Analyze unknown branch-table label error = %v", err)
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
