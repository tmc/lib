package cudacluster

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudashared"
)

func TestCapabilityOf(t *testing.T) {
	tests := []struct {
		target string
		want   Capability
		dotted string
	}{
		{"sm_80", 80, "8.0"},
		{"sm_86", 86, "8.6"},
		{"sm_90a", 90, "9.0"},
		{"sm_121a", 121, "12.1"},
		{"SM_121A", 121, "12.1"},
	}
	for _, tt := range tests {
		got, err := CapabilityOf(tt.target)
		if err != nil {
			t.Errorf("CapabilityOf(%q) error: %v", tt.target, err)
			continue
		}
		if got != tt.want {
			t.Errorf("CapabilityOf(%q) = %d, want %d", tt.target, got, tt.want)
		}
		if got.String() != tt.dotted {
			t.Errorf("CapabilityOf(%q).String() = %q, want %q", tt.target, got.String(), tt.dotted)
		}
	}
	for _, bad := range []string{"", "sm_", "compute_90", "90"} {
		if _, err := CapabilityOf(bad); err == nil {
			t.Errorf("CapabilityOf(%q) = nil error, want failure", bad)
		}
	}
}

// TestCluster_UnsupportedPartReturnsPending is lane 56's acceptance test, and
// it distinguishes the two answers that a single bool would collapse.
//
// sm_80 was probed and refused, so it is ErrUnsupported. sm_86 is below the
// capability fence, which is also a refusal, but a lexical one. sm_89 appears
// in no feature record at all -- however, it is below the fence too, so the
// fence answers before the table is consulted. The case that actually reaches
// ErrHardwarePending is a target above the fence with no record, which is why
// it is tested explicitly rather than assumed.
func TestCluster_UnsupportedPartReturnsPending(t *testing.T) {
	tests := []struct {
		target string
		want   error
	}{
		{"sm_80", ErrUnsupported},
		{"sm_86", ErrHardwarePending},
		{"sm_89", ErrUnsupported},
		{"sm_98a", ErrHardwarePending}, // above the fence, no record anywhere
		{"sm_121a", nil},
		{"sm_90a", nil},
	}
	for _, tt := range tests {
		err := Check(tt.target)
		switch {
		case tt.want == nil && err != nil:
			t.Errorf("Check(%q) = %v, want nil", tt.target, err)
		case tt.want != nil && !errors.Is(err, tt.want):
			t.Errorf("Check(%q) = %v, want %v", tt.target, err, tt.want)
		}
	}
}

func TestCapabilityLookupErrorIsPending(t *testing.T) {
	err := check("sm_90a", func(string) (cudadevicetable.Target, bool) {
		return cudadevicetable.Target{Target: "sm_90a"}, true
	})
	if !errors.Is(err, ErrHardwarePending) {
		t.Fatalf("check with missing feature = %v, want ErrHardwarePending", err)
	}
}

func TestCluster_PointerFromUnsupportedTargetIsRefused(t *testing.T) {
	if _, err := Open("sm_86"); !errors.Is(err, ErrHardwarePending) {
		t.Fatalf("Open(sm_86) = %v, want ErrHardwarePending", err)
	}
}

func TestPointerCannotCrossClusterScopes(t *testing.T) {
	c1, plan, region := testPlan(t)
	c2, err := Open("sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	_, ptr, err := c1.MapSharedRank("%r1", "%r9", plan, region, "%r2")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c2.EmitLoadU32("%r3", ptr); err == nil {
		t.Fatal("foreign scope load accepted")
	}
}

// TestCluster_ClusterFactIsNotADSMEMFact pins the distinction the device
// table cannot currently make. If someone adds a dsmem fixture and this test
// starts failing, that is the signal to rewrite CheckDSMEM, not to delete it.
func TestCluster_ClusterFactIsNotADSMEMFact(t *testing.T) {
	const target = "sm_121a"
	if err := Check(target); err != nil {
		t.Fatalf("Check(%q) = %v, want nil", target, err)
	}
	err := CheckDSMEM(target)
	if !errors.Is(err, ErrHardwarePending) {
		t.Fatalf("CheckDSMEM(%q) = %v, want ErrHardwarePending", target, err)
	}
	if !strings.Contains(err.Error(), "mapa.shared::cluster") {
		t.Errorf("CheckDSMEM error does not name the unprobed construct: %v", err)
	}
}

func testPlan(t *testing.T) (*Cluster, *cudashared.Plan, cudashared.Region) {
	t.Helper()
	limits := cudashared.Limits{StaticMax: 49152, DynamicMax: 102400, TotalMax: 102400, Source: "test target sm_121a"}
	plan, err := cudashared.NewPlan(limits)
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}
	region, err := plan.Add(cudashared.Request{Name: "tile", Size: 1024, Alignment: 16, FirstStage: 0, LastStage: 0})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	c, err := Open("sm_121a")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return c, plan, region
}

func TestMapSharedRank(t *testing.T) {
	c, plan, region := testPlan(t)
	ptx, p, err := c.MapSharedRank("%r1", "%r9", plan, region, "%r2")
	if err != nil {
		t.Fatalf("MapSharedRank: %v", err)
	}
	if !strings.HasPrefix(ptx, "mapa.shared::cluster.u32 %r1, %r9") || !strings.HasSuffix(ptx, ", %r2;") {
		t.Errorf("MapSharedRank emitted %q", ptx)
	}
	if !p.Valid() || p.Mutable() {
		t.Errorf("read-only mapping: Valid=%v Mutable=%v", p.Valid(), p.Mutable())
	}
	if p.Region().Name != "tile" {
		t.Errorf("Region().Name = %q, want %q", p.Region().Name, "tile")
	}
}

func TestReductionFixtureUsesEmitters(t *testing.T) {
	src, err := ReductionFixture(4, MutantNone)
	if err != nil {
		t.Fatal(err)
	}
	for _, emitted := range []string{
		"mov.u32 %0, smem;",
		"mapa.shared::cluster.u32 %1, %2, %3;",
		"ld.shared::cluster.u32 %0, [%1];",
		"barrier.cluster.arrive;",
		"barrier.cluster.wait;",
	} {
		if !strings.Contains(src, emitted) {
			t.Errorf("generated fixture lacks emitter output %q", emitted)
		}
	}
	mutant, err := ReductionFixture(4, MutantLocalWindow)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(mutant, "mapa.shared::cluster.u32 %1, %2, %4;") {
		t.Fatalf("local-window mutant does not substitute current rank")
	}
}

func TestReductionFixtureRejectsUnknownMutant(t *testing.T) {
	if _, err := ReductionFixture(4, Mutant("bogus")); err == nil {
		t.Fatal("unknown mutant accepted")
	}
}

func TestInheritedLockRejectsPresenceOnlyValue(t *testing.T) {
	_, err := cudalock.Inherited(func(k string) string {
		if k == "GPU_LOCK_HELD" {
			return "1"
		}
		return ""
	})
	if err == nil {
		t.Fatal("presence-only lock value accepted")
	}
}

func mustEmit(s string, err error) string {
	if err != nil {
		panic(err)
	}
	return s
}

// TestCluster_PointerTypeIsDistinct is lane 56's requirement that a cluster
// pointer not be interchangeable with a local shared address. Go enforces it
// at compile time: ClusterPtr is a struct with no exported fields and no string
// conversion, so there is no expression that produces one from an offset. The
// test asserts the runtime half -- that a zero value is refused rather than
// emitting an instruction with an empty operand.
func TestCluster_PointerTypeIsDistinct(t *testing.T) {
	c, _, _ := testPlan(t)
	var zero ClusterPtr
	if zero.Valid() {
		t.Fatal("zero ClusterPtr reports Valid")
	}
	if _, err := c.EmitLoadU32("%r1", zero); err == nil {
		t.Error("EmitLoadU32 through a zero ClusterPtr = nil error, want failure")
	}
	if _, err := c.EmitStoreU32(zero, "%r1"); err == nil {
		t.Error("EmitStoreU32 through a zero ClusterPtr = nil error, want failure")
	}
}

func TestStoreThroughReadOnlyMappingIsRefused(t *testing.T) {
	c, plan, region := testPlan(t)
	_, ro, err := c.MapSharedRank("%r1", "%r9", plan, region, "%r2")
	if err != nil {
		t.Fatalf("MapSharedRank: %v", err)
	}
	if _, err := c.EmitStoreU32(ro, "%r3"); err == nil {
		t.Fatal("store through read-only mapping = nil error, want failure")
	}
	_, rw, err := c.MapSharedRankMut("%r1", "%r9", plan, region, "%r2")
	if err != nil {
		t.Fatalf("MapSharedRankMut: %v", err)
	}
	ptx, err := c.EmitStoreU32(rw, "%r3")
	if err != nil {
		t.Fatalf("store through mutable mapping: %v", err)
	}
	if want := "st.shared::cluster.u32 [%r1], %r3;"; ptx != want {
		t.Errorf("EmitStoreU32 = %q, want %q", ptx, want)
	}
}

func TestMapSharedRankRejectsForeignRegion(t *testing.T) {
	c, plan, _ := testPlan(t)
	foreign := cudashared.Region{Name: "elsewhere", Offset: 4096, Size: 16}
	if _, _, err := c.MapSharedRank("%r1", "%r9", plan, foreign, "%r2"); err == nil {
		t.Fatal("MapSharedRank with a region not in the plan = nil error, want failure")
	}
}

func TestMapSharedRankRejectsSameNameWithDifferentSize(t *testing.T) {
	c, plan, region := testPlan(t)
	region.Size++
	if _, _, err := c.MapSharedRank("%r1", "%r9", plan, region, "%r2"); err == nil {
		t.Fatal("region with mismatched size accepted")
	}
}

func TestEmitReadU32IsTheMappingPlusTheLoad(t *testing.T) {
	c, plan, region := testPlan(t)
	combined, err := c.EmitReadU32("%r1", "%r2", "%r9", plan, region, "%r3")
	if err != nil {
		t.Fatalf("EmitReadU32: %v", err)
	}
	mapPTX, p, err := c.MapSharedRank("%r2", "%r9", plan, region, "%r3")
	if err != nil {
		t.Fatalf("MapSharedRank: %v", err)
	}
	ldPTX, err := c.EmitLoadU32("%r1", p)
	if err != nil {
		t.Fatalf("EmitLoadU32: %v", err)
	}
	want := []string{mapPTX, ldPTX}
	if len(combined) != len(want) {
		t.Fatalf("EmitReadU32 emitted %d instructions, want %d", len(combined), len(want))
	}
	for i := range want {
		if combined[i] != want[i] {
			t.Errorf("EmitReadU32[%d] = %q, want %q", i, combined[i], want[i])
		}
	}
}

// TestClusterSyncMatchesTheAssembledFixture pins the barrier spelling to the
// one cudadevicetable actually assembled, so that a silent change to the
// cuda-oxide spelling fails here rather than at ptxas on a locked device.
func TestClusterSyncMatchesTheAssembledFixture(t *testing.T) {
	got := EmitClusterSync()
	want := []string{"barrier.cluster.arrive;", "barrier.cluster.wait;"}
	if len(got) != len(want) {
		t.Fatalf("EmitClusterSync returned %d instructions, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("EmitClusterSync[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestGeometryRegisterSpelling(t *testing.T) {
	// Every cluster special register carries an underscore after "cluster".
	// Lane 56 spells two of them without one; no assembler accepts that.
	emitted := []string{
		EmitBlockRank("%r1"),
		EmitClusterSize("%r1"),
		mustEmit(EmitClusterCTAID("%r1", DimX)),
		mustEmit(EmitClusterNCTAID("%r1", DimZ)),
	}
	for _, ptx := range emitted {
		if !strings.Contains(ptx, "%cluster_") {
			t.Errorf("%q does not use the %%cluster_ spelling", ptx)
		}
	}
	if got := EmitBlockRank("%r1"); got != "mov.u32 %r1, %cluster_ctarank;" {
		t.Errorf("EmitBlockRank = %q", got)
	}
	if got := mustEmit(EmitClusterCTAID("%r7", DimY)); got != "mov.u32 %r7, %cluster_ctaid.y;" {
		t.Errorf("EmitClusterCTAID = %q", got)
	}
}

func TestGeometryRejectsInvalidDim(t *testing.T) {
	if _, err := EmitClusterCTAID("%r1", Dim(3)); err == nil {
		t.Fatal("CTAID accepted invalid Dim")
	}
	if _, err := EmitClusterNCTAID("%r1", Dim(3)); err == nil {
		t.Fatal("NCTAID accepted invalid Dim")
	}
}

// TestMapSharedRankTakesAnAddressNotAnOffset pins the fix for a defect found by
// reading the emitted text rather than by a failing test: the base operand was
// a bare region offset, and mapa takes an address, so the instruction would not
// have assembled. Nothing in the host-side suite could have caught it, which is
// the point of pinning it here.
func TestMapSharedRankTakesAnAddressNotAnOffset(t *testing.T) {
	c, plan, region := testPlan(t)
	addr, err := EmitSharedAddr("%r9", "smem", plan, region)
	if err != nil {
		t.Fatalf("EmitSharedAddr: %v", err)
	}
	if region.Offset == 0 {
		if want := "mov.u32 %r9, smem;"; addr != want {
			t.Errorf("EmitSharedAddr = %q, want %q", addr, want)
		}
	}
	ptx, _, err := c.MapSharedRank("%r1", "%r9", plan, region, "%r2")
	if err != nil {
		t.Fatalf("MapSharedRank: %v", err)
	}
	if want := "mapa.shared::cluster.u32 %r1, %r9, %r2;"; ptx != want {
		t.Errorf("MapSharedRank = %q, want %q", ptx, want)
	}
	for _, digit := range []string{", 0,", ", 1024,"} {
		if strings.Contains(ptx, digit) {
			t.Errorf("MapSharedRank emitted a bare integer operand: %q", ptx)
		}
	}
	second, err := plan.Add(cudashared.Request{Name: "second", Size: 32, Alignment: 16, FirstStage: 0, LastStage: 0})
	if err != nil {
		t.Fatal(err)
	}
	addr, err = EmitSharedAddr("%r8", "smem", plan, second)
	if err != nil {
		t.Fatal(err)
	}
	if want := "mov.u32 %r8, smem+1024;"; addr != want {
		t.Fatalf("offset address = %q, want %q", addr, want)
	}
	if _, _, err := c.MapSharedRank("%r1", "", plan, region, "%r2"); err == nil {
		t.Error("MapSharedRank with no base operand = nil error, want failure")
	}
}
