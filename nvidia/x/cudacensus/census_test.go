package cudacensus

import (
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func TestCensusAggregates(t *testing.T) {
	launches := []Launch{
		{Name: "a", StartNs: 0, EndNs: 100, Grid: "1x1x1", Block: "32x1x1"},
		{Name: "b", StartNs: 100, EndNs: 400, Grid: "2x1x1", Block: "64x1x1"},
		{Name: "a", StartNs: 400, EndNs: 500, Grid: "3x1x1", Block: "32x1x1"},
	}
	got := Census(launches, nil)
	if len(got) != 2 {
		t.Fatalf("entries = %d, want 2", len(got))
	}
	// Sorted by descending device time: b (300 ns) before a (200 ns).
	if got[0].Name != "b" || got[0].DeviceNs != 300 {
		t.Errorf("entry 0 = %+v, want b/300", got[0])
	}
	if got[1].Name != "a" || got[1].DeviceNs != 200 || got[1].Launches != 2 {
		t.Errorf("entry 1 = %+v, want a/200/2", got[1])
	}
	if want := []string{"1x1x1", "3x1x1"}; !equal(got[1].Grids, want) {
		t.Errorf("a grids = %v, want %v", got[1].Grids, want)
	}
	if want := []string{"32x1x1"}; !equal(got[1].Blocks, want) {
		t.Errorf("a blocks = %v, want %v", got[1].Blocks, want)
	}
	for _, e := range got {
		if e.Modeled {
			t.Errorf("%s: Modeled with a nil Coster", e.Name)
		}
	}
}

func TestCensusCoverage(t *testing.T) {
	launches := []Launch{
		{Name: "modeled", EndNs: 700},
		{Name: "unmodeled", EndNs: 300},
	}
	table := Table{"modeled": {Bytes: 4096, FLOPs: 2048, Residency: cudaroofline.DeviceResident}}
	entries := Census(launches, table)

	modeled, total := Coverage(entries)
	if modeled != 700 || total != 1000 {
		t.Errorf("Coverage = %d/%d, want 700/1000", modeled, total)
	}
	for _, e := range entries {
		switch e.Name {
		case "modeled":
			if !e.Modeled || e.Bytes != 4096 || e.FLOPs != 2048 || e.Residency != cudaroofline.DeviceResident {
				t.Errorf("modeled entry = %+v", e)
			}
		case "unmodeled":
			if e.Modeled || e.Bytes != 0 {
				t.Errorf("unmodeled entry = %+v", e)
			}
			if e.Launches != 1 || e.DeviceNs != 300 {
				t.Errorf("unmodeled entry lost its timing: %+v", e)
			}
		}
	}
}

// A kernel whose model covers only some of its launches must not report a
// partial byte total as if it were complete.
func TestPartialCoverageIsNotModeled(t *testing.T) {
	launches := []Launch{
		{Name: "k", Grid: "1x1x1", EndNs: 100},
		{Name: "k", Grid: "2x1x1", EndNs: 100},
	}
	byGrid := CosterFunc(func(l Launch) (Cost, bool) {
		if l.Grid == "1x1x1" {
			return Cost{Bytes: 1024, Residency: cudaroofline.DeviceResident}, true
		}
		return Cost{}, false
	})
	e := Census(launches, byGrid)[0]
	if e.Modeled {
		t.Error("Modeled = true with one launch uncovered")
	}
	if e.Bytes != 0 {
		t.Errorf("Bytes = %d, want 0 for a partially covered kernel", e.Bytes)
	}
	if modeled, total := Coverage([]Entry{e}); modeled != 0 || total != 200 {
		t.Errorf("Coverage = %d/%d, want 0/200", modeled, total)
	}
}

func TestPoints(t *testing.T) {
	m, err := cudaroofline.NewModel("sm_121", cudaroofline.HostPageTableResident)
	if err != nil {
		t.Fatalf("no ceiling for sm_121/host_page_table: %v", err)
	}
	entries := []Entry{
		{Name: "covered", Launches: 1, DeviceNs: 1000, Bytes: 100000, Modeled: true, Residency: cudaroofline.HostPageTableResident},
		{Name: "bare", Launches: 1, DeviceNs: 1000},
		{Name: "instant", Launches: 1, DeviceNs: 0, Bytes: 10, Modeled: true, Residency: cudaroofline.HostPageTableResident},
	}
	pts, err := Points(m, entries)
	if err != nil {
		t.Fatalf("Points: %v", err)
	}
	if len(pts) != 1 {
		t.Fatalf("points = %d, want 1 (unmodeled and zero-duration entries are skipped)", len(pts))
	}
	// 100000 bytes in 1000 ns is 100 GB/s.
	if got := pts[0].AttainedGBps; got != 100 {
		t.Errorf("AttainedGBps = %v, want 100", got)
	}
	// With no FLOPs modeled, intensity is zero and the classification says
	// nothing. This is a documented limitation, asserted so it stays visible.
	if pts[0].Intensity != 0 {
		t.Errorf("Intensity = %v, want 0 with no FLOPs", pts[0].Intensity)
	}
	if got, want := pts[0].Classification, cudaroofline.ClassUnconstrained; got != want {
		t.Errorf("Classification = %q, want %q", got, want)
	}
}

func TestParseTable(t *testing.T) {
	for _, tt := range []struct {
		name string
		in   string
		want Table
		err  string
	}{
		{
			name: "bytes and flops",
			in:   "# comment\n\nk1 device 1024 2048\nk2 host_page_table 512\nk3 unknown 8\n",
			want: Table{
				"k1": {Bytes: 1024, FLOPs: 2048, Residency: cudaroofline.DeviceResident},
				"k2": {Bytes: 512, Residency: cudaroofline.HostPageTableResident},
				"k3": {Bytes: 8, Residency: UnknownResidency},
			},
		},
		{name: "too few fields", in: "k1 device\n", err: "want"},
		{name: "too many fields", in: "k1 device 1 2 3\n", err: "want"},
		{name: "residency omitted", in: "k1 1024 2048\n", err: "residency"},
		{name: "residency misspelled", in: "k1 host 1024\n", err: "residency"},
		{name: "duplicate", in: "k1 device 1\nk1 device 2\n", err: "duplicate"},
		{name: "bad bytes", in: "k1 device x\n", err: "bytes"},
		{name: "bad flops", in: "k1 device 1 x\n", err: "flops"},
		{name: "negative bytes", in: "k1 device -1\n", err: "bytes"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseTable(strings.NewReader(tt.in))
			if tt.err != "" {
				if err == nil || !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("err = %v, want one containing %q", err, tt.err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
			for k, v := range tt.want {
				if got[k] != v {
					t.Errorf("%q = %+v, want %+v", k, got[k], v)
				}
			}
		})
	}
}

// The end-to-end shape: a real capture, a declared table, a census, and an
// honest coverage number. The table in testdata covers three of the ten
// kernels and its numbers are illustrative, so the assertion here is about
// coverage being reported, not about bandwidth being right.
func TestCensusOverFixture(t *testing.T) {
	c, err := ReadDir(fixture)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open("testdata/cost-table.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	table, err := ParseTable(f)
	if err != nil {
		t.Fatal(err)
	}
	tok, _ := c.Span("token")
	entries := Census(c.Window(tok.StartNs, tok.EndNs), table)
	if got, want := len(entries), 10; got != want {
		t.Errorf("kernels = %d, want %d", got, want)
	}
	var modeledKernels int
	for _, e := range entries {
		if e.Modeled {
			modeledKernels++
		}
	}
	if got, want := modeledKernels, len(table); got != want {
		t.Errorf("modeled kernels = %d, want %d", got, want)
	}
	modeled, total := Coverage(entries)
	if modeled <= 0 || modeled >= total {
		t.Errorf("Coverage = %d/%d, want a partial cover", modeled, total)
	}

	// The testdata table states "unknown" residency throughout, honestly, so
	// every modeled entry is refused by Points rather than judged against a
	// ceiling that might be 1.43x wrong.
	m, mErr := cudaroofline.NewModel("sm_121", cudaroofline.HostPageTableResident)
	if mErr != nil {
		t.Fatal(mErr)
	}
	{
		pts, pErr := Points(m, entries)
		if pErr == nil {
			t.Error("Points: want an error for the unknown-residency table")
		}
		if len(pts) != 0 {
			t.Errorf("points = %d, want 0 with no residency stated", len(pts))
		}
	}

	// The busiest kernel runs under two different grids, which is exactly the
	// case a per-name Table cannot describe.
	top := entries[0]
	if len(top.Grids) < 2 {
		t.Errorf("top kernel %s has grids %v, want more than one", top.Name, top.Grids)
	}
	if top.Modeled {
		t.Errorf("top kernel %s is modeled by the illustrative table; it should not be", top.Name)
	}
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// The seam this package exists to not get wrong: a byte count whose residency
// is unstated, or stated as the other path, must not be divided into this
// model's ceiling. On the measured GB10 the two ceilings differ by 1.43x, so a
// silent default would misstate headroom by that factor with nothing to show
// for it.
func TestPointsRefusesWrongResidency(t *testing.T) {
	m, err := cudaroofline.NewModel("sm_121", cudaroofline.HostPageTableResident)
	if err != nil {
		t.Fatalf("no ceiling for sm_121/host_page_table: %v", err)
	}
	entries := []Entry{
		{Name: "mapped", DeviceNs: 1000, Bytes: 100000, Modeled: true, Residency: cudaroofline.HostPageTableResident},
		{Name: "unstated", DeviceNs: 1000, Bytes: 100000, Modeled: true},
		{Name: "onDevice", DeviceNs: 1000, Bytes: 100000, Modeled: true, Residency: cudaroofline.DeviceResident},
	}
	pts, err := Points(m, entries)
	if err == nil {
		t.Fatal("Points: want an error naming the refused entries")
	}
	for _, want := range []string{"unstated", "onDevice", "device", "host_page_table"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name %q", err, want)
		}
	}
	// The matching entry is still returned, so a caller who reads the error
	// can proceed with what was judgeable.
	if len(pts) != 1 || pts[0].Name != "mapped" {
		t.Fatalf("points = %+v, want just mapped", pts)
	}
	// And a caller who ignores the error gets fewer points, never a point
	// computed against the wrong ceiling.
	for _, p := range pts {
		if p.Name == "onDevice" || p.Name == "unstated" {
			t.Errorf("%s was evaluated against the %s ceiling", p.Name, m.Ceiling.Residency)
		}
	}
}

// A residency the model states inconsistently across one kernel's launches is
// no residency at all.
func TestDisagreeingResidencyIsUnknown(t *testing.T) {
	launches := []Launch{
		{Name: "k", Grid: "1x1x1", EndNs: 100},
		{Name: "k", Grid: "2x1x1", EndNs: 100},
	}
	mixed := CosterFunc(func(l Launch) (Cost, bool) {
		if l.Grid == "1x1x1" {
			return Cost{Bytes: 1024, Residency: cudaroofline.DeviceResident}, true
		}
		return Cost{Bytes: 1024, Residency: cudaroofline.HostPageTableResident}, true
	})
	e := Census(launches, mixed)[0]
	if !e.Modeled {
		t.Error("Modeled = false; both launches were covered")
	}
	if e.Residency != UnknownResidency {
		t.Errorf("Residency = %q, want unknown for disagreeing launches", e.Residency)
	}

	m, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatalf("no ceiling: %v", err)
	}
	pts, err := Points(m, []Entry{e})
	if err == nil || len(pts) != 0 {
		t.Errorf("Points = %v, %v; want no points and an error", pts, err)
	}
}

// UnknownResidency must not be equal to either real path, or "unset" would
// quietly mean "device".
func TestUnknownResidencyIsDistinct(t *testing.T) {
	if UnknownResidency == cudaroofline.DeviceResident {
		t.Error("UnknownResidency == DeviceResident")
	}
	if UnknownResidency == cudaroofline.HostPageTableResident {
		t.Error("UnknownResidency == HostPageTableResident")
	}
	var zero Cost
	if zero.Residency != UnknownResidency {
		t.Errorf("zero Cost residency = %q, want unknown", zero.Residency)
	}
}

// The exact hazard, in the direction it actually occurs: ds4's weights are
// mmap'd, so their bytes are HostPageTableResident, and the mistake to prevent
// is dividing them into the device-resident GB10 ceiling. Both ceilings are
// positive numbers, so the quotient looks plausible either way; only this
// check distinguishes them.
func TestMappedBytesAgainstDeviceCeiling(t *testing.T) {
	device, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatalf("sm_121 device ceiling: %v", err)
	}
	mapped, err := cudaroofline.NewModel("sm_121", cudaroofline.HostPageTableResident)
	if err != nil {
		t.Fatalf("sm_121 host_page_table ceiling: %v", err)
	}
	// The ceilings must actually differ, or this test proves nothing.
	if device.Ceiling.PeakMemoryGBps == mapped.Ceiling.PeakMemoryGBps {
		t.Fatalf("both ceilings are %v GB/s; the mismatch cannot matter",
			device.Ceiling.PeakMemoryGBps)
	}

	// One kernel reading 9.6 GB of mmap'd weights, as ds4 decode does.
	entries := []Entry{{
		Name: "weights_matvec", Launches: 1, DeviceNs: 60_000_000,
		Bytes: 9_601_439_048, Modeled: true,
		Residency: cudaroofline.HostPageTableResident,
	}}

	pts, err := Points(device, entries)
	if err == nil {
		t.Error("Points against the device ceiling: want an error")
	}
	if len(pts) != 0 {
		t.Fatalf("points = %d, want 0: mapped bytes must not be judged against the device ceiling", len(pts))
	}

	// Against the right ceiling the same entry evaluates.
	pts, err = Points(mapped, entries)
	if err != nil {
		t.Fatalf("Points against the host_page_table ceiling: %v", err)
	}
	if len(pts) != 1 {
		t.Fatalf("points = %d, want 1", len(pts))
	}

	// Coverage is fully satisfied either way, which is why it is not
	// sufficient on its own; EvaluatedCoverage is what separates them.
	if modeled, total := Coverage(entries); modeled != total {
		t.Errorf("Coverage = %d/%d, want full", modeled, total)
	}
	if ev, total := EvaluatedCoverage(device, entries); ev != 0 || total != 60_000_000 {
		t.Errorf("EvaluatedCoverage(device) = %d/%d, want 0/60000000", ev, total)
	}
	if ev, total := EvaluatedCoverage(mapped, entries); ev != total {
		t.Errorf("EvaluatedCoverage(mapped) = %d/%d, want full", ev, total)
	}
}

// Census must carry the Coster's residency onto the Entry. This fails if the
// propagation is removed.
func TestCensusPropagatesResidency(t *testing.T) {
	for _, want := range []cudaroofline.Residency{
		cudaroofline.DeviceResident,
		cudaroofline.HostPageTableResident,
		UnknownResidency,
	} {
		launches := []Launch{{Name: "k", EndNs: 100}, {Name: "k", EndNs: 100}}
		coster := CosterFunc(func(Launch) (Cost, bool) {
			return Cost{Bytes: 512, Residency: want}, true
		})
		got := Census(launches, coster)[0]
		if got.Residency != want {
			t.Errorf("Residency = %q, want %q", got.Residency, want)
		}
	}
}
