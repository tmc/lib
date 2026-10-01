package traceanalysis

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestScopedUnionsAndMissingDelivery(t *testing.T) {
	records := timeline.Records{Activities: []timeline.Activity{{Journal: "p", Subtype: "kernel", Start: 0, End: 10, Device: 0, Sequence: 1}, {Journal: "p", Subtype: "kernel", Start: 5, End: 15, Device: 0, Sequence: 2}, {Journal: "p", Subtype: "memcpy", Start: 20, End: 30, Device: 0}, {Journal: "q", Subtype: "kernel", Start: 0, End: 10, Device: 0}, {Journal: "p", Subtype: "kernel", Start: 0, End: 7, Device: 1}}}
	s, err := New(records, Options{CollectorComplete: true})
	if err != nil {
		t.Fatal(err)
	}
	r := s.Summary()
	if len(r.Windows) != 3 || r.Windows[0].UnionNS != 25 || len(r.Windows[0].Gaps) != 1 || r.Windows[0].Gaps[0].Start != 15 || r.Windows[0].Gaps[0].End != 20 {
		t.Fatalf("windows=%+v", r.Windows)
	}
	if r.Dependencies.State != "partial" || len(r.Dependencies.CriticalPath) != 0 {
		t.Fatalf("dependencies=%+v", r.Dependencies)
	}
	query, err := s.Query("delivery-cadence")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(fmt.Sprint(query), "request") {
		t.Fatalf("cadence=%v", query)
	}
	if r.Overview.Hierarchy.Metrics.KernelSumNanos != 37 {
		t.Fatalf("kernel sum=%d", r.Overview.Hierarchy.Metrics.KernelSumNanos)
	}
}
func TestEndpointsAndCollectorFailure(t *testing.T) {
	token := 0
	delivered, _ := marker.Encode(marker.Metadata{Phase: "decode", Token: &token, Event: "delivered"})
	eos, _ := marker.Encode(marker.Metadata{Phase: "decode", Token: &token, Event: "eos"})
	records := timeline.Records{Markers: []timeline.Marker{{Journal: "p", Name: delivered, Flags: timeline.MarkerInstantaneous, Time: 30, Sequence: 8}, {Journal: "p", Name: eos, Flags: timeline.MarkerInstantaneous, Time: 40, Sequence: 9}, {Journal: "p", Name: delivered, Flags: timeline.MarkerStart, Time: 10}}}
	s, err := New(records, Options{CollectorComplete: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Summary().Endpoints) != 2 || s.Summary().Endpoints[0].Source.Sequence != 8 {
		t.Fatal(s.Summary().Endpoints)
	}
	failed, err := New(records, Options{CollectorFailure: "dropped records"})
	if err != nil {
		t.Fatal(err)
	}
	if len(failed.Summary().Endpoints) != 0 || failed.Summary().Counts["marker"] != 3 {
		t.Fatal("partial records qualified")
	}
	query, err := failed.Query("recorded-gaps")
	if err != nil || !strings.Contains(fmt.Sprint(query), "unavailable") {
		t.Fatalf("failed query=%v %v", query, err)
	}
	if _, err := s.Query("sql arbitrary"); err == nil {
		t.Fatal("accepted arbitrary query")
	}
	if _, err := New(records, Options{Domain: "other"}); err == nil {
		t.Fatal("accepted other domain")
	}
}
func Example() {
	s, _ := New(timeline.Records{}, Options{})
	fmt.Println(s.Summary().Gates["correctness"]) // Output: unverified
}
func ExampleCatalog() {
	fmt.Println(Catalog()[0].Schema) // Output: cudaprof.inventory.v1
}
func ExampleSession_Query() {
	s, _ := New(timeline.Records{}, Options{})
	_, err := s.Query("unknown")
	fmt.Println(err) // Output: trace query: unknown name "unknown"
}

func TestKernelDistributionsAndBounds(t *testing.T) {
	records := timeline.Records{Activities: []timeline.Activity{{Journal: "p", Subtype: "kernel", Name: "gemv", Start: 1, End: 11, Grid: [3]int{2, 1, 1}}, {Journal: "p", Subtype: "kernel", Name: "gemv", Start: 20, End: 40, Grid: [3]int{2, 1, 1}}}}
	session, err := New(records, Options{CollectorComplete: true})
	if err != nil {
		t.Fatal(err)
	}
	value, err := session.Query("kernel-shapes")
	if err != nil {
		t.Fatal(err)
	}
	report := value.(KernelShapes)
	if len(report.Distributions) != 1 || report.Distributions[0].Count != 2 || report.Distributions[0].Sum != 30 || report.Distributions[0].Median != 20 {
		t.Fatal(report)
	}
	records.Activities[0].End = math.MaxInt64
	records.Activities[1].End = math.MaxInt64
	if _, err := New(records, Options{CollectorComplete: true}); err == nil {
		t.Fatal("overflow accepted")
	}
}
func ExampleNew() {
	session, _ := New(timeline.Records{}, Options{})
	fmt.Println(session.Summary().Version)
	// Output: 1
}
func ExampleSession_Summary() {
	session, _ := New(timeline.Records{}, Options{})
	fmt.Println(session.Summary().Gates["dependency"])
	// Output: partial
}
func ExampleOptions() {
	options := Options{Domain: "runtime"}
	fmt.Println(options.Domain)
	// Output: runtime
}
func ExampleQuery() {
	query := Catalog()[0]
	fmt.Println(query.Name, query.Version)
	// Output: inventory 1
}
func ExampleSource() {
	source := Source{Journal: "process.journal", Kind: "activity", Index: 0, Sequence: 4}
	fmt.Println(source.Kind, source.Sequence)
	// Output: activity 4
}
func ExampleEndpoint() {
	endpoint := Endpoint{Time: 10, Metadata: marker.Metadata{Phase: "decode", Event: "delivered"}}
	fmt.Println(endpoint.Metadata.Event, endpoint.Time)
	// Output: delivered 10
}
func ExampleDeviceWindows() {
	window := DeviceWindows{Journal: "p", Device: 1, UnionNS: 10}
	fmt.Println(window.Journal, window.Device, window.UnionNS)
	// Output: p 1 10
}
func ExampleLaunch() {
	launch := Launch{Symbol: "gemv", Duration: 10}
	fmt.Println(launch.Symbol, launch.Duration)
	// Output: gemv 10
}
func ExampleSummary() {
	report := Summary{Version: 1}
	fmt.Println(report.Version)
	// Output: 1
}
func ExampleKernelShapes() {
	report := KernelShapes{Qualification: "unqualified"}
	fmt.Println(report.Qualification)
	// Output: unqualified
}
func ExampleLaunchDistribution() {
	report := LaunchDistribution{Count: 2, Sum: 30}
	fmt.Println(report.Count, report.Sum)
	// Output: 2 30
}

func ExampleSQL() {
	sql, err := SQL("inventory")
	fmt.Println(err == nil, strings.Contains(sql, "cudaprof_inventory_v1"))
	// Output: true true
}

func ExampleSQLDomain() {
	sql, err := SQLDomain("correlations", "runtime")
	fmt.Println(err == nil, strings.Contains(sql, "domain='runtime'"))
	// Output: true true
}
