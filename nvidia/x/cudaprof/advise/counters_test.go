package advise

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func counterFixture(t *testing.T) []CounterAction {
	t.Helper()
	raw, err := os.ReadFile("testdata/counters-d5.json")
	if err != nil {
		t.Fatal(err)
	}
	var actions []CounterAction
	if err := json.Unmarshal(raw, &actions); err != nil {
		t.Fatal(err)
	}
	return actions
}
func counterMetric(t *testing.T, a *CounterAction, name string) (*cudacapture.MetricDescriptor, *cudacapture.Observation) {
	t.Helper()
	for i := range a.Descriptors {
		d := &a.Descriptors[i]
		if d.Name == name {
			for j := range a.Observations {
				if a.Observations[j].DescriptorID == d.ID {
					return d, &a.Observations[j]
				}
			}
		}
	}
	t.Fatalf("missing %s", name)
	return nil, nil
}
func counterHas(r *CounterReport, text string) bool {
	for _, f := range r.Findings {
		if strings.Contains(f.Item.Finding, text) {
			return true
		}
	}
	return false
}
func TestAnalyzeCountersD5(t *testing.T) {
	actions := counterFixture(t)
	r, err := AnalyzeCounters(actions)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := AnalyzeCounters(actions)
	if err != nil || !reflect.DeepEqual(r, r2) {
		t.Fatal("nondeterministic counter report")
	}
	if r.Mode != "COUNTER_ONLY" || !reflect.DeepEqual(r.Actions, actions) {
		t.Fatal("lost exact counter inputs")
	}
	for _, text := range []string{"jointly limit reported capacity to 2", "408179456 bytes over 143616 ns", "2842.158645276292", "823866916.0 inst; domain THREAD", "4849664.0 inst; domain WARP", "50.80827429574509 %; domain ACTION; denominator active", "22.86404964800125 %; domain ACTION; denominator elapsed"} {
		if !counterHas(r, text) {
			t.Errorf("missing fact %q", text)
		}
	}
	if counterHas(r, "nominal capacity slots") {
		t.Error("invented missing SM count")
	}
	for _, f := range r.Findings {
		i := f.Item
		if !i.NoRuntimeAttribution || i.Severity != SeverityContext || i.DeviceTimeNanos != 0 || i.HostTimeNanos != 0 || i.ShareOfProfile != 0 || i.ClassifiedShare != nil || i.Gap != nil || i.Location.PC != 0 || i.Location.Module != "" {
			t.Fatalf("runtime attribution leaked: %+v", f)
		}
		if len(f.MetricIDs) == 0 || f.ActionID == "" {
			t.Fatal("finding without exact input refs")
		}
	}
	// Report owns both structured input and raw numeric bytes.
	actions[0].Observations[0].Value[0] = '9'
	if reflect.DeepEqual(r.Actions, actions) {
		t.Fatal("report aliases mutable input")
	}
}
func TestAnalyzeCountersStructuralFaults(t *testing.T) {
	tests := []struct {
		name   string
		change func([]CounterAction)
	}{
		{"duplicate action", func(a []CounterAction) { a[1].ID = a[0].ID }},
		{"source digest", func(a []CounterAction) { a[0].Identity.CandidateSourceSHA256 = "invalid" }},
		{"case digest", func(a []CounterAction) { a[0].Identity.CaseConfigSHA256 = "" }},
		{"missing device", func(a []CounterAction) { a[0].Identity.DeviceUUID = "" }},
		{"missing census", func(a []CounterAction) { a[0].Identity.TypedCensus.SHA256 = "" }},
		{"cross acquisition", func(a []CounterAction) { a[0].Observations[0].RunID = a[1].AcquisitionID }},
		{"cross action", func(a []CounterAction) { a[0].Observations[0].EntityID = a[1].ID }},
		{"duration cross acquisition", func(a []CounterAction) {
			_, o := counterMetric(t, &a[1], "gpu__time_duration.sum")
			o.RunID = a[2].AcquisitionID
		}},
		{"pass count as ID", func(a []CounterAction) { a[0].Observations[0].PassID = "1" }},
		{"duplicate descriptor", func(a []CounterAction) { a[0].Descriptors = append(a[0].Descriptors, a[0].Descriptors[0]) }},
		{"duplicate name", func(a []CounterAction) {
			d := a[0].Descriptors[0]
			d.ID = "duplicate"
			a[0].Descriptors = append(a[0].Descriptors, d)
		}},
		{"duplicate observation", func(a []CounterAction) { a[0].Observations = append(a[0].Observations, a[0].Observations[0]) }},
		{"unknown descriptor", func(a []CounterAction) { a[0].Observations[0].DescriptorID = "missing" }},
		{"wrong unit", func(a []CounterAction) { d, _ := counterMetric(t, &a[1], "gpu__time_duration.sum"); d.Unit = "us" }},
		{"thread warp mismatch", func(a []CounterAction) {
			d, _ := counterMetric(t, &a[2], "smsp__sass_thread_inst_executed_op_integer_pred_on.sum")
			d.Scope = "WARP"
		}},
		{"active elapsed mismatch", func(a []CounterAction) {
			d, _ := counterMetric(t, &a[2], "smsp__issue_active.avg.pct_of_peak_sustained_active")
			d.Denominator = "elapsed"
		}},
		{"boolean", func(a []CounterAction) { a[0].Observations[0].Value = json.RawMessage(`true`) }},
		{"numeric string", func(a []CounterAction) { a[0].Observations[0].Value = json.RawMessage(`"1"`) }},
		{"null", func(a []CounterAction) { a[0].Observations[0].Value = json.RawMessage(`null`) }},
		{"double underflow", func(a []CounterAction) { a[0].Observations[0].Value = json.RawMessage(`1e-1000000000`) }},
		{"double overflow", func(a []CounterAction) { a[0].Observations[0].Value = json.RawMessage(`1e309`) }},
		{"uint64 overflow", func(a []CounterAction) {
			d, o := counterMetric(t, &a[0], "gpu__time_duration.sum")
			d.NumericType = "uint64"
			o.Value = json.RawMessage(`18446744073709551616`)
		}},
		{"negative byte", func(a []CounterAction) {
			_, o := counterMetric(t, &a[1], "dram__bytes_read.sum")
			o.Value = json.RawMessage(`-1`)
		}},
		{"fractional resource limit", func(a []CounterAction) {
			_, o := counterMetric(t, &a[0], "launch__occupancy_limit_registers")
			o.Value = json.RawMessage(`2.5`)
		}},
		{"unsourced SM count", func(a []CounterAction) { a[0].SMCount = &CounterSMCount{Count: 132} }},
		{"zero grid", func(a []CounterAction) { a[0].Launch.Grid[0] = 0 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := counterFixture(t)
			tt.change(a)
			if _, err := AnalyzeCounters(a); err == nil {
				t.Fatal("accepted invalid counter evidence")
			}
		})
	}
}
func TestAnalyzeCountersGeometry(t *testing.T) {
	a := counterFixture(t)[:1]
	a[0].SMCount = &CounterSMCount{Count: 132, Source: a[0].Identity.TypedCensus}
	r, err := AnalyzeCounters(a)
	if err != nil {
		t.Fatal(err)
	}
	if !counterHas(r, "Grid 256 CTAs is fewer than 264 nominal capacity slots") {
		t.Fatal("missing sourced geometry comparison")
	}
	a[0].Launch.Grid = [3]uint64{1024, 1, 1}
	r, err = AnalyzeCounters(a)
	if err != nil || !counterHas(r, "Grid 1024 CTAs is at least 264") || !counterHas(r, "jointly limit reported capacity to 2") {
		t.Fatal("geometry changed joint constraints")
	}
	a[0].Launch.Grid = [3]uint64{^uint64(0), ^uint64(0), ^uint64(0)}
	r, err = AnalyzeCounters(a)
	if err != nil || !counterHas(r, "at least 264") {
		t.Fatal("grid product overflowed")
	}
}
func TestAnalyzeCountersMissingAndUnknown(t *testing.T) {
	a := counterFixture(t)
	_, o := counterMetric(t, &a[1], "gpu__time_duration.sum")
	o.Value = json.RawMessage(`0`)
	r, err := AnalyzeCounters(a)
	if err != nil || counterHas(r, "decimal GB/s") {
		t.Fatal("zero duration became rate")
	}
	o.Availability = "NOT_COLLECTED"
	o.Value = nil
	_, o = counterMetric(t, &a[1], "dram__bytes_write.sum")
	o.Availability = "NOT_COLLECTED"
	o.Value = nil
	a[0].Descriptors = append(a[0].Descriptors, cudacapture.MetricDescriptor{ID: "unknown", Name: "future_quantity", Scope: "UNKNOWN", NumericType: "uint64"})
	a[0].Observations = append(a[0].Observations, cudacapture.Observation{DescriptorID: "unknown", EntityID: a[0].ID, RunID: a[0].AcquisitionID, Availability: "MEASURED", Value: json.RawMessage(`18446744073709551615`)})
	r, err = AnalyzeCounters(a)
	if err != nil || counterHas(r, "future_quantity") || counterHas(r, "decimal GB/s") {
		t.Fatal("unknown or unavailable quantity interpreted", err)
	}
	if got := string(r.Actions[0].Observations[len(a[0].Observations)-1].Value); got != "18446744073709551615" {
		t.Fatal("lost uint64 precision", got)
	}
	a[0].Descriptors[len(a[0].Descriptors)-1].NumericType = "unknown"
	a[0].Observations[len(a[0].Observations)-1].Value = json.RawMessage(` { "opaque_vendor_value" : true } `)
	opaque, err := AnalyzeCounters(a)
	if err != nil {
		t.Fatal("lost unsupported raw native value", err)
	}
	if got := string(opaque.Actions[0].Observations[len(a[0].Observations)-1].Value); got != ` { "opaque_vendor_value" : true } ` {
		t.Fatal("changed opaque raw bytes", got)
	}
}
func TestAnalyzeCountersSameKernelDifferentActions(t *testing.T) {
	a := counterFixture(t)
	a[1].Kernel = a[0].Kernel
	r, err := AnalyzeCounters(a)
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]bool{}
	for _, f := range r.Findings {
		found[f.ActionID] = true
	}
	if !found[a[0].ID] || !found[a[1].ID] {
		t.Fatal("merged identical display names")
	}
}
