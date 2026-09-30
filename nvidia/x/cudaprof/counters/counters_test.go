package counters

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func counterReceipt() Receipt {
	value := 10.0
	return Receipt{Identity: shape.Identity{Phase: "decode", Operation: "gemv", DType: "float32", Grid: [3]int{1, 1, 1}, Block: [3]int{32, 1, 1}}, CaptureSHA256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", Journal: "j", RawSymbol: "kernel", Device: "uuid", Chip: "chip", DriverVersion: "driver", ToolVersion: "tool", Replay: "user; state restored", ReplayComplete: true, Overhead: "unmeasured", Properties: cuptirange.MetricProperties{Name: "metric", Description: "native definition", HardwareUnit: "unit", Dimension: "byte", CUPTIVersion: 130200}, Value: &value}
}

func TestCompare(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*Receipt)
		want bool
	}{
		{"compatible", func(r *Receipt) { value := 20.0; r.Value = &value }, true},
		{"units", func(r *Receipt) { r.Properties.Dimension = "cycle" }, false},
		{"description", func(r *Receipt) { r.Properties.Description = "other definition" }, false},
		{"version", func(r *Receipt) { r.Properties.CUPTIVersion++ }, false},
		{"device", func(r *Receipt) { r.Device = "other" }, false},
		{"tool", func(r *Receipt) { r.ToolVersion = "other" }, false},
		{"replay", func(r *Receipt) { r.ReplayComplete = false }, false},
		{"error", func(r *Receipt) { r.Failure = "permission denied" }, false},
		{"missing", func(r *Receipt) { r.Value = nil }, false},
		{"invalid scope", func(r *Receipt) { r.Properties.Scope = cupti.MetricCollectionScopeInvalid }, false},
		{"invalid type", func(r *Receipt) { r.Properties.Type = cupti.MetricTypeCount }, false},
		{"negative layer", func(r *Receipt) { v := -1; r.Identity.Layer = &v }, false},
		{"negative cache", func(r *Receipt) { r.Identity.CacheLength = -1 }, false},
		{"invalid hash", func(r *Receipt) { r.CaptureSHA256 = "bad" }, false},
		{"nonhex hash", func(r *Receipt) { r.CaptureSHA256 = "z123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" }, false},
		{"nan", func(r *Receipt) { v := math.NaN(); r.Value = &v }, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b, r := counterReceipt(), counterReceipt()
			tt.edit(&r)
			got := Compare(b, r)
			if got.Compatible != tt.want {
				t.Fatalf("got %+v", got)
			}
			if !tt.want && (got.Delta != nil || got.Reason == "") {
				t.Fatalf("unavailable comparison: %+v", got)
			}
		})
	}
}

func TestSelect(t *testing.T) {
	id := shape.Identity{Phase: "decode", Operation: "gemv", DType: "float32", Grid: [3]int{1, 1, 1}, Block: [3]int{32, 1, 1}}
	launch := shape.Launch{Journal: "j", RawSymbol: "kernel", Start: 1, Duration: 2, Identity: &id, APIStatus: "matched", MetadataStatus: "matched"}
	report := shape.Report{Version: 1, Pass: true, Groups: []shape.Group{{Identity: id, Base: []shape.Launch{launch}, Target: []shape.Launch{launch}}}}
	if _, err := Select(report, id); err != nil {
		t.Fatal(err)
	}
	report.Pass = false
	if _, err := Select(report, id); err == nil {
		t.Fatal("accepted failed gate")
	}
	report.Pass = true
	report.Groups = append(report.Groups, report.Groups[0])
	if _, err := Select(report, id); err == nil {
		t.Fatal("accepted ambiguous selection")
	}
}

func TestSelectLaunchIdentity(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*shape.Launch)
	}{
		{"missing journal", func(l *shape.Launch) { l.Journal = "" }},
		{"missing raw symbol", func(l *shape.Launch) { l.RawSymbol = "" }},
		{"missing duration", func(l *shape.Launch) { l.Duration = 0 }},
		{"negative duration", func(l *shape.Launch) { l.Duration = -1 }},
		{"missing timestamp", func(l *shape.Launch) { l.Start = 0 }},
		{"negative activity", func(l *shape.Launch) { l.Activity = -1 }},
		{"overflow duration", func(l *shape.Launch) { l.Start = math.MaxInt64; l.Duration = 2 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			id := counterReceipt().Identity
			l := shape.Launch{Journal: "j", RawSymbol: "kernel", Start: 1, Duration: 2, Identity: &id, APIStatus: "matched", MetadataStatus: "matched"}
			target := l
			tt.edit(&l)
			report := shape.Report{Version: 1, Pass: true, Groups: []shape.Group{{Identity: id, Base: []shape.Launch{l}, Target: []shape.Launch{target}}}}
			if _, err := Select(report, id); err == nil {
				t.Fatal("accepted incomplete launch identity")
			}
		})
	}
}
