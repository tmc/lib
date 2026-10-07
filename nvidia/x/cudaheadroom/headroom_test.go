package cudaheadroom

import (
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudashared"
	"testing"
)

func TestMeasuredBudget(t *testing.T) {
	for _, tt := range []struct {
		name                        string
		user, static, dynamic, want int64
		bad                         bool
	}{
		{"user excludes reservation", 101376, 0, 101376, 101376, false},
		{"static counted once", 101376, 1024, 101376, 100352, false},
		{"lower function cap", 101376, 1024, 48000, 48000, false},
		{"zero dynamic", 101376, 0, 0, 0, false},
		{"negative", 101376, -1, 0, 0, true},
		{"unavailable", 0, 0, 0, 0, true},
		{"static over user", 1, 2, 0, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b, err := budget(tt.user, tt.static, tt.dynamic)
			if (err != nil) != tt.bad {
				t.Fatalf("budget error %v", err)
			}
			if err == nil && b.Ceiling != tt.want {
				t.Fatalf("ceiling %d want %d", b.Ceiling, tt.want)
			}
		})
	}
}
func testPlan(t *testing.T, static, dynamic int64) *cudashared.Plan {
	t.Helper()
	p, err := cudashared.NewPlan(cudashared.Limits{StaticMax: 101376, DynamicMax: 101376, TotalMax: 101376, Source: "explicit host fixture"})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range []cudashared.Request{{Name: "static", Size: static, Alignment: 1}, {Name: "dynamic", Size: dynamic, Alignment: 1, Dynamic: true}} {
		if r.Size > 0 {
			if _, err := p.Add(r); err != nil {
				t.Fatal(err)
			}
		}
	}
	return p
}
func TestPlanCounting(t *testing.T) {
	b, _ := budget(101376, 1024, 48000)
	for _, tt := range []struct {
		name                   string
		static, dynamic, other int64
		bad                    bool
	}{
		{"boundary", 512, 48000, 512, false}, {"over function", 512, 48001, 512, true},
		{"wrong static", 512, 48000, 0, true}, {"negative other", 512, 1, -1, true},
		{"compiled includes plan", 1024, 48000, 0, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := checkPlan(b, testPlan(t, tt.static, tt.dynamic), tt.other, true)
			if (err != nil) != tt.bad {
				t.Fatalf("check: %v", err)
			}
		})
	}
	if err := checkPlan(b, nil, 0, true); err == nil {
		t.Fatal("nil plan accepted")
	}
}
func TestNilAndZeroOwners(t *testing.T) {
	for _, k := range []*cuda.Kernel{nil, {}} {
		if _, err := Inspect(k); err == nil {
			t.Fatal("unowned kernel accepted")
		}
		if _, err := CheckPlan(k, testPlan(t, 0, 1), 0); err == nil {
			t.Fatal("unowned plan accepted")
		}
		if _, err := PreparePlan(k, testPlan(t, 0, 1), 0); err == nil {
			t.Fatal("unowned preparation accepted")
		}
	}
}
