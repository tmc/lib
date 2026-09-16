package cudacensus

import (
	"strings"
	"testing"
)

func TestActiveIntervalNs(t *testing.T) {
	tests := []struct {
		name      string
		intervals [][2]int64
		want      int64
	}{
		{
			name:      "empty",
			intervals: nil,
			want:      0,
		},
		{
			name:      "single",
			intervals: [][2]int64{{100, 200}},
			want:      100,
		},
		{
			name: "disjoint",
			intervals: [][2]int64{
				{100, 200},
				{300, 450},
			},
			want: 250,
		},
		{
			name: "overlapping",
			intervals: [][2]int64{
				{100, 200},
				{150, 250},
			},
			want: 150,
		},
		{
			name: "fully contained",
			intervals: [][2]int64{
				{100, 300},
				{120, 200},
				{150, 250},
			},
			want: 200,
		},
		{
			name: "adjacent contiguous",
			intervals: [][2]int64{
				{100, 200},
				{200, 300},
			},
			want: 200,
		},
		{
			name: "unsorted multi-stream overlap",
			intervals: [][2]int64{
				{500, 600},
				{100, 200},
				{150, 300},
				{250, 350},
			},
			want: 350, // [100, 350] = 250, plus [500, 600] = 100 -> 350
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ActiveIntervalNs(tt.intervals)
			if got != tt.want {
				t.Errorf("ActiveIntervalNs = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestActiveNsVsTotalNs(t *testing.T) {
	launches := []Launch{
		{Name: "k1", StartNs: 100, EndNs: 200},
		{Name: "k2", StartNs: 150, EndNs: 250},
	}
	var total int64
	for _, l := range launches {
		total += l.DurationNs()
	}
	if total != 200 {
		t.Fatalf("total = %d, want 200", total)
	}

	active := ActiveNs(launches)
	if active != 150 {
		t.Fatalf("active = %d, want 150", active)
	}
}

func TestAnalyzeCapture(t *testing.T) {
	c, err := ReadDir(fixture)
	if err != nil {
		t.Fatal(err)
	}
	tok, ok := c.Span("token")
	if !ok {
		t.Fatal("no token span")
	}

	p := Phase{
		Name:    tok.Name,
		StartNs: tok.StartNs,
		EndNs:   tok.EndNs,
	}

	rules := []FamilyRule{
		{
			Name: "norm",
			Match: func(sym string) bool {
				return strings.Contains(sym, "rms") || strings.Contains(sym, "norm")
			},
		},
		{
			Name: "gemm",
			Match: func(sym string) bool {
				return strings.Contains(sym, "gemm") || strings.Contains(sym, "matmul")
			},
		},
	}

	rep := c.Analyze([]Phase{p}, nil, rules...)
	if !rep.Complete {
		t.Errorf("expected complete report, got missing: %v", rep.Missing)
	}
	if len(rep.Phases) != 1 {
		t.Fatalf("phases count = %d, want 1", len(rep.Phases))
	}

	pr := rep.Phases[0]
	if pr.Launches != 132 {
		t.Errorf("pr.Launches = %d, want 132", pr.Launches)
	}
	if pr.TotalDeviceNs <= 0 {
		t.Errorf("pr.TotalDeviceNs = %d, want > 0", pr.TotalDeviceNs)
	}
	if pr.ActiveDeviceNs <= 0 || pr.ActiveDeviceNs > pr.TotalDeviceNs {
		t.Errorf("pr.ActiveDeviceNs = %d, invalid relative to TotalDeviceNs %d", pr.ActiveDeviceNs, pr.TotalDeviceNs)
	}
	if pr.CoverageCount+pr.UnknownCount != pr.Launches {
		t.Errorf("coverage (%d) + unknown (%d) != total (%d)", pr.CoverageCount, pr.UnknownCount, pr.Launches)
	}
	if pr.UnknownCount == 0 {
		t.Errorf("expected some unknown launches in fixture without complete rules")
	}
	if rep.UnknownLaunch != pr.UnknownCount {
		t.Errorf("rep.UnknownLaunch = %d, want %d", rep.UnknownLaunch, pr.UnknownCount)
	}
}

func TestAnalyzeInvalidPhase(t *testing.T) {
	c := new(Capture)
	badPhase := Phase{
		Name:    "bad",
		StartNs: 200,
		EndNs:   100,
	}
	rep := c.Analyze([]Phase{badPhase}, nil)
	if rep.Complete {
		t.Errorf("expected Complete=false for non-positive duration phase")
	}
	if len(rep.Missing) == 0 {
		t.Errorf("expected Missing description for invalid phase")
	}
}
