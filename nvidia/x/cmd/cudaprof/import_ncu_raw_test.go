package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nsightcompute"
)

func TestImportNCURaw(t *testing.T) {
	data, err := os.ReadFile("../../nsightcompute/testdata/h200-raw.csv")
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := runImportNCURaw(data, len(data), 2740, "", false, &out); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 10 {
		t.Fatalf("got %d lines, want header and 9 launches:\n%s", len(lines), out.String())
	}
	// Launch 6 read 8,617,984 bytes in 16,608 ns: 518.9 GB/s, 0.189 of 2740.
	want := strings.Fields("6 16608 8617984 518.9 0.189 12.30 132")
	if got := strings.Fields(lines[7])[:7]; strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("launch 6 = %q, want %q", got, want)
	}
}

func TestNCURawUnits(t *testing.T) {
	const csv = `"ID","Kernel Name","gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum","sm__warps_active.avg.pct_of_peak_sustained_active","sm__ctas_launched.sum"
"","","usecond","byte","Kbyte","ratio","warp"
"0","k","2","1,000","10","0.5","4"
`
	var out bytes.Buffer
	if err := runImportNCURaw([]byte(csv), 1<<20, 0, "", true, &out); err != nil {
		t.Fatal(err)
	}
	s := out.String()
	if !strings.Contains(s, `"duration_ns": 2000`) {
		t.Errorf("usecond duration not converted:\n%s", s)
	}
	for _, absent := range []string{"dram_bytes", "warps_active_pct", "ctas_launched", "ceiling_fraction"} {
		if strings.Contains(s, absent) {
			t.Errorf("%s derived from an unconverted unit:\n%s", absent, s)
		}
	}
}

func TestImportNCURawFlags(t *testing.T) {
	for _, args := range [][]string{
		{"-raw", "-capture", "c.pprof", "x.csv"},
		{"-ceiling", "100", "x.json"},
		{"-raw", "-ceiling", "-1", "x.csv"},
		{"-raw", "-ceiling", "+Inf", "../../nsightcompute/testdata/h200-raw.csv"},
		{"-raw", "-ceiling", "NaN", "../../nsightcompute/testdata/h200-raw.csv"},
		{"-raw", "-ceiling", "100", "-device", "auto", "x.csv"},
		{"-device", "auto", "x.json"},
	} {
		if err := runImportNCUCommand(args, &bytes.Buffer{}); err == nil {
			t.Errorf("import-ncu %q succeeded", args)
		}
	}
}

func TestImportNCURawDevice(t *testing.T) {
	data, err := os.ReadFile("../../nsightcompute/testdata/h200-raw.csv")
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		device  string
		json    bool
		want    []string
		wantErr string
	}{
		// 518.9 GB/s against the measured H200 copy ceiling of 2740.48.
		{device: "auto", want: []string{"ceiling NVIDIA H200: 2740.48 GB/s (measured)", " 518.9 ", " 0.189 "}},
		{device: "nvidia h200", json: true, want: []string{`"ceiling_device": "NVIDIA H200"`, `"ceiling_provenance": "measured"`, `"ceiling_gbps": 2740.48`}},
		{device: "NVIDIA GB10", wantErr: `-device "NVIDIA GB10": export was recorded on "NVIDIA H200"`},
	}
	for _, tt := range tests {
		var out bytes.Buffer
		err := runImportNCURaw(data, len(data), 0, tt.device, tt.json, &out)
		if tt.wantErr != "" {
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("-device %q: error %v, want %q", tt.device, err, tt.wantErr)
			}
			continue
		}
		if err != nil {
			t.Errorf("-device %q: %v", tt.device, err)
			continue
		}
		for _, w := range tt.want {
			if !strings.Contains(out.String(), w) {
				t.Errorf("-device %q: output lacks %q:\n%s", tt.device, w, out.String())
			}
		}
	}

	// An export without the display-name metric cannot say which part it
	// ran on, so auto has nothing to look up.
	csv := "\"ID\",\"Kernel Name\",\"gpu__time_duration.sum\"\n\"\",\"\",\"nsecond\"\n\"0\",\"k\",\"5\"\n"
	err = runImportNCURaw([]byte(csv), 1<<20, 0, "auto", false, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "names 0 devices") {
		t.Errorf("auto without display name: %v", err)
	}
}

func TestNCURawDerivedValues(t *testing.T) {
	tests := []struct {
		name                                             string
		duration, durationUnit, read, write, warps, ctas string
		ceiling                                          float64
		want                                             map[string]float64
	}{
		{"valid", "2", "us", "1000", "1000", "50", "4", 2, map[string]float64{"duration_ns": 2000, "dram_bytes": 2000, "dram_gbps": 1, "ceiling_fraction": 0.5, "warps_active_pct": 50, "ctas_launched": 4}},
		{"zero counts", "1", "ns", "0", "0", "0", "0", 2, map[string]float64{"duration_ns": 1, "dram_bytes": 0, "dram_gbps": 0, "ceiling_fraction": 0, "warps_active_pct": 0, "ctas_launched": 0}},
		{"negative read", "1", "ns", "-1", "2", "n/a", "n/a", 0, map[string]float64{"duration_ns": 1}},
		{"negative write", "1", "ns", "2", "-1", "n/a", "n/a", 0, map[string]float64{"duration_ns": 1}},
		{"negative duration", "-1", "ns", "1", "1", "n/a", "n/a", 0, map[string]float64{"dram_bytes": 2}},
		{"zero duration", "0", "ns", "1", "1", "n/a", "n/a", 0, map[string]float64{"dram_bytes": 2}},
		{"missing duration", "n/a", "ns", "1", "1", "n/a", "n/a", 0, map[string]float64{"dram_bytes": 2}},
		{"missing duration unit", "1", "", "1", "1", "n/a", "n/a", 0, map[string]float64{"dram_bytes": 2}},
		{"conversion overflow", "1e308", "s", "1", "1", "n/a", "n/a", 0, map[string]float64{"dram_bytes": 2}},
		{"sum overflow", "1", "ns", "1e308", "1e308", "n/a", "n/a", 0, map[string]float64{"duration_ns": 1}},
		{"rate overflow", "1e-308", "ns", "1e308", "0", "n/a", "n/a", 0, map[string]float64{"duration_ns": 1e-308, "dram_bytes": 1e308}},
		{"fraction overflow", "1", "ns", "1e308", "0", "n/a", "n/a", 1e-308, map[string]float64{"duration_ns": 1, "dram_bytes": 1e308, "dram_gbps": 1e308}},
		{"negative occupancy and CTAs", "1", "ns", "n/a", "n/a", "-1", "-1", 0, map[string]float64{"duration_ns": 1}},
		{"occupancy above 100", "1", "ns", "n/a", "n/a", "101", "2", 0, map[string]float64{"duration_ns": 1, "ctas_launched": 2}},
		{"occupancy upper bound", "1", "ns", "n/a", "n/a", "100", "2", 0, map[string]float64{"duration_ns": 1, "warps_active_pct": 100, "ctas_launched": 2}},
		{"infinite input", "1e309", "ns", "1e309", "1", "1e309", "1e309", 0, map[string]float64{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			table := &nsightcompute.RawTable{
				Metrics:  []string{"gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum", "sm__warps_active.avg.pct_of_peak_sustained_active", "sm__ctas_launched.sum"},
				Units:    []string{tt.durationUnit, "byte", "byte", "%", "cta"},
				Launches: []nsightcompute.RawLaunch{{ID: "0", Kernel: "k", Raw: []string{tt.duration, tt.read, tt.write, tt.warps, tt.ctas}}},
			}
			got := ncuRawLaunches(table, tt.ceiling)[0]
			fields := map[string]*float64{"duration_ns": got.DurationNS, "dram_bytes": got.DRAMBytes, "dram_gbps": got.DRAMGBps, "ceiling_fraction": got.CeilingFrac, "warps_active_pct": got.WarpsActive, "ctas_launched": got.CTAsLaunched}
			for name, p := range fields {
				want, ok := tt.want[name]
				if !ok {
					if p != nil {
						t.Errorf("%s = %g, want absent", name, *p)
					}
					continue
				}
				if p == nil || *p != want {
					t.Errorf("%s = %v, want %g", name, p, want)
				}
			}
			if _, err := json.Marshal(got); err != nil {
				t.Errorf("derived launch cannot be encoded: %v", err)
			}
			csv := fmt.Sprintf("\"ID\",\"Kernel Name\",\"gpu__time_duration.sum\",\"dram__bytes_read.sum\",\"dram__bytes_write.sum\",\"sm__warps_active.avg.pct_of_peak_sustained_active\",\"sm__ctas_launched.sum\"\n\"\",\"\",%q,\"byte\",\"byte\",\"%%\",\"cta\"\n\"0\",\"k\",%q,%q,%q,%q,%q\n", tt.durationUnit, tt.duration, tt.read, tt.write, tt.warps, tt.ctas)
			for _, isJSON := range []bool{false, true} {
				var out bytes.Buffer
				if err := runImportNCURaw([]byte(csv), 1<<20, tt.ceiling, "", isJSON, &out); err != nil {
					t.Fatal(err)
				}
				if strings.Contains(out.String(), "Inf") || strings.Contains(out.String(), "NaN") {
					t.Errorf("nonfinite output: %s", out.String())
				}
			}
		})
	}
}

func TestNCURawInvalidCeiling(t *testing.T) {
	for _, v := range []float64{-1, math.Inf(1), math.Inf(-1), math.NaN()} {
		if _, err := resolveNCUCeiling(&nsightcompute.RawTable{}, v, ""); err == nil {
			t.Errorf("ceiling %g accepted", v)
		}
	}
}
