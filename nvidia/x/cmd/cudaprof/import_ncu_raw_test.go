package main

import (
	"bytes"
	"os"
	"strings"
	"testing"
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
