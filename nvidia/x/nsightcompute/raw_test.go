package nsightcompute

import (
	"os"
	"strings"
	"testing"
)

func TestReadRawFixture(t *testing.T) {
	f, err := os.Open("testdata/h200-raw.csv")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tab, err := ReadRaw(f, MaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	if len(tab.Preamble) != 3 || !strings.HasPrefix(tab.Preamble[0], "==PROF==") {
		t.Errorf("preamble = %q, want three ==PROF== lines", tab.Preamble)
	}
	if len(tab.Launches) != 9 {
		t.Fatalf("got %d launches, want 9", len(tab.Launches))
	}
	tests := []struct {
		launch int
		metric string
		want   float64
		unit   string
	}{
		{0, "gpu__time_duration.sum", 4512, "ns"},
		{6, "dram__bytes_read.sum", 8617984, "byte"},
		{6, "sm__ctas_launched.sum", 132, "block"},
		{8, "sm__warps_active.avg.pct_of_peak_sustained_active", 12.09, "%"},
		{0, "dram__bytes_write.sum", 0, "byte"},
	}
	for _, tt := range tests {
		got, unit, err := tab.Value(&tab.Launches[tt.launch], tt.metric)
		if err != nil || got != tt.want || unit != tt.unit {
			t.Errorf("Value(%d, %s) = %v %q, %v; want %v %q", tt.launch, tt.metric, got, unit, err, tt.want, tt.unit)
		}
	}
	l := tab.Launches[0]
	if l.Kernel != "_qk_rms_norm_fwd_kernel" || l.Block != "(128, 1, 1)" || l.Grid != "(72, 1, 1)" {
		t.Errorf("launch 0 = %q %q %q", l.Kernel, l.Block, l.Grid)
	}
	if _, _, err := tab.Value(&l, "device__attribute_display_name"); err == nil {
		t.Error("Value(display name) succeeded on a string metric")
	}
	if _, _, err := tab.Value(&l, "no__such_metric"); err == nil {
		t.Error("Value(missing metric) succeeded")
	}
}

func TestReadRaw(t *testing.T) {
	const head = `"ID","Kernel Name","Block Size","m.sum","n.pct"` + "\n" + `"","","","ns","%"` + "\n"
	tests := []struct {
		name    string
		in      string
		wantErr string
		check   func(*RawTable) string
	}{
		{name: "ok", in: head + `"0","k","(1, 1, 1)","1,024","50.5"` + "\n",
			check: func(t *RawTable) string {
				v, u, err := t.Value(&t.Launches[0], "m.sum")
				if err != nil || v != 1024 || u != "ns" {
					return "m.sum not 1024 ns"
				}
				return ""
			}},
		{name: "crlf preamble", in: "==PROF== x\r\n" + head + `"0","k","(1, 1, 1)","1","2"` + "\n",
			check: func(t *RawTable) string {
				if len(t.Preamble) != 1 || t.Preamble[0] != "==PROF== x" {
					return "bad preamble"
				}
				return ""
			}},
		{name: "n/a is retained but not a value", in: head + `"0","k","(1, 1, 1)","n/a","2"` + "\n",
			check: func(t *RawTable) string {
				if _, _, err := t.Value(&t.Launches[0], "m.sum"); err == nil || t.Launches[0].Raw[0] != "n/a" {
					return "n/a read as a number"
				}
				return ""
			}},
		{name: "bad grouping", in: head + `"0","k","(1, 1, 1)","1,02","2"` + "\n",
			check: func(t *RawTable) string {
				if _, _, err := t.Value(&t.Launches[0], "m.sum"); err == nil {
					return "1,02 read as a number"
				}
				return ""
			}},
		{name: "permission error", in: "==PROF== Connected\n==ERROR== ERR_NVGPUCTRPERM - The user does not have permission\n" + head,
			wantErr: "ERR_NVGPUCTRPERM"},
		{name: "error after header", in: head + `"0","k","(1, 1, 1)","1","2"` + "\n==ERROR== failed to profile\n",
			wantErr: "failed to profile"},
		{name: "no header", in: "==PROF== Disconnected\n", wantErr: "no header"},
		{name: "no units", in: `"ID","Kernel Name"` + "\n", wantErr: "no units row"},
		{name: "duplicate column", in: `"ID","Kernel Name","m","m"` + "\n" + `"","","",""` + "\n", wantErr: "duplicate raw column"},
		{name: "duplicate launch", in: head + `"0","k","b","1","2"` + "\n" + `"0","k","b","1","2"` + "\n", wantErr: "duplicate launch ID"},
		{name: "short row", in: head + `"0","k","b","1"` + "\n", wantErr: "wrong number of fields"},
		{name: "unit on launch column", in: `"ID","Kernel Name"` + "\n" + `"x",""` + "\n", wantErr: "launch column"},
		{name: "empty kernel", in: head + `"0","","b","1","2"` + "\n", wantErr: "empty kernel name"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tab, err := ReadRaw(strings.NewReader(tt.in), MaxBytes)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("ReadRaw() error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if msg := tt.check(tab); msg != "" {
				t.Error(msg)
			}
		})
	}
	if _, err := ReadRaw(strings.NewReader(head), 10); err == nil {
		t.Error("ReadRaw over budget succeeded")
	}
}
