package main

import (
	"bytes"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// kernelsCUBIN is testdata's real CUBIN, which defines the kernel compute.
func kernelsCUBIN(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "kernels.cubin.hex"))
	if err != nil {
		t.Fatal(err)
	}
	cubin, err := hex.DecodeString(strings.Join(strings.Fields(string(b)), ""))
	if err != nil {
		t.Fatal(err)
	}
	return cubin
}

func TestCubinFunctions(t *testing.T) {
	got := cubinFunctions(kernelsCUBIN(t))
	found := false
	for _, name := range got {
		found = found || name == "compute"
	}
	if !found {
		t.Errorf("cubinFunctions = %q, want compute among them", got)
	}
	for _, in := range [][]byte{nil, []byte("CUBIN_MOCK")} {
		if got := cubinFunctions(in); got != nil {
			t.Errorf("cubinFunctions(%q) = %q, want none", in, got)
		}
	}
}

// TestInferModules builds a capture with the shape a runtime-API program
// produces: several modules and kernel samples none of which name one.
func TestInferModules(t *testing.T) {
	cubin := kernelsCUBIN(t)
	for _, tt := range []struct {
		name    string
		modules []cudacapture.Module
		files   map[string][]byte
		label   map[string][]string
		want    moduleInference
		wantID  string
	}{
		{
			name:    "one module defines it",
			modules: []cudacapture.Module{{ID: "a", DriverID: 7, Code: cudacapture.CodeFiles{CUBIN: "a.cubin"}}, {ID: "b", DriverID: 8, Code: cudacapture.CodeFiles{CUBIN: "b.cubin"}}},
			files:   map[string][]byte{"a.cubin": []byte("CUBIN_MOCK"), "b.cubin": cubin},
			label:   map[string][]string{"subtype": {"kernel"}, "name": {"compute"}},
			want:    moduleInference{Inferred: 1},
			wantID:  "8",
		},
		{
			name:    "two modules define it",
			modules: []cudacapture.Module{{ID: "a", DriverID: 7, Code: cudacapture.CodeFiles{CUBIN: "a.cubin"}}, {ID: "b", DriverID: 8, Code: cudacapture.CodeFiles{CUBIN: "b.cubin"}}},
			files:   map[string][]byte{"a.cubin": cubin, "b.cubin": cubin},
			label:   map[string][]string{"subtype": {"kernel"}, "name": {"compute"}},
			want:    moduleInference{Ambiguous: 1},
		},
		{
			name:    "no module defines it",
			modules: []cudacapture.Module{{ID: "a", DriverID: 7, Code: cudacapture.CodeFiles{CUBIN: "a.cubin"}}, {ID: "b", DriverID: 8, Code: cudacapture.CodeFiles{CUBIN: "b.cubin"}}},
			files:   map[string][]byte{"a.cubin": cubin, "b.cubin": cubin},
			label:   map[string][]string{"subtype": {"kernel"}, "name": {"other"}},
			want:    moduleInference{Absent: 1},
		},
		{
			name:    "the defining module has no driver number",
			modules: []cudacapture.Module{{ID: "a", DriverID: 7, Code: cudacapture.CodeFiles{CUBIN: "a.cubin"}}, {ID: "b", Code: cudacapture.CodeFiles{CUBIN: "b.cubin"}}},
			files:   map[string][]byte{"a.cubin": []byte("CUBIN_MOCK"), "b.cubin": cubin},
			label:   map[string][]string{"subtype": {"kernel"}, "name": {"compute"}},
			want:    moduleInference{Absent: 1},
		},
		{
			name:    "the producer's label stands",
			modules: []cudacapture.Module{{ID: "a", DriverID: 7, Code: cudacapture.CodeFiles{CUBIN: "a.cubin"}}, {ID: "b", DriverID: 8, Code: cudacapture.CodeFiles{CUBIN: "b.cubin"}}},
			files:   map[string][]byte{"a.cubin": []byte("CUBIN_MOCK"), "b.cubin": cubin},
			label:   map[string][]string{"subtype": {"kernel"}, "name": {"compute"}, "module_id": {"7"}},
			wantID:  "7",
		},
		{
			name:    "a copy belongs to no module",
			modules: []cudacapture.Module{{ID: "a", DriverID: 7, Code: cudacapture.CodeFiles{CUBIN: "a.cubin"}}, {ID: "b", DriverID: 8, Code: cudacapture.CodeFiles{CUBIN: "b.cubin"}}},
			files:   map[string][]byte{"a.cubin": []byte("CUBIN_MOCK"), "b.cubin": cubin},
			label:   map[string][]string{"subtype": {"memcpy"}, "name": {"compute"}},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := &profile.Sample{Value: []int64{1}, Label: tt.label}
			cap := &cudacapture.Capture{
				Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}}, Sample: []*profile.Sample{s}},
				Files:   tt.files,
				Modules: tt.modules,
			}
			if got := inferModules(cap); got != tt.want {
				t.Errorf("inferModules = %+v, want %+v", got, tt.want)
			}
			var id string
			if v := s.Label["module_id"]; len(v) == 1 {
				id = v[0]
			}
			if id != tt.wantID {
				t.Errorf("module_id = %q, want %q", id, tt.wantID)
			}
			inferred := len(s.Label["module_inferred"]) == 1
			if inferred != (tt.want.Inferred == 1) {
				t.Errorf("module_inferred present = %v, want %v", inferred, tt.want.Inferred == 1)
			}
		})
	}
}

// TestPrintCaptureLinesInfersModules is the end-to-end form: an unlabelled
// kernel sample lands in the table of the one module that defines it.
func TestPrintCaptureLinesInfersModules(t *testing.T) {
	cap := fusionFixture(t, 0)
	cap.Files["other.cubin"] = []byte("CUBIN_MOCK")
	cap.Modules = []cudacapture.Module{
		{ID: "m1", DriverID: 1, Name: "first", Code: cudacapture.CodeFiles{CUBIN: "other.cubin", SASS: "kernels.sass"}},
		{ID: "m2", DriverID: 2, Name: "second", Code: cap.Code},
	}
	cap.Profile = &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
		Sample: []*profile.Sample{
			{Value: []int64{5}, Label: map[string][]string{"subtype": {"kernel"}, "name": {"compute"}}},
			{Value: []int64{3}, Label: map[string][]string{"subtype": {"kernel"}, "name": {"elsewhere"}}},
		},
	}
	if got := inferModules(cap); got != (moduleInference{Inferred: 1, Absent: 1}) {
		t.Fatalf("inferModules = %+v", got)
	}
	if local := moduleView(cap, cap.Modules[1]); len(local.Profile.Sample) != 1 {
		t.Errorf("module second holds %d samples, want the inferred one", len(local.Profile.Sample))
	}
	if local := moduleView(cap, cap.Modules[0]); len(local.Profile.Sample) != 0 {
		t.Errorf("module first holds %d samples, want none", len(local.Profile.Sample))
	}
	var buf bytes.Buffer
	cap.Profile.Sample[0].Label = map[string][]string{"subtype": {"kernel"}, "name": {"compute"}}
	if err := printCaptureLines(&buf, cap, "", "kernel_time", "digraph"); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"attributed by symbol: 1 records", "naming no module: 1 records", "1 with a symbol in no module, 0 in several"} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, buf.String())
		}
	}
}
