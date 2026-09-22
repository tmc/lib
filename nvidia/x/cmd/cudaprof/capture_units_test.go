package main

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func unitJSON(name string) []byte {
	return []byte(`{"Name":"` + name + `","Source":"__global__ void ` + name + `() {}\n","LineMap":{}}`)
}

func TestAttachUnits(t *testing.T) {
	kernel := func(module uint32, name string) captureActivity {
		return captureActivity{Subtype: "kernel", ModuleID: module, Name: name}
	}
	for _, tt := range []struct {
		name        string
		modules     []cudacapture.Module
		activities  []captureActivity
		attachments map[string][]byte
		wantCapture string
		wantModules []string
	}{{
		name:        "one unit one module",
		modules:     []cudacapture.Module{{DriverID: 22}},
		attachments: map[string][]byte{"unit-fused_gate.json": unitJSON("fused_gate")},
		wantCapture: "unit-fused_gate.json",
		wantModules: []string{"unit-fused_gate.json"},
	}, {
		name:        "no units",
		modules:     []cudacapture.Module{{DriverID: 22}},
		attachments: map[string][]byte{"module-22.cubin": []byte("CUBIN")},
		wantModules: []string{""},
	}, {
		// Two builds of one unit are two modules naming one unit.
		name:        "one unit two modules",
		modules:     []cudacapture.Module{{DriverID: 22}, {DriverID: 23}},
		activities:  []captureActivity{kernel(22, "fused_gate"), kernel(23, "fused_gate")},
		attachments: map[string][]byte{"unit-fused_gate.json": unitJSON("fused_gate")},
		wantModules: []string{"unit-fused_gate.json", "unit-fused_gate.json"},
	}, {
		// The longer name wins, so the module running the residual kernel
		// is not matched to the unit its name has as a prefix.
		name:       "prefix names",
		modules:    []cudacapture.Module{{DriverID: 22}, {DriverID: 23}},
		activities: []captureActivity{kernel(22, "fused"), kernel(23, "fused_residual")},
		attachments: map[string][]byte{
			"unit-fused.json":          unitJSON("fused"),
			"unit-fused_residual.json": unitJSON("fused_residual"),
		},
		wantModules: []string{"unit-fused.json", "unit-fused_residual.json"},
	}, {
		name:       "no kernel named the unit",
		modules:    []cudacapture.Module{{DriverID: 22}, {DriverID: 23}},
		activities: []captureActivity{kernel(22, "memset_kernel"), kernel(23, "fused_gate")},
		attachments: map[string][]byte{
			"unit-fused_gate.json":  unitJSON("fused_gate"),
			"unit-fused_scale.json": unitJSON("fused_scale"),
		},
		wantModules: []string{"", "unit-fused_gate.json"},
	}, {
		name:        "malformed unit",
		modules:     []cudacapture.Module{{DriverID: 22}},
		attachments: map[string][]byte{"unit-broken.json": []byte("{")},
		wantModules: []string{""},
	}} {
		t.Run(tt.name, func(t *testing.T) {
			c := &cudacapture.Capture{Modules: tt.modules}
			attachUnits(c, tt.activities, tt.attachments)
			if c.Code.Unit != tt.wantCapture {
				t.Errorf("capture unit = %q, want %q", c.Code.Unit, tt.wantCapture)
			}
			for i, want := range tt.wantModules {
				if got := c.Modules[i].Code.Unit; got != want {
					t.Errorf("module %d unit = %q, want %q", i, got, want)
				}
			}
		})
	}
}
