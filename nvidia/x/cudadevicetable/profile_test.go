package cudadevicetable

import (
	"strings"
	"testing"
)

func TestLookupProfile(t *testing.T) {
	tests := []struct {
		name      string
		target    string
		product   string
		sms       int64
		optin     int64
		tcgen05   bool
		wgmma     bool
		probedSMs bool
	}{
		{"sm_90a", "sm_90a", "H200", 132, 227 << 10, false, true, false},
		{"H100", "sm_90a", "H100", 132, 227 << 10, false, true, false},
		{"sm_100a", "sm_100a", "B200", 148, 227 << 10, true, false, false},
		{"sm_103a", "sm_103a", "GB300", 160, 227 << 10, true, false, false},
		{"NVIDIA B300", "sm_103a", "B300", 160, 227 << 10, true, false, false},
		{"sm_120a", "sm_120a", "RTX PRO 6000", 188, 99 << 10, false, false, false},
		{"NVIDIA GeForce RTX 5090", "sm_120a", "RTX 5090", 170, 99 << 10, false, false, false},
		{"sm_121a", "sm_121a", "GB10", 48, 99 << 10, false, false, true},
		{"NVIDIA GB10", "sm_121a", "GB10", 48, 99 << 10, false, false, true},
		{"sm_110a", "sm_110a", "Thor", 20, 227 << 10, true, false, false},
		{"sm_89", "sm_89", "L40S", 142, 99 << 10, false, false, false},
		{"NVIDIA L4", "sm_89", "L4", 58, 99 << 10, false, false, false},
		{"Orin", "sm_87", "Orin", 16, 163 << 10, false, false, false},
		{"NVIDIA A10", "sm_86", "A10", 72, 99 << 10, false, false, false},
		{"A100", "sm_80", "A100", 108, 163 << 10, false, false, false},
		{"sm_75", "sm_75", "T4", 40, 64 << 10, false, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := LookupProfile(tt.name)
			if err != nil {
				t.Fatal(err)
			}
			if p.Target != tt.target || p.Product != tt.product {
				t.Errorf("profile = %s/%s, want %s/%s", p.Target, p.Product, tt.target, tt.product)
			}
			if p.SMs.Value != tt.sms || p.SharedPerBlockOptin.Value != tt.optin {
				t.Errorf("SMs, shared opt-in = %d, %d; want %d, %d", p.SMs.Value, p.SharedPerBlockOptin.Value, tt.sms, tt.optin)
			}
			if got := p.SMs.Source == Probed; got != tt.probedSMs {
				t.Errorf("SMs source = %s, want probed %v", p.SMs.Source, tt.probedSMs)
			}
			edge := tt.target == "sm_120a" || tt.target == "sm_121a"
			// The mmasync probe is a BF16 MMA, which sm_75 lacks.
			mmasync := tt.target != "sm_75"
			for feature, want := range map[string]bool{"tcgen05": tt.tcgen05, "wgmma": tt.wgmma, "mmasync": mmasync, "mmasyncmxf4": edge} {
				got, err := p.Supports(feature)
				if err != nil {
					t.Fatal(err)
				}
				if got != want {
					t.Errorf("Supports(%q) = %v, want %v", feature, got, want)
				}
			}
			if (p.TMEMPerSM.Value > 0) != tt.tcgen05 {
				t.Errorf("TMEM = %d bytes, but tcgen05 = %v", p.TMEMPerSM.Value, tt.tcgen05)
			}
		})
	}
}

func TestLookupProfileMiss(t *testing.T) {
	// CUDA 13 ptxas no longer accepts sm_70, so it has no profile.
	_, err := LookupProfile("sm_70")
	if err == nil || !strings.Contains(err.Error(), "sm_103a") {
		t.Errorf("LookupProfile(sm_70) error = %v, want a miss listing the known names", err)
	}
}

// TestDatasheetProvenance checks that every datasheet value is unverified
// and explains itself, so that only a probe can make a value [Probed].
func TestDatasheetProvenance(t *testing.T) {
	for _, p := range datasheet {
		for name, v := range p.values() {
			if v.Source != Unverified || v.Note == "" {
				t.Errorf("%s/%s: %s = %+v, want unverified with a note", p.Target, p.Product, name, *v)
			}
		}
		if _, ok := LookupTarget(p.Target); !ok {
			t.Errorf("%s/%s: target has no probed features", p.Target, p.Product)
		}
	}
	for _, name := range ProfileNames() {
		p, err := LookupProfile(name)
		if err != nil {
			t.Fatal(err)
		}
		if err := p.Validate(); err != nil {
			t.Error(err)
		}
	}
}

// TestProbeOverlaysOnlyItsProduct checks that the GB10 probe replaces GB10's
// datasheet values and no other product's.
func TestProbeOverlaysOnlyItsProduct(t *testing.T) {
	gb10, err := LookupProfile("GB10")
	if err != nil {
		t.Fatal(err)
	}
	if gb10.SharedPerBlockOptin.Source != Probed || gb10.SharedPerBlockOptin.Instrument != "DevAttrMaxSharedMemoryPerBlockOptin" {
		t.Errorf("GB10 shared opt-in = %+v, want the probed value", gb10.SharedPerBlockOptin)
	}
	if gb10.MemoryGBps.Source != Unverified {
		t.Errorf("GB10 memory bandwidth = %+v; no driver attribute measures it", gb10.MemoryGBps)
	}
	for _, p := range TargetProfiles("sm_120a") {
		if p.SMs.Source == Probed {
			t.Errorf("%s: SM count %+v came from another product's probe", p.Product, p.SMs)
		}
	}
}
