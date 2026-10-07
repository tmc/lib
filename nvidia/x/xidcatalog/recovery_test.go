package xidcatalog

import (
	"fmt"
	"strings"
	"testing"
)

const recoveryFixture = "NVRM: Xid (PCI:0008:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x1 (GPU Reset Required)"

func TestRecoveryTransition(t *testing.T) {
	r, err := ParseRecoveryTransition(recoveryFixture)
	if err != nil {
		t.Fatal(err)
	}
	if r.XID != 154 || r.PCI != "0008:01:00" || r.Message != recoveryFixture || r.From.Hex != "0x0" || r.From.Value != 0 || r.From.Text != "None" || r.From.State != RecoveryNone || !r.From.Recognized || r.To.Hex != "0x1" || r.To.Value != 1 || r.To.State != RecoveryGPUReset {
		t.Fatalf("raw transition lost %+v", r)
	}
	for _, test := range []struct {
		text  string
		state RecoveryText
	}{{"None", RecoveryNone}, {"Drain P2P", RecoveryDrainP2P}, {"Drain and Reset", RecoveryDrainAndReset}, {"GPU Reset Required", RecoveryGPUReset}, {"Node Reboot Required", RecoveryNodeReboot}} {
		state, known := RecognizeRecoveryText(test.text)
		if !known || state != test.state {
			t.Fatal(test)
		}
		message := fmt.Sprintf("Xid 154 GPU recovery action changed from 0x0 (None) to 0xFFFFFFFF (%s)", test.text)
		got, err := ParseRecoveryTransition(message)
		if err != nil || got.PCI != "" || got.To.Value != 0xffffffff || got.To.Hex != "0xFFFFFFFF" || got.To.Text != test.text || got.To.State != test.state {
			t.Fatalf("independent tokens: %+v %v", got, err)
		}
	}
	message := "Xid 154 GPU recovery action changed from 0xab (Future old) to 0xcd (Future new)"
	unknown, err := ParseRecoveryTransition(message)
	if err != nil || unknown.From.Recognized || unknown.To.Recognized || unknown.From.State != UnknownRecoveryText || unknown.To.Value != 0xcd || unknown.To.Text != "Future new" {
		t.Fatalf("unknown token lost %+v %v", unknown, err)
	}
}

func TestRecoveryRefusals(t *testing.T) {
	for _, test := range []struct{ name, message string }{
		{"occurrence_only", "NVRM: Xid (PCI:0008:01:00): 154"},
		{"missing_from", strings.Replace(recoveryFixture, "from 0x0 (None)", "", 1)},
		{"missing_to", strings.Split(recoveryFixture, " to ")[0]},
		{"truncated", recoveryFixture[:len(recoveryFixture)-1]},
		{"multiple", recoveryFixture + recoveryFixture},
		{"multiline", recoveryFixture + "\n" + recoveryFixture},
		{"overflow", strings.Replace(recoveryFixture, "0x1", "0x100000000", 1)},
		{"signed", strings.Replace(recoveryFixture, "0x1", "-0x1", 1)},
		{"decimal", strings.Replace(recoveryFixture, "0x1", "1", 1)},
		{"missing_text", strings.Replace(recoveryFixture, "(GPU Reset Required)", "()", 1)},
		{"extra_parenthesis", recoveryFixture + " (None)"},
		{"text_whitespace", strings.Replace(recoveryFixture, "(None)", "( None)", 1)},
		{"pci_device", strings.Replace(recoveryFixture, "01:00", "01:ff", 1)},
		{"wrong_xid", strings.Replace(recoveryFixture, "154,", "95,", 1)},
		{"control", recoveryFixture + "\x00"},
		{"byte_bound", strings.Repeat("x", MaxRecoveryMessageBytes+1)},
		{"invalid_utf8", recoveryFixture + "\xff"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := ParseRecoveryTransition(test.message); err == nil {
				t.Fatal("incomplete/invalid event accepted")
			}
		})
	}
}

func FuzzRecoveryTransition(f *testing.F) {
	f.Add(recoveryFixture)
	f.Add("Xid 154 GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)")
	f.Add("Xid 154")
	f.Fuzz(func(t *testing.T, s string) {
		got, err := ParseRecoveryTransition(s)
		if err == nil && (got.XID != 154 || got.Message != s || got.From.Hex == "" || got.To.Hex == "" || len(s) > MaxRecoveryMessageBytes) {
			t.Fatal("accepted incomplete event")
		}
	})
}
