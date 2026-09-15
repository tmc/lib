package cudadevicetable

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNormalize(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"underscore", "sm_121", "sm_121"},
		{"no underscore", "sm121", "sm_121"},
		{"uppercase", "SM_121", "sm_121"},
		{"padded", "  sm_121  ", "sm_121"},
		{"feature suffix", "sm_121a", "sm_121"},
		{"feature suffix bare", "sm121a", "sm_121"},
		{"two digits", "sm_90", "sm_90"},
		{"two digits suffixed", "sm_90a", "sm_90"},
		{"empty", "", ""},
		{"letters only", "sm_", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Normalize(tt.in); got != tt.want {
				t.Errorf("Normalize(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// TestLookupMissIsNotASubstitute pins the contract that a miss is a miss. The
// tempting failure is to fall back to a neighbouring architecture, which
// yields a number that is plausible, wrong, and unattributable.
func TestLookupMissIsNotASubstitute(t *testing.T) {
	if _, ok := Lookup("sm_75"); ok {
		t.Fatal("sm_75 resolved; no sm_75 device has been probed, so the table must report a miss")
	}
	if _, ok := Lookup("sm_121"); !ok {
		t.Fatal("sm_121 did not resolve; the embedded table should carry the probed GB10 record")
	}
	// The feature suffix selects an instruction set, not different hardware.
	a, ok := Lookup("sm_121a")
	if !ok {
		t.Fatal("sm_121a did not resolve to its base architecture")
	}
	if a.Architecture != "sm_121" {
		t.Errorf("sm_121a resolved to %q, want sm_121", a.Architecture)
	}
}

// TestProbedAttrRefusesUnverified is the point of the package: a value nobody
// measured must not be returnable through the accessor used where a wrong
// value would flow onward silently.
func TestProbedAttrRefusesUnverified(t *testing.T) {
	a, ok := Lookup("sm_121")
	if !ok {
		t.Fatal("sm_121 missing")
	}
	if _, err := a.ProbedAttr("maxSharedMemoryPerMultiprocessor"); err != nil {
		t.Errorf("probed attribute rejected: %v", err)
	}
	err := errString(a.ProbedAttr("regAllocationUnit"))
	if err == "" {
		t.Fatal("regAllocationUnit is unverified but ProbedAttr returned it")
	}
	if !strings.Contains(err, "unverified") {
		t.Errorf("error %q does not say the value is unverified", err)
	}
	if errString(a.ProbedAttr("noSuchAttribute")) == "" {
		t.Error("ProbedAttr returned a missing attribute without error")
	}
}

func errString(_ int64, err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// TestEmbeddedTableValidates re-runs the rules against the shipped file. init
// already panics on a bad table, but a test states the requirement where
// someone editing devices.json by hand will see it.
func TestEmbeddedTableValidates(t *testing.T) {
	if err := table.Validate(); err != nil {
		t.Fatalf("embedded devices.json: %v", err)
	}
	if len(Architectures()) == 0 {
		t.Fatal("embedded table is empty")
	}
}

// TestValidateRejects is the guard that makes provenance structural rather
// than editorial. Each case is a way somebody could record a device fact
// dishonestly, and each must be refused.
func TestValidateRejects(t *testing.T) {
	tests := []struct {
		name string
		json string
		want string
	}{
		{
			name: "probed without an instrument",
			json: `{"schema":1,"architectures":{"sm_121":{"architecture":"sm_121",
				"probe":{"device":"d","computeCapability":"12.1","host":"h","probedAt":"p","tool":"t"},
				"attributes":{"x":{"value":1,"source":"probed"}}}}}`,
			want: "names no instrument",
		},
		{
			name: "probed naming something that is not an instrument",
			json: `{"schema":1,"architectures":{"sm_121":{"architecture":"sm_121",
				"probe":{"device":"d","computeCapability":"12.1","host":"h","probedAt":"p","tool":"t"},
				"attributes":{"x":{"value":1,"source":"probed","instrument":"the datasheet"}}}}}`,
			want: "want a DevAttr or cudaDeviceProp. prefix",
		},
		{
			name: "unverified borrowing an instrument name",
			json: `{"schema":1,"architectures":{"sm_121":{"architecture":"sm_121",
				"attributes":{"x":{"value":1,"source":"unverified","note":"n","instrument":"DevAttrWarpSize"}}}}}`,
			want: "names instrument",
		},
		{
			name: "unverified with no explanation",
			json: `{"schema":1,"architectures":{"sm_121":{"architecture":"sm_121",
				"attributes":{"x":{"value":1,"source":"unverified"}}}}}`,
			want: "with no note explaining why",
		},
		{
			name: "no source at all",
			json: `{"schema":1,"architectures":{"sm_121":{"architecture":"sm_121",
				"attributes":{"x":{"value":1}}}}}`,
			want: "want probed, derived or unverified",
		},
		{
			name: "probed values with no probe record",
			json: `{"schema":1,"architectures":{"sm_121":{"architecture":"sm_121",
				"attributes":{"x":{"value":1,"source":"probed","instrument":"DevAttrWarpSize"}}}}}`,
			want: "no probe record",
		},
		{
			name: "probe record with an empty field",
			json: `{"schema":1,"architectures":{"sm_121":{"architecture":"sm_121",
				"probe":{"device":"d","computeCapability":"12.1","host":"","probedAt":"p","tool":"t"},
				"attributes":{"x":{"value":1,"source":"probed","instrument":"DevAttrWarpSize"}}}}}`,
			want: "empty host",
		},
		{
			name: "key disagreeing with the record",
			json: `{"schema":1,"architectures":{"sm_121":{"architecture":"sm_90",
				"attributes":{"x":{"value":1,"source":"unverified","note":"n"}}}}}`,
			want: "records Architecture",
		},
		{
			name: "unnormalized key",
			json: `{"schema":1,"architectures":{"sm121":{"architecture":"sm121",
				"attributes":{"x":{"value":1,"source":"unverified","note":"n"}}}}}`,
			want: "is not normalized",
		},
		{
			name: "unknown schema",
			json: `{"schema":2,"architectures":{}}`,
			want: "unsupported schema",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var tbl Table
			if err := json.Unmarshal([]byte(tt.json), &tbl); err != nil {
				t.Fatalf("test fixture is not valid JSON: %v", err)
			}
			err := tbl.Validate()
			if err == nil {
				t.Fatalf("Validate accepted %s", tt.name)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Validate said %q, want it to mention %q", err, tt.want)
			}
		})
	}
}

// TestValidateAcceptsAnHonestRecord is the positive control: the rejections
// above mean nothing if Validate refuses everything.
func TestValidateAcceptsAnHonestRecord(t *testing.T) {
	const good = `{"schema":1,"architectures":{"sm_121":{"architecture":"sm_121",
		"probe":{"device":"d","computeCapability":"12.1","host":"h","probedAt":"p","tool":"t"},
		"attributes":{
			"a":{"value":1,"source":"probed","instrument":"DevAttrWarpSize"},
			"b":{"value":2,"source":"probed","instrument":"cudaDeviceProp.l2CacheSize"},
			"c":{"value":3,"source":"derived","note":"a*b"},
			"d":{"value":4,"source":"unverified","note":"no instrument exposes it"}}}}}`
	var tbl Table
	if err := json.Unmarshal([]byte(good), &tbl); err != nil {
		t.Fatalf("fixture: %v", err)
	}
	if err := tbl.Validate(); err != nil {
		t.Errorf("Validate rejected an honest record: %v", err)
	}
}

// TestOptinIsCapacityLessReservation turns a comment into a check. The
// KernelResource doc in x/cudaoccupancy/nvinfo.go observes that
// MaxSharedMemoryOptin is 1024 bytes below the per-SM capacity and calls the
// mechanism "unproven". The driver reports the reservation directly, so on any
// device that reports all three the identity is now arithmetic rather than a
// conjecture -- and a device that breaks it is worth knowing about.
func TestOptinIsCapacityLessReservation(t *testing.T) {
	checked := 0
	for _, arch := range Architectures() {
		rec, ok := Lookup(arch)
		if !ok {
			continue
		}
		capacity, err1 := rec.ProbedAttr("maxSharedMemoryPerMultiprocessor")
		optin, err2 := rec.ProbedAttr("maxSharedMemoryPerBlockOptin")
		reserved, err3 := rec.ProbedAttr("reservedSharedMemoryPerBlock")
		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}
		checked++
		if got := capacity - reserved; got != optin {
			t.Errorf("%s: capacity %d - reserved %d = %d, but the device reports optin %d; "+
				"the per-block opt-in maximum is not the per-SM capacity less the driver "+
				"reservation on this device", arch, capacity, reserved, got, optin)
		}
	}
	if checked == 0 {
		t.Skip("no probed device reports capacity, optin and reservation together")
	}
}
