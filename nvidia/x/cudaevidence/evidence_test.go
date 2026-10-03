package cudaevidence_test

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

func number(n float64) *float64 { return &n }
func identity() cudaevidence.Identity {
	return cudaevidence.Identity{ArtifactDigest: strings.Repeat("1", 64), Entry: "kernel", Target: "sm_70", CompilerKey: strings.Repeat("2", 64), WorkloadKey: strings.Repeat("3", 64)}
}
func record(class cudaevidence.Class) cudaevidence.Record {
	return cudaevidence.Record{Class: class, Number: number(0), Unit: "bytes", Basis: "producer record", Source: cudaevidence.Source{Identity: identity(), Producer: "test producer", Locator: "record/1"}}
}
func TestEvidenceZeroValueIsUnread(t *testing.T) {
	var v cudaevidence.Value
	r := v.Record()
	if r.Class != cudaevidence.Unread || r.Number != nil || r.Reason == "" {
		t.Fatalf("zero record = %+v", r)
	}
	if v.CheckClaim(cudaevidence.Measured, identity()) == nil {
		t.Fatal("zero value satisfies numeric claim")
	}
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"class":"unread"`) || !strings.Contains(string(b), `"number":null`) {
		t.Fatalf("zero JSON = %s", b)
	}
}
func TestEvidenceRequiresBasisAndIdentity(t *testing.T) {
	changes := []struct {
		name   string
		change func(*cudaevidence.Record)
	}{
		{"basis", func(r *cudaevidence.Record) { r.Basis = " " }},
		{"producer", func(r *cudaevidence.Record) { r.Source.Producer = "" }},
		{"locator", func(r *cudaevidence.Record) { r.Source.Locator = "" }},
		{"artifact", func(r *cudaevidence.Record) { r.Source.Identity.ArtifactDigest = "" }},
		{"entry", func(r *cudaevidence.Record) { r.Source.Identity.Entry = " " }},
		{"target", func(r *cudaevidence.Record) { r.Source.Identity.Target = "" }},
		{"compiler", func(r *cudaevidence.Record) { r.Source.Identity.CompilerKey = "bad" }},
		{"workload", func(r *cudaevidence.Record) { r.Source.Identity.WorkloadKey = "" }},
	}
	for _, class := range []cudaevidence.Class{cudaevidence.Measured, cudaevidence.Derived, cudaevidence.Inferred, cudaevidence.Modeled} {
		t.Run(string(class), func(t *testing.T) {
			for _, tc := range changes {
				t.Run(tc.name, func(t *testing.T) {
					r := record(class)
					tc.change(&r)
					if _, err := cudaevidence.New(r); err == nil {
						t.Fatal("accepted missing basis/source/identity")
					}
				})
			}
		})
	}
}
func TestEvidenceRejectsNumericUnread(t *testing.T) {
	for _, n := range []float64{0, 1, math.NaN(), math.Inf(1)} {
		if _, err := cudaevidence.New(cudaevidence.Record{Class: cudaevidence.Unread, Number: number(n), Reason: "not sampled"}); err == nil {
			t.Fatalf("accepted unread number %v", n)
		}
	}
	if _, err := cudaevidence.New(cudaevidence.Record{Reason: "not sampled", Uncertainty: number(0)}); err == nil {
		t.Fatal("accepted numeric unread uncertainty")
	}
	if _, err := cudaevidence.New(cudaevidence.Record{}); err == nil {
		t.Fatal("accepted unread without reason")
	}
}
func TestEvidenceNumericValidation(t *testing.T) {
	for _, n := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		r := record(cudaevidence.Measured)
		r.Number = number(n)
		if _, err := cudaevidence.New(r); err == nil {
			t.Fatalf("accepted nonfinite %v", n)
		}
	}
	r := record(cudaevidence.Measured)
	r.Number = nil
	if _, err := cudaevidence.New(r); err == nil {
		t.Fatal("accepted missing number")
	}
	for _, n := range []float64{-1, math.NaN(), math.Inf(1)} {
		r := record(cudaevidence.Measured)
		r.Uncertainty = number(n)
		if _, err := cudaevidence.New(r); err == nil {
			t.Fatalf("accepted uncertainty %v", n)
		}
	}
	for _, loss := range []cudaevidence.Loss{{0, 0}, {0, 5}, {2, 5}, {5, 5}} {
		r := record(cudaevidence.Measured)
		r.Loss = &loss
		r.Uncertainty = number(0)
		if _, err := cudaevidence.New(r); err != nil {
			t.Fatal(err)
		}
	}
	r = record(cudaevidence.Measured)
	r.Loss = &cudaevidence.Loss{Dropped: 2, Total: 1}
	if _, err := cudaevidence.New(r); err == nil {
		t.Fatal("accepted impossible losses")
	}
	r = record(cudaevidence.Measured)
	r.Reason = "unread"
	if _, err := cudaevidence.New(r); err == nil {
		t.Fatal("accepted unread reason on available value")
	}
	r = record("decoded")
	if _, err := cudaevidence.New(r); err == nil {
		t.Fatal("accepted decoded peer class")
	}
}
func TestEvidenceRejectsIdentityMismatch(t *testing.T) {
	v, err := cudaevidence.New(record(cudaevidence.Measured))
	if err != nil {
		t.Fatal(err)
	}
	if err := v.CheckClaim(cudaevidence.Measured, identity()); err != nil {
		t.Fatal(err)
	}
	changes := []struct {
		name   string
		change func(*cudaevidence.Identity)
	}{
		{"artifact", func(id *cudaevidence.Identity) { id.ArtifactDigest = strings.Repeat("4", 64) }},
		{"entry", func(id *cudaevidence.Identity) { id.Entry = "other" }},
		{"target", func(id *cudaevidence.Identity) { id.Target = "sm_80" }},
		{"compiler", func(id *cudaevidence.Identity) { id.CompilerKey = strings.Repeat("5", 64) }},
		{"workload", func(id *cudaevidence.Identity) { id.WorkloadKey = strings.Repeat("6", 64) }},
	}
	for _, tc := range changes {
		t.Run(tc.name, func(t *testing.T) {
			id := identity()
			tc.change(&id)
			if v.CheckClaim(cudaevidence.Measured, id) == nil {
				t.Fatal("accepted mismatched identity")
			}
		})
	}
}
func TestEvidenceDoesNotPromoteInference(t *testing.T) {
	for _, class := range []cudaevidence.Class{cudaevidence.Derived, cudaevidence.Inferred, cudaevidence.Modeled} {
		v, err := cudaevidence.New(record(class))
		if err != nil {
			t.Fatal(err)
		}
		if v.CheckClaim(class, identity()) != nil {
			t.Fatal("refused matching class")
		}
		if v.CheckClaim(cudaevidence.Measured, identity()) == nil {
			t.Fatalf("promoted %s", class)
		}
	}
}
func TestEvidenceImmutableCopies(t *testing.T) {
	r := record(cudaevidence.Measured)
	r.Uncertainty = number(2)
	r.Loss = &cudaevidence.Loss{Dropped: 1, Total: 3}
	v, err := cudaevidence.New(r)
	if err != nil {
		t.Fatal(err)
	}
	*r.Number = 99
	*r.Uncertainty = 99
	r.Loss.Dropped = 99
	r.Source.Producer = "changed"
	got := v.Record()
	if *got.Number != 0 || *got.Uncertainty != 2 || got.Loss.Dropped != 1 || got.Source.Producer != "test producer" {
		t.Fatalf("constructor alias: %+v", got)
	}
	*got.Number = 88
	*got.Uncertainty = 88
	got.Loss.Dropped = 88
	got.Source.Identity.Entry = "changed"
	again := v.Record()
	if *again.Number != 0 || *again.Uncertainty != 2 || again.Loss.Dropped != 1 || again.Source.Identity.Entry != "kernel" {
		t.Fatalf("record alias: %+v", again)
	}
}
func TestEvidenceJSONRoundTrip(t *testing.T) {
	records := []cudaevidence.Record{{Reason: "counter unavailable"}}
	for _, class := range []cudaevidence.Class{cudaevidence.Measured, cudaevidence.Derived, cudaevidence.Inferred, cudaevidence.Modeled} {
		r := record(class)
		r.Uncertainty = number(0.5)
		r.Loss = &cudaevidence.Loss{Dropped: 2, Total: 10}
		records = append(records, r)
	}
	for _, r := range records {
		v, err := cudaevidence.New(r)
		if err != nil {
			t.Fatal(err)
		}
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		got, err := cudaevidence.Decode(b)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(v.Record(), got.Record()) {
			t.Fatalf("roundtrip %s: %+v", b, got.Record())
		}
	}
}
func TestEvidenceJSONRejectsInvalidRecords(t *testing.T) {
	b, err := json.Marshal(mustValue(t, record(cudaevidence.Measured)))
	if err != nil {
		t.Fatal(err)
	}
	inputs := []string{"", "null", "[]", "{", string(b) + ` {}`, string(b) + ` garbage`, strings.Replace(string(b), `"basis":`, `"unknown":1,"basis":`, 1), strings.Replace(string(b), `"producer":`, `"unknown":1,"producer":`, 1), strings.Replace(string(b), `"entry":`, `"unknown":1,"entry":`, 1), `{"class":"unread","number":0,"reason":"missing"}`, `{"class":"decoded"}`, strings.Replace(string(b), `"number":0`, `"number":1e999`, 1)}
	for _, s := range inputs {
		if _, err := cudaevidence.Decode([]byte(s)); err == nil {
			t.Fatalf("accepted JSON %q", s)
		}
	}
}
func mustValue(t *testing.T, r cudaevidence.Record) cudaevidence.Value {
	t.Helper()
	v, err := cudaevidence.New(r)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestEvidenceIdentityValidation(t *testing.T) {
	for _, field := range []string{"artifact", "compiler", "workload"} {
		for _, bad := range []string{"", strings.Repeat("1", 62), strings.Repeat("g", 64), strings.Repeat("A", 64)} {
			id := identity()
			switch field {
			case "artifact":
				id.ArtifactDigest = bad
			case "compiler":
				id.CompilerKey = bad
			case "workload":
				id.WorkloadKey = bad
			}
			if id.Validate() == nil {
				t.Fatalf("accepted %s digest %q", field, bad)
			}
			v := mustValue(t, record(cudaevidence.Measured))
			if v.CheckClaim(cudaevidence.Measured, id) == nil {
				t.Fatal("accepted invalid claimed identity")
			}
		}
	}
}
