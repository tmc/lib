package nccljoin

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

func TestRankProcessIdentityAcrossSequences(t *testing.T) {
	for _, change := range []struct {
		name string
		edit func(*ncclinspector.Record)
	}{
		{"pid", func(r *ncclinspector.Record) { r.Metadata.PID++ }},
		{"host", func(r *ncclinspector.Record) { r.Metadata.Hostname = "other" }},
	} {
		t.Run(change.name, func(t *testing.T) {
			a, b := record(0, 1), record(0, 2)
			opts := Options{Scope: Scope{"capture", "job", "epoch"}}
			if _, err := Join([]Capture{{ID: "a", Records: []ncclinspector.Record{a, b}}}, opts); err != nil {
				t.Fatal(err)
			}
			change.edit(&b)
			if _, err := Join([]Capture{{ID: "a", Records: []ncclinspector.Record{a, b}}}, opts); err == nil {
				t.Fatal("accepted conflicting rank process within one epoch")
			}
		})
	}
}

func TestGroupMembershipBudget(t *testing.T) {
	a, b, c := record(0, 1), record(0, 2), record(0, 3)
	opts := Options{Scope: Scope{"capture", "job", "epoch"}, MaxRecords: 4}
	if _, err := Join([]Capture{{ID: "a", Records: []ncclinspector.Record{a, b}}}, opts); err != nil {
		t.Fatal(err)
	}
	if _, err := Join([]Capture{{ID: "a", Records: []ncclinspector.Record{a, b, c}}}, opts); err == nil {
		t.Fatal("accepted membership cross product beyond record budget")
	}
}
