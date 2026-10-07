package nccljoin

import (
	"bytes"
	"encoding/json"
	"github.com/tmc/lib/nvidia/x/ncclinspector"
	"reflect"
	"testing"
)

func TestStatisticsOrderIndependent(t *testing.T) {
	a := record(0, 1)
	a.Collective = nil
	a.Stats = &ncclinspector.DumpStats{Collectives: 1, CollectiveDroppedTotal: 7, CollectiveDropped: 2}
	b := a
	b.Stats = &ncclinspector.DumpStats{Collectives: 2, CollectiveDroppedTotal: 9, CollectiveDropped: 2}
	opts := Options{Scope: Scope{"capture", "job", "epoch"}}
	first, err := Join([]Capture{{ID: "a", Partial: true, Records: []ncclinspector.Record{a, b}}}, opts)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Join([]Capture{{ID: "a", Partial: true, Records: []ncclinspector.Record{b, a}}}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("statistics depend on input order")
	}
	if len(first.Captures[0].Statistics) != 2 || len(first.Collectives) != 0 || !first.Captures[0].Partial {
		t.Fatal("statistics provenance lost")
	}
	a.Stats.CollectiveDroppedTotal = 99
	if first.Captures[0].Statistics[0].Stats.CollectiveDroppedTotal == 99 {
		t.Fatal("statistics aliased")
	}
}

func TestJoinRawAndVersion(t *testing.T) {
	r := record(0, 1)
	b, _ := json.Marshal(r)
	r.Raw = append(b, '\n')
	opts := Options{Scope: Scope{"capture", "job", "epoch"}}
	got, err := Join([]Capture{{ID: "a", Truncated: true, Records: []ncclinspector.Record{r}}}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got.Collectives[0].Observations[0].Raw, r.Raw) || !got.Captures[0].Truncated {
		t.Fatal("raw facts lost")
	}
	r.Collective.Micros++
	if _, err = Join([]Capture{{ID: "a", Records: []ncclinspector.Record{r}}}, opts); err == nil {
		t.Fatal("accepted stale raw")
	}
	r.Raw = nil
	r.Metadata.Version = "v999"
	if _, err = Join([]Capture{{ID: "a", Records: []ncclinspector.Record{r}}}, opts); err == nil {
		t.Fatal("accepted unknown version")
	}
}

func TestJoinGroupLimit(t *testing.T) {
	o := Options{Scope: Scope{"capture", "job", "epoch"}, MaxGroups: 1}
	if _, err := Join([]Capture{{ID: "a", Records: []ncclinspector.Record{record(0, 1)}}}, o); err != nil {
		t.Fatal(err)
	}
	if _, err := Join([]Capture{{ID: "a", Records: []ncclinspector.Record{record(0, 1), record(0, 2)}}}, o); err == nil {
		t.Fatal("accepted excess groups")
	}
}

func FuzzJoin(f *testing.F) {
	b, _ := json.Marshal(record(0, 1))
	f.Add(append(b, '\n'))
	f.Fuzz(func(t *testing.T, b []byte) {
		r, err := ncclinspector.Read(bytes.NewReader(b), ncclinspector.Limits{Bytes: 4096, Line: 4096, Records: 8})
		if err != nil {
			return
		}
		o := Options{Scope: Scope{"capture", "job", "epoch"}, MaxRecords: 16, MaxRanks: 8, MaxGroups: 8, MaxBytes: 16384}
		got, err := Join([]Capture{{ID: "a", Records: r}}, o)
		if err != nil {
			return
		}
		for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
			r[i], r[j] = r[j], r[i]
		}
		other, err := Join([]Capture{{ID: "a", Records: r}}, o)
		if err != nil || !reflect.DeepEqual(got, other) {
			t.Fatal("input order changed result")
		}
	})
}
