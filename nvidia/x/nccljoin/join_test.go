package nccljoin

import (
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

func record(rank int, sequence uint64) ncclinspector.Record {
	return ncclinspector.Record{Header: ncclinspector.Header{ID: "comm", Rank: rank, Ranks: 2, Nodes: 1}, Metadata: ncclinspector.Metadata{Version: "v4.1", Mechanism: "ring", Hostname: "host", PID: rank + 1, TimestampMicros: 100}, Collective: &ncclinspector.Collective{Name: "AllReduce", Sequence: sequence, Bytes: 8, Micros: 4, TimingSource: "kernel_gpu"}}
}

func TestJoinFacts(t *testing.T) {
	opts := Options{Scope: Scope{"capture", "job", "epoch"}}
	captures := []Capture{{ID: "b", Records: []ncclinspector.Record{record(1, 2), record(0, 1)}}, {ID: "a", Partial: true, Records: []ncclinspector.Record{record(0, 1)}}}
	got, err := Join(captures, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Collectives) != 2 || len(got.Collectives[0].Observations) != 2 || !reflect.DeepEqual(got.Collectives[0].Unobserved, []int{1}) || !got.Captures[0].Partial {
		t.Fatalf("facts lost: %+v", got)
	}
	captures[0], captures[1] = captures[1], captures[0]
	other, err := Join(captures, opts)
	if err != nil || !reflect.DeepEqual(got, other) {
		t.Fatalf("order dependent: %v", err)
	}
	captures[0].Records[0].Collective.Micros = 99
	if got.Collectives[0].Observations[0].Collective.Micros != 4 {
		t.Fatal("retained input alias")
	}
}

func TestJoinRefusals(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*Capture, *Options)
	}{
		{"scope", func(c *Capture, o *Options) { o.Scope.Epoch = "" }},
		{"rank", func(c *Capture, o *Options) { c.Records[0].Header.Rank = 2 }},
		{"membership", func(c *Capture, o *Options) { o.Expected = map[string][]int{"comm": {0}} }},
		{"duplicate conflict", func(c *Capture, o *Options) {
			r := record(0, 1)
			r.Collective.Micros++
			c.Records = append(c.Records, r)
		}},
		{"p2p", func(c *Capture, o *Options) {
			c.Records[0].Collective = nil
			c.Records[0].PointToPoint = &ncclinspector.PointToPoint{Name: "Send", Peer: 1}
		}},
		{"bounds", func(c *Capture, o *Options) { o.MaxBytes = 1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := Capture{ID: "a", Records: []ncclinspector.Record{record(0, 1)}}
			o := Options{Scope: Scope{"capture", "job", "epoch"}}
			tc.change(&c, &o)
			if _, err := Join([]Capture{c}, o); err == nil {
				t.Fatal("accepted invalid input")
			}
		})
	}
}
