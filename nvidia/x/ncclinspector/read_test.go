package ncclinspector

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func fixture(t testing.TB, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name + ".jsonl")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestOfficialRecords(t *testing.T) {
	for _, name := range []string{"collective", "stats", "verbose"} {
		t.Run(name, func(t *testing.T) {
			b := fixture(t, name)
			rs, err := Read(bytes.NewReader(b), Limits{})
			if err != nil {
				t.Fatal(err)
			}
			if len(rs) != 1 || !bytes.Equal(rs[0].Raw, b) || rs[0].Metadata.Version != "v4.1" {
				t.Fatalf("record identity: %+v", rs)
			}
			r := rs[0]
			switch name {
			case "collective":
				if r.Header.ID != "0x7f8c496ae9f661" || r.Header.Rank != 2 || r.Metadata.PID != 1639453 || r.Collective.Sequence != 1407 || r.Collective.Bytes != 17179869184 || r.Collective.Micros != 61974 || r.Collective.TimingSource != "" || r.Stats != nil {
					t.Fatalf("literal collective example: %+v", r)
				}
			case "stats":
				if r.Stats.Collectives != 1024 || r.Stats.CollectiveDroppedTotal != 4096 || r.Stats.CollectiveDropped != 512 || r.Collective != nil || r.PointToPoint != nil {
					t.Fatalf("drop information erased: %+v", r)
				}
			case "verbose":
				if !bytes.Contains(r.Raw, []byte(`"event_trace_sn"`)) || !bytes.Contains(r.Raw, []byte(`"event_trace_ts"`)) || r.Collective.Name != "ReduceScatter" || r.Collective.Sequence != 1231 || r.Collective.TimingSource != "kernel_gpu" {
					t.Fatalf("verbose source fields missing: %+v", r)
				}
			}
		})
	}
}

func TestRecordRefusals(t *testing.T) {
	b := string(fixture(t, "collective"))
	for _, tc := range []struct{ name, old, new string }{
		{"version", `"v4.1"`, `"v4.0"`},
		{"missing_version", `"inspector_output_format_version"`, `"other_version"`},
		{"duplicate", `"rank":2`, `"rank":2,"rank":1`},
		{"alias", `"rank":2`, `"Rank":1,"rank":2`},
		{"unicode_alias", `"rank":2`, `"ran\u212a":1,"rank":2`},
		{"missing_sequence", `"coll_sn":1407,`, ``},
		{"float_sequence", `"coll_sn":1407`, `"coll_sn":1407.0`},
		{"negative_sequence", `"coll_sn":1407`, `"coll_sn":-1`},
		{"overflow_sequence", `"coll_sn":1407`, `"coll_sn":18446744073709551616`},
		{"null_sequence", `"coll_sn":1407`, `"coll_sn":null`},
		{"empty_id", `"id":"0x7f8c496ae9f661"`, `"id":""`},
		{"rank_range", `"rank":2`, `"rank":8`},
		{"pid", `"pid":1639453`, `"pid":0`},
		{"nodes", `"nnodes":1`, `"nnodes":9`},
		{"unknown_operation", `"AllReduce"`, `"MadeUp"`},
		{"negative_bandwidth", `"coll_algobw_gbs":277.210914`, `"coll_algobw_gbs":-1`},
		{"infinite_bandwidth", `"coll_algobw_gbs":277.210914`, `"coll_algobw_gbs":1e999`},
		{"null_header", `"header":{`, `"header":null,"other":{`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !strings.Contains(b, tc.old) {
				t.Fatal("control substitution missing")
			}
			if _, err := Read(strings.NewReader(strings.Replace(b, tc.old, tc.new, 1)), Limits{}); err == nil {
				t.Fatal("accepted malformed record")
			}
		})
	}
	for _, bad := range []string{"", "\n", "{}\n", "[]\n", "null\n", strings.TrimSuffix(b, "\n"), b[:len(b)-4] + "\n", b + "{}", b[:len(b)-1] + " {}\n"} {
		if _, err := Read(strings.NewReader(bad), Limits{}); err == nil {
			t.Fatalf("accepted malformed capture %q", bad)
		}
	}
	stats := string(fixture(t, "stats"))
	stats = strings.Replace(stats, `"coll_dropped_since_last_dump":512`, `"coll_dropped_since_last_dump":4097`, 1)
	if _, err := Read(strings.NewReader(stats), Limits{}); err == nil {
		t.Fatal("accepted impossible drop counters")
	}
	stats = string(fixture(t, "stats"))
	stats = strings.Replace(stats, `"coll_dropped_total":4096`, `"coll_dropped_total":null`, 1)
	if _, err := Read(strings.NewReader(stats), Limits{}); err == nil {
		t.Fatal("accepted unavailable counter as zero")
	}
}

// This is a constructed record, not measured output. Its field names and
// operation names come from the pinned inspectorCompletedP2p producer.
func p2pFixture(t testing.TB) []byte {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(fixture(t, "collective"), &m); err != nil {
		t.Fatal(err)
	}
	delete(m, "coll_perf")
	m["p2p_perf"] = json.RawMessage(`{"p2p":"Send","p2p_sn":1407,"p2p_peer":1,"p2p_msg_size_bytes":16,"p2p_exec_time_us":1,"p2p_timing_source":"kernel_gpu","p2p_algobw_gbs":0.016,"p2p_busbw_gbs":0.016}`)
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return append(b, '\n')
}

func TestSequenceDomainsAndUnknown(t *testing.T) {
	coll := fixture(t, "collective")
	p2p := p2pFixture(t)
	other := bytes.Replace(coll, []byte(`"rank":2`), []byte(`"rank":3`), 1)
	other = bytes.Replace(other, []byte(`"id":"0x7f8c496ae9f661"`), []byte(`"id":"0xother"`), 1)
	b := append(append(append([]byte(nil), coll...), p2p...), other...)
	r, err := Read(bytes.NewReader(b), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(r) != 3 || r[0].Collective.Sequence != 1407 || r[1].PointToPoint.Sequence != 1407 || r[1].PointToPoint.Peer != 1 || r[2].Header.ID == r[0].Header.ID || r[2].Header.Rank == r[0].Header.Rank {
		t.Fatalf("sequence domains collapsed: %+v", r)
	}
	for _, bad := range []string{strings.Replace(string(p2p), `"p2p_peer":1`, `"p2p_peer":8`, 1), strings.Replace(string(p2p), `"p2p_sn":1407`, `"p2p_sn":null`, 1)} {
		if _, err := Read(strings.NewReader(bad), Limits{}); err == nil {
			t.Fatal("accepted malformed P2P identity")
		}
	}
	var m map[string]json.RawMessage
	json.Unmarshal(coll, &m)
	m["p2p_perf"] = json.RawMessage(`{}`)
	multi, _ := json.Marshal(m)
	if _, err := Read(bytes.NewReader(append(multi, '\n')), Limits{}); err == nil {
		t.Fatal("accepted multiple bodies")
	}
	unknown := bytes.Replace(coll, []byte(`"header"`), []byte(`"future":{"sampled":null},"header"`), 1)
	r, err = Read(bytes.NewReader(unknown), Limits{})
	if err != nil || !bytes.Equal(r[0].Raw, unknown) {
		t.Fatalf("unknown field lost: %v", err)
	}
}

func TestLimits(t *testing.T) {
	b := fixture(t, "collective")
	for _, l := range []Limits{{Bytes: len(b) - 1}, {Line: len(b) - 2}, {Bytes: -1}, {Records: -1}, {Bytes: 16<<20 + 1}, {Line: 1<<20 + 1}, {Records: 262145}} {
		if _, err := Read(bytes.NewReader(b), l); err == nil {
			t.Fatalf("accepted invalid bound %+v", l)
		}
	}
	if _, err := Read(bytes.NewReader(b), Limits{Bytes: len(b), Line: len(b) - 1, Records: 1}); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(bytes.NewReader(append(append([]byte(nil), b...), b...)), Limits{Records: 1}); err == nil {
		t.Fatal("record bound ignored")
	}
	if _, err := Read(nil, Limits{}); err == nil {
		t.Fatal("nil reader")
	}
	deep := `{"future":` + strings.Repeat("[", 34) + "0" + strings.Repeat("]", 34) + "}\n"
	if _, err := Read(strings.NewReader(deep), Limits{}); err == nil {
		t.Fatal("nesting bound ignored")
	}
}

func FuzzRead(f *testing.F) {
	for _, name := range []string{"collective", "stats", "verbose"} {
		f.Add(fixture(f, name))
	}
	f.Add(p2pFixture(f))
	f.Add([]byte("{}\n"))
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 32768 {
			return
		}
		r, err := Read(bytes.NewReader(b), Limits{Bytes: 32768, Line: 32768, Records: 128})
		if err == nil {
			var rebuilt []byte
			for _, x := range r {
				rebuilt = append(rebuilt, x.Raw...)
				if x.Metadata.Version != "v4.1" {
					t.Fatal("version guard lost")
				}
			}
			if !bytes.Equal(rebuilt, b) {
				t.Fatal("raw framing lost")
			}
		}
	})
}
