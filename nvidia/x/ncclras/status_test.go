package ncclras

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"testing"
)

func statusFixture(t testing.TB) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/status.json")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestOfficialStatus(t *testing.T) {
	b := statusFixture(t)
	s, err := Read(bytes.NewReader(b), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if s.Version != "2.29.1" || s.CUDARuntime != 13000 || len(s.Communicators) != 1 || s.Communicators[0].Size != 2 || len(s.Communicators[0].Ranks) != 1 || len(s.Communicators[0].MissingRanks) != 1 {
		t.Fatalf("literal official population: %+v", s)
	}
	r := s.Communicators[0].Ranks[0]
	missing := s.Communicators[0].MissingRanks[0]
	if r.Rank != 0 || r.Host != "172.16.64.245" || r.PID != 1524344 || r.Counts["AllReduce"] != 0 || missing.Rank != 1 || missing.PID != 1524345 || !missing.Status.Unresponsive || missing.Status.ConsideredDead {
		t.Fatalf("literal official rank observations: %+v %+v", r, missing)
	}
	if !bytes.Equal(s.Raw, b) {
		t.Fatal("raw source changed")
	}
	// A separate constructed collection timeout is retained, never cleared.
	c := bytes.Replace(b, []byte(`"timeouts_count": 0`), []byte(`"timeouts_count": 3`), 1)
	s, err = Read(bytes.NewReader(c), Limits{})
	if err != nil || s.Collection.Timeouts != 3 {
		t.Fatalf("timeouts: %+v %v", s, err)
	}
}

func TestStatusRefusals(t *testing.T) {
	b := string(statusFixture(t))
	for _, tc := range []struct{ name, old, new string }{
		{"version", `"2.29.1"`, `"2.99.0"`},
		{"null_version", `"nccl_version": "2.29.1"`, `"nccl_version": null`},
		{"missing_version", `"nccl_version"`, `"other_version"`},
		{"duplicate_version", `"nccl_version": "2.29.1"`, `"nccl_version": "2.29.1", "nccl_version":"2.32.3"`},
		{"case_alias", `"nccl_version": "2.29.1"`, `"NCCL_VERSION": "2.32.3", "nccl_version": "2.29.1"`},
		{"escaped_duplicate", `"nccl_version": "2.29.1"`, `"nccl_\u0076ersion": "2.32.3", "nccl_version": "2.29.1"`},
		{"comm_count", `"communicators_count": 1`, `"communicators_count": 2`},
		{"rank_count", `"ranks_count": 1`, `"ranks_count": 0`},
		{"missing_count", `"missing_ranks_count": 1`, `"missing_ranks_count": 0`},
		{"population", `"size": 2`, `"size": 3`},
		{"reused_rank", `"rank": 1`, `"rank": 0`},
		{"rank_oob", `"rank": 1`, `"rank": 2`},
		{"negative_pid", `"pid": 1524344`, `"pid": -1`},
		{"float_rank", `"rank": 0`, `"rank": 0.0`},
		{"float_count", `"AllReduce": 0`, `"AllReduce": 0.0`},
		{"negative_count", `"AllReduce": 0`, `"AllReduce": -1`},
		{"null_count", `"AllReduce": 0`, `"AllReduce": null`},
		{"absent_status", `"init_state": 0,`, ``},
		{"null_status", `"unresponsive": true`, `"unresponsive": null`},
		{"missing_boolean", `"abort_flag": false`, `"other_flag": false`},
		{"negative_duration", `"collection_time_sec": 0.000`, `"collection_time_sec": -1`},
		{"infinite_duration", `"collection_time_sec": 0.000`, `"collection_time_sec": 1e999`},
		{"null_ranks", `"ranks": [`, `"ranks": null, "unexpected": [`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !strings.Contains(b, tc.old) {
				t.Fatal("control substitution missing")
			}
			bad := strings.Replace(b, tc.old, tc.new, 1)
			if _, err := Read(strings.NewReader(bad), Limits{}); err == nil {
				t.Fatal("accepted malformed response")
			}
		})
	}
	for _, bad := range []string{"", "{}", "[]", "null", b[:len(b)-3], b + "{}", "Job summary\n===========\n"} {
		if _, err := Read(strings.NewReader(bad), Limits{}); err == nil {
			t.Fatalf("accepted %q", bad)
		}
	}
}

func TestStatusBoundsAndUnknown(t *testing.T) {
	b := statusFixture(t)
	for _, l := range []Limits{{Bytes: len(b) - 1}, {Line: 1}, {Records: 2}, {Bytes: -1}, {Bytes: 16<<20 + 1}, {Line: 1<<20 + 1}, {Records: 262145}} {
		if _, err := Read(bytes.NewReader(b), l); err == nil {
			t.Fatalf("accepted limits %+v", l)
		}
	}
	if _, err := Read(bytes.NewReader(b), Limits{Bytes: len(b), Records: 3}); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(nil, Limits{}); err == nil {
		t.Fatal("nil reader")
	}
	b = bytes.Replace(b, []byte(`"nccl_version"`), []byte(`"future": {"nested": [1,true,null]}, "nccl_version"`), 1)
	s, err := Read(bytes.NewReader(b), Limits{})
	if err != nil || !bytes.Equal(s.Raw, b) {
		t.Fatalf("unknown field not retained: %v", err)
	}
	deep := `{"future":` + strings.Repeat("[", 34) + "0" + strings.Repeat("]", 34) + "}"
	if _, err := Read(strings.NewReader(deep), Limits{}); err == nil {
		t.Fatal("accepted excessive nesting")
	}
	if _, err := Read(failingReader{}, Limits{}); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("reader error lost: %v", err)
	}
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }

func FuzzRead(f *testing.F) {
	f.Add(statusFixture(f))
	f.Add([]byte("{}"))
	f.Add([]byte("null"))
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 32768 {
			return
		}
		s, err := Read(bytes.NewReader(b), Limits{Bytes: 32768, Line: 32768, Records: 128})
		if err == nil {
			if !bytes.Equal(s.Raw, b) || s.CommunicatorsCount != len(s.Communicators) {
				t.Fatal("accepted inconsistent response")
			}
			var v any
			if json.Unmarshal(s.Raw, &v) != nil {
				t.Fatal("accepted invalid JSON")
			}
		}
	})
}
