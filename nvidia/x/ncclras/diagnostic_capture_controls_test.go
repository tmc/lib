package ncclras_test

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

func diagnosticRecords(t *testing.T) []byte {
	t.Helper()
	var b []byte
	for _, name := range []string{"stats", "collective"} {
		raw, err := os.ReadFile("../ncclinspector/testdata/" + name + ".jsonl")
		if err != nil {
			t.Fatal(err)
		}
		records, err := ncclinspector.Read(bytes.NewReader(raw), ncclinspector.Limits{})
		if err != nil {
			t.Fatal(err)
		}
		for rank := 0; rank < 2; rank++ {
			r := records[0]
			r.Header.Rank, r.Header.Ranks, r.Header.ID = rank, 2, "control"
			r.Metadata.PID, r.Metadata.Hostname = 123, "host"
			if r.Collective != nil {
				r.Collective.Bytes = 65536
			}
			if r.Stats != nil {
				r.Stats = &ncclinspector.DumpStats{Collectives: 4}
			}
			line, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			if r.Collective == nil {
				b = append(b, append(line, '\n')...)
			} else {
				for sequence := uint64(0); sequence < 4; sequence++ {
					r.Collective.Sequence = sequence
					line, err = json.Marshal(r)
					if err != nil {
						t.Fatal(err)
					}
					b = append(b, append(line, '\n')...)
				}
			}
		}
	}
	return b
}

func TestDiagnosticCapture(t *testing.T) {
	good := diagnosticRecords(t)
	for _, tc := range []struct {
		name string
		data []byte
		want string
	}{
		{"complete", good, ""},
		{"missing", nil, "observation budget"},
		{"partial", good[:len(good)-1], "observation budget"},
		{"malformed", []byte("{bad}\n"), "inspector record"},
		{"pid", bytes.ReplaceAll(good, []byte(`"pid":123`), []byte(`"pid":124`)), "membership differs"},
		{"host", bytes.ReplaceAll(good, []byte(`"host"`), []byte(`"other"`)), "membership differs"},
		{"hash", bytes.ReplaceAll(good, []byte(`"control"`), []byte(`"other"`)), "membership differs"},
		{"bytes", bytes.ReplaceAll(good, []byte(`65536`), []byte(`65535`)), "operation differs"},
		{"drops", bytes.ReplaceAll(good, []byte(`"coll_dropped_total":0`), []byte(`"coll_dropped_total":1`)), "dropped Inspector"},
		{"membership", bytes.ReplaceAll(good, []byte(`"n_ranks":2`), []byte(`"n_ranks":3`)), "membership differs"},
		{"nodes", bytes.ReplaceAll(good, []byte(`"nnodes":1`), []byte(`"nnodes":2`)), "membership differs"},
		{"operation", bytes.ReplaceAll(good, []byte(`"AllReduce"`), []byte(`"Reduce"`)), "operation differs"},
		{"duplicate", bytes.ReplaceAll(good, []byte(`"coll_sn":1`), []byte(`"coll_sn":0`)), "duplicate Inspector"},
		{"underannounced", bytes.ReplaceAll(good, []byte(`"coll_records":4`), []byte(`"coll_records":3`)), "collective count differs"},
		{"overannounced", bytes.ReplaceAll(good, []byte(`"coll_records":4`), []byte(`"coll_records":5`)), "observation budget"},
		{"p2p_stats", bytes.ReplaceAll(good, []byte(`"p2p_records":0`), []byte(`"p2p_records":1`)), "point-to-point records"},
		{"file_budget", make([]byte, diagnosticCaptureBytes+1), "byte budget"},
		{"aggregate_budget", make([]byte, 1<<20), "byte budget"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := waitDiagnosticCapture(context.Background(), func() ([]byte, error) { return tc.data, nil }, "host", 123, "control", 0)
			if tc.want == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
	t.Run("live_partial_then_complete", func(t *testing.T) {
		calls := 0
		b, err := waitDiagnosticCapture(context.Background(), func() ([]byte, error) {
			calls++
			if calls == 1 {
				return good[:len(good)-1], nil
			}
			return good, nil
		}, "host", 123, "control", 0)
		if err != nil || calls != 2 || !bytes.Equal(b, good) {
			t.Fatalf("calls=%d err=%v", calls, err)
		}
	})
	t.Run("deadline", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := waitDiagnosticCapture(ctx, func() ([]byte, error) { t.Fatal("read after cancellation"); return nil, nil }, "host", 123, "control", 0)
		if err == nil {
			t.Fatal("accepted expired deadline")
		}
	})
}

// The pinned Inspector's 17-byte hash buffer loses the last two hex digits of
// a full-width uint64 hash. The missing bits cannot establish exact identity.
func TestDiagnosticCaptureTruncatedHash(t *testing.T) {
	good := diagnosticRecords(t)
	for _, tc := range []struct {
		name string
		id   string
		want string
	}{
		{"full", "0x91541b8a24505d53", ""},
		{"truncated", "0x91541b8a24505d", "membership differs"},
		{"same_prefix_other_hash", "0x91541b8a24505d54", "membership differs"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := bytes.ReplaceAll(good, []byte(`"control"`), []byte(`"`+tc.id+`"`))
			calls := 0
			_, err := waitDiagnosticCapture(context.Background(), func() ([]byte, error) {
				calls++
				return b, nil
			}, "host", 123, "0x91541b8a24505d53", 0)
			if calls != 1 {
				t.Fatalf("read calls = %d, want 1", calls)
			}
			if tc.want == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}
