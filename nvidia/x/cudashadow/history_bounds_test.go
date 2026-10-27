package cudashadow_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudashadow"
)

// oracle returns the standard encoding/json encoding of a whole history.
func oracle(t *testing.T, version int, history []cudashadow.Snapshot) []byte {
	t.Helper()
	data, err := json.Marshal(struct {
		Version   int                   `json:"version"`
		Snapshots []cudashadow.Snapshot `json:"snapshots"`
	}{version, history})
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// A historyWriter counts Write calls and passes them to Writer, failing with
// err or writing one byte short if requested.
type historyWriter struct {
	io.Writer
	calls int
	short bool
	err   error
}

func (w *historyWriter) Write(p []byte) (int, error) {
	w.calls++
	if w.err != nil {
		return 0, w.err
	}
	if w.short {
		p = p[:len(p)-1]
	}
	return w.Writer.Write(p)
}

func TestWriteHistoryEncoding(t *testing.T) {
	d := ptxDecision(t)
	legacy := ptxSnapshot(0, d, map[string]string{"escape": "<&>\u2028\u2029\"\\\n\t\x01é"})
	stale := ptxSnapshot(1, cudashadow.Decision{Type: cudashadow.DecisionSkip, Reason: "stale <input>"}, nil)
	stale.Status = cudashadow.StatusStale
	unavailable := ptxSnapshot(2, d, nil)
	unavailable.ObservedOutcome = &cudashadow.ObservedOutcome{LatencyUnavailable: true, TimestampNano: unavailable.Provenance.TimestampNano, Action: "baseline"}
	unknown := ptxSnapshot(3, d, nil)
	unknown.Input.PTX.FunctionOffsetUnknown = true
	tests := []struct {
		name    string
		history []cudashadow.Snapshot
		version int
		want    string // literal encoding, if checked
	}{
		{"nil", nil, 1, `{"version":1,"snapshots":null}`},
		{"empty", []cudashadow.Snapshot{}, 1, `{"version":1,"snapshots":[]}`},
		{"v1", []cudashadow.Snapshot{legacy, stale}, 1, ""},
		{"v2-outcome", []cudashadow.Snapshot{legacy, stale, unavailable}, 2, ""},
		{"v2-ptx", []cudashadow.Snapshot{unknown}, 2, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var b bytes.Buffer
			w := &historyWriter{Writer: &b}
			if err := cudashadow.WriteHistory(w, tt.history); err != nil || w.calls != 1 {
				t.Fatalf("WriteHistory = %v after %d writes; want nil after 1", err, w.calls)
			}
			if want := oracle(t, tt.version, tt.history); !bytes.Equal(b.Bytes(), want) {
				t.Errorf("WriteHistory wrote\n%s\nwant\n%s", b.Bytes(), want)
			}
			if tt.want != "" && b.String() != tt.want {
				t.Errorf("WriteHistory wrote %s, want %s", b.Bytes(), tt.want)
			}
			got, err := cudashadow.ReadHistory(&b)
			if err != nil || !reflect.DeepEqual(got, tt.history) {
				t.Errorf("ReadHistory = %#v, %v; want %#v", got, err, tt.history)
			}
		})
	}
}

func TestWriteHistoryBounds(t *testing.T) {
	invalid := func(t *testing.T) []cudashadow.Snapshot {
		history := ptxHistory(t, 3, nil)
		history[1].ID = ""
		return history
	}
	tests := []struct {
		name    string
		history func(*testing.T) []cudashadow.Snapshot
		size    int    // oracle encoding size, if checked
		err     string // empty if admitted
	}{
		{"4095", func(t *testing.T) []cudashadow.Snapshot { return ptxHistory(t, 4095, nil) }, 0, ""},
		{"4096", func(t *testing.T) []cudashadow.Snapshot { return ptxHistory(t, 4096, nil) }, 0, ""},
		{"4097", func(t *testing.T) []cudashadow.Snapshot { return ptxHistory(t, 4097, nil) }, 0, "capacity exceeded"},
		{"4097-zero", func(*testing.T) []cudashadow.Snapshot { return make([]cudashadow.Snapshot, 4097) }, 0, "capacity exceeded"},
		{"invalid", invalid, 0, "cudashadow history snapshot 1: "},
		{"16MiB", func(t *testing.T) []cudashadow.Snapshot { return sizedHistory(t, maxHistoryBytes) }, maxHistoryBytes, ""},
		{"16MiB+1", overHistory, maxHistoryBytes + 1, "exceeds 16 MiB"},
		{"shared", func(t *testing.T) []cudashadow.Snapshot { return ptxHistory(t, 4096, bulkDetails()) }, 0, "exceeds 16 MiB"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			history := tt.history(t)
			if tt.size != 0 {
				if n := len(oracle(t, 1, history)); n != tt.size {
					t.Fatalf("history encodes to %d bytes, want %d", n, tt.size)
				}
			}
			h := sha256.New()
			w := &historyWriter{Writer: h}
			err := cudashadow.WriteHistory(w, history)
			if tt.err != "" {
				if err == nil || !strings.Contains(err.Error(), tt.err) || w.calls != 0 {
					t.Errorf("WriteHistory = %v after %d writes; want error containing %q and no writes", err, w.calls, tt.err)
				}
				return
			}
			want := sha256.Sum256(oracle(t, 1, history))
			if err != nil || w.calls != 1 {
				t.Fatalf("WriteHistory = %v after %d writes; want nil after 1", err, w.calls)
			}
			if got := h.Sum(nil); !bytes.Equal(got, want[:]) {
				t.Errorf("WriteHistory wrote sha256 %x, want %x", got, want)
			}
		})
	}
}

func TestWriteHistoryWriter(t *testing.T) {
	history := ptxHistory(t, 2, nil)
	errWrite := errors.New("write failed")
	w := &historyWriter{Writer: io.Discard, err: errWrite}
	if err := cudashadow.WriteHistory(w, history); err != errWrite || w.calls != 1 {
		t.Errorf("WriteHistory to failing writer = %v after %d writes; want %v after 1", err, w.calls, errWrite)
	}
	w = &historyWriter{Writer: io.Discard, short: true}
	if err := cudashadow.WriteHistory(w, history); err != io.ErrShortWrite || w.calls != 1 {
		t.Errorf("WriteHistory to short writer = %v after %d writes; want %v after 1", err, w.calls, io.ErrShortWrite)
	}
	if err := cudashadow.WriteHistory(nil, history); err == nil {
		t.Error("WriteHistory to nil writer succeeded")
	}
}

// TestWriteHistoryAllocation checks that WriteHistory refuses an oversized
// history without encoding all of it. The history encodes to about 64 MiB,
// so encoding it whole before the size check allocates at least twice that:
// the encoding and its copy.
func TestWriteHistoryAllocation(t *testing.T) {
	history := ptxHistory(t, 4*maxHistoryBytes/600_000, bulkDetails())
	w := &historyWriter{Writer: io.Discard}
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	err := cudashadow.WriteHistory(w, history)
	runtime.ReadMemStats(&after)
	if err == nil || !strings.Contains(err.Error(), "exceeds 16 MiB") || w.calls != 0 {
		t.Errorf("WriteHistory = %v after %d writes; want size refusal and no writes", err, w.calls)
	}
	const limit = 6 * maxHistoryBytes
	if n := after.TotalAlloc - before.TotalAlloc; n > limit {
		t.Errorf("refusing %d snapshots allocated %d bytes; want at most %d", len(history), n, limit)
	}
}
