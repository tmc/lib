package cudaflightrecorder

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
)

func checkGoTraceLoss(t *testing.T, err error, retained, dropped uint64, overflow bool) {
	t.Helper()
	if dropped == 0 && !overflow {
		if err != nil {
			t.Fatalf("unexpected loss: %v", err)
		}
		return
	}
	var loss *goTraceLossError
	if !errors.Is(err, errGoTraceTruncated) || !errors.As(err, &loss) {
		t.Fatalf("loss identity missing: %v", err)
	}
	if loss.retained != retained || loss.dropped != dropped || loss.overflow != overflow {
		t.Fatalf("loss = %+v; want retained=%d dropped=%d overflow=%t", loss, retained, dropped, overflow)
	}
	want := fmt.Sprintf("cudaflightrecorder: go trace truncated: retained %d bytes, dropped %d bytes", retained, dropped)
	if overflow {
		want = fmt.Sprintf("cudaflightrecorder: go trace truncated: retained %d bytes, dropped at least %d bytes (count overflow)", retained, dropped)
	}
	if err.Error() != want {
		t.Errorf("loss detail = %q; want %q", err, want)
	}
}

func TestGoTraceCapLossWrites(t *testing.T) {
	for _, tt := range []struct {
		name    string
		cap     uint64
		writes  []string
		want    string
		dropped uint64
	}{
		{"unlimited", 0, []string{"abc", "def"}, "abcdef", 0},
		{"below", 4, []string{"ab", "c"}, "abc", 0},
		{"exact", 4, []string{"ab", "cd", ""}, "abcd", 0},
		{"partial", 4, []string{"abc", "de"}, "abcd", 1},
		{"full", 4, []string{"abcd", "ef"}, "abcd", 2},
		{"repeated", 4, []string{"abc", "def", "gh", ""}, "abcd", 4},
		{"large cap", ^uint64(0), []string{"abc", "def"}, "abcdef", 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := &syncBuffer{maxBytes: tt.cap}
			for _, p := range tt.writes {
				if n, err := b.Write([]byte(p)); n != len(p) || err != nil {
					t.Fatalf("Write(%q) = %d, %v; want %d, nil", p, n, err, len(p))
				}
			}
			for range 2 {
				data, err := b.snapshot()
				if string(data) != tt.want {
					t.Fatalf("retained = %q; want %q", data, tt.want)
				}
				checkGoTraceLoss(t, err, uint64(len(tt.want)), tt.dropped, false)
				if len(data) > 0 {
					data[0] = 'X'
				}
			}
		})
	}
}

func TestGoTraceCapLossOverflow(t *testing.T) {
	b := &syncBuffer{maxBytes: 1, dropped: ^uint64(0) - 1}
	b.Write([]byte("abcd"))
	for range 2 {
		data, err := b.snapshot()
		if string(data) != "a" {
			t.Fatalf("retained = %q", data)
		}
		checkGoTraceLoss(t, err, 1, ^uint64(0), true)
		b.Write([]byte("ef"))
	}
}

func TestGoTraceCapLossSnapshots(t *testing.T) {
	for _, truncated := range []bool{false, true} {
		t.Run(fmt.Sprint(truncated), func(t *testing.T) {
			b := &syncBuffer{maxBytes: 4}
			b.Write([]byte("abcd"))
			s := &standardTraceRecorder{buf: b, running: true}
			data, err := s.Snapshot(context.Background())
			if string(data) != "abcd" || err != nil {
				t.Fatalf("before loss = %q, %v", data, err)
			}
			var dropped uint64
			if truncated {
				b.Write([]byte("ef"))
				dropped = 2
			}
			for range 2 {
				data, err = s.Snapshot(context.Background())
				if string(data) != "abcd" {
					t.Fatalf("live retained = %q", data)
				}
				checkGoTraceLoss(t, err, 4, dropped, false)
				data[0] = 'X'
			}
			stops := 0
			stop := func() {
				stops++
				if truncated {
					b.Write([]byte("g")) // Final runtime write before Stop returns.
				}
			}
			err = s.stop(stop)
			if truncated {
				dropped++
			}
			checkGoTraceLoss(t, err, 4, dropped, false)
			for range 2 {
				checkGoTraceLoss(t, s.stop(stop), 4, dropped, false)
				data, err = s.Snapshot(context.Background())
				if string(data) != "abcd" {
					t.Fatalf("stopped retained = %q", data)
				}
				checkGoTraceLoss(t, err, 4, dropped, false)
				data[0] = 'X'
			}
			if stops != 1 || s.running {
				t.Fatalf("stops=%d running=%t", stops, s.running)
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if data, err := s.Snapshot(ctx); data != nil || !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled snapshot = %q, %v", data, err)
			}
		})
	}
}

func TestGoTraceCapLossConcurrent(t *testing.T) {
	b := &syncBuffer{maxBytes: 16}
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			for range 100 {
				b.Write([]byte("abcd"))
				b.snapshot()
			}
		})
	}
	wg.Wait()
	data, err := b.snapshot()
	if len(data) != 16 || strings.Trim(string(data), "abcd") != "" {
		t.Fatalf("retained = %q", data)
	}
	checkGoTraceLoss(t, err, 16, 1584, false)
}
