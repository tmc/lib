package perfetto_test

import (
	"errors"
	"io"
	"testing"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestShortWrites(t *testing.T) {
	for _, tt := range []struct {
		name  string
		write func(*perfetto.Writer) error
	}{
		{"descriptor", func(w *perfetto.Writer) error { return w.WriteTrackDescriptor(1, "track", 0) }},
		{"clock", func(w *perfetto.Writer) error { return w.WriteClockSnapshot(1, 2) }},
		{"counter", func(w *perfetto.Writer) error { return w.WriteCounterValue(1, 2, 3) }},
		{"double counter", func(w *perfetto.Writer) error { return w.WriteDoubleCounterValue(1, 2, 3) }},
		{"begin", func(w *perfetto.Writer) error { return w.WriteSliceBegin(1, 2, "slice", "category") }},
		{"end", func(w *perfetto.Writer) error { return w.WriteSliceEnd(1, 2) }},
		{"instant", func(w *perfetto.Writer) error { return w.WriteInstant(1, 2, "event", "category") }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.write(perfetto.NewWriter(shortWriter{})); !errors.Is(err, io.ErrShortWrite) {
				t.Fatalf("got %v, want short write", err)
			}
		})
	}
}
