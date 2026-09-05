package cuptiactivity

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

func TestWriteRecord(t *testing.T) {
	for _, tt := range []struct {
		name                string
		start, end, origin  uint64
		complete, wantError bool
	}{
		{"complete", 100, 200, 100, true, false},
		{"zero duration", 100, 100, 0, true, false},
		{"incomplete", 100, 200, 0, false, true},
		{"missing start", 0, 200, 0, true, true},
		{"reversed", 200, 100, 0, true, true},
		{"before origin", 100, 200, 101, true, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var b bytes.Buffer
			err := WriteRecord(perfetto.NewWriter(&b), 1, tt.origin, Record{Start: tt.start, End: tt.end, Complete: tt.complete})
			if (err != nil) != tt.wantError {
				t.Fatalf("error = %v", err)
			}
			if tt.wantError && b.Len() != 0 {
				t.Fatal("invalid record wrote trace bytes")
			}
			if !tt.wantError && b.Len() == 0 {
				t.Fatal("complete record wrote no trace bytes")
			}
		})
	}
	if err := WriteRecord(nil, 1, 0, Record{}); err == nil {
		t.Fatal("nil writer accepted")
	}
	err := WriteRecord(perfetto.NewWriter(traceFailure{}), 1, 0, Record{Start: 1, End: 2, Complete: true})
	if !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("write error = %v", err)
	}
}

type traceFailure struct{}

func (traceFailure) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }

func ExampleWriteRecord() {
	var b bytes.Buffer
	w := perfetto.NewWriter(&b)
	err := w.WriteTrackDescriptor(1, "kernel", 0)
	if err == nil {
		err = WriteRecord(w, 1, 100, Record{Name: "gemm", Start: 100, End: 200, Complete: true})
	}
	fmt.Println(err)
	// Output: <nil>
}

func TestTraceTimestamps(t *testing.T) {
	var b bytes.Buffer
	err := WriteRecord(perfetto.NewWriter(&b), 7, 1000, Record{Name: "kernel", Start: 1020, End: 1080, Complete: true, CorrelationID: 42})
	if err != nil {
		t.Fatal(err)
	}
	var timestamps []uint64
	visitFields(t, b.Bytes(), func(field int, value uint64, data []byte) {
		if field != 1 {
			return
		}
		visitFields(t, data, func(field int, value uint64, data []byte) {
			if field == 8 {
				timestamps = append(timestamps, value)
			}
		})
	})
	if len(timestamps) != 2 || timestamps[0] != 20 || timestamps[1] != 80 {
		t.Fatalf("packet timestamps = %v, want [20 80]", timestamps)
	}
	for _, text := range []string{"cupti-relative", "timestamp_origin_ns", "correlation_id", "kernel"} {
		if !bytes.Contains(b.Bytes(), []byte(text)) {
			t.Errorf("missing %q annotation or name", text)
		}
	}
}

func visitFields(t *testing.T, data []byte, visit func(int, uint64, []byte)) {
	t.Helper()
	for len(data) > 0 {
		tag, n := binary.Uvarint(data)
		if n <= 0 {
			t.Fatal("invalid field tag")
		}
		data = data[n:]
		switch tag & 7 {
		case 0:
			value, n := binary.Uvarint(data)
			if n <= 0 {
				t.Fatal("invalid varint")
			}
			data = data[n:]
			visit(int(tag>>3), value, nil)
		case 2:
			size, n := binary.Uvarint(data)
			if n <= 0 || size > uint64(len(data)-n) {
				t.Fatal("invalid field length")
			}
			data = data[n:]
			visit(int(tag>>3), 0, data[:int(size)])
			data = data[int(size):]
		default:
			t.Fatalf("unexpected wire type %d", tag&7)
		}
	}
}
