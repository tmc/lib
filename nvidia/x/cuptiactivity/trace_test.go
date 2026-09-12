package cuptiactivity

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
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

func TestTraceOptionalActivityFields(t *testing.T) {
	for _, tt := range []struct {
		name            string
		record          Record
		present, absent []string
	}{
		{"kernel", Record{Kind: cupti.ActivityKindConcurrentKernel, Queued: 5, Submitted: 8}, []string{"queued_ns", "submitted_ns"}, []string{"copy_kind"}},
		{"unknown queue", Record{Kind: cupti.ActivityKindConcurrentKernel}, nil, []string{"queued_ns", "submitted_ns", "copy_kind"}},
		{"launch", Record{Kind: cupti.ActivityKindConcurrentKernel}, []string{"grid_x", "block_z", "registers_per_thread", "dynamic_shared_memory_bytes", "local_memory_total_bytes", "graph_node_id"}, []string{"occupancy", "shared_traffic"}},
		{"copy", Record{Kind: cupti.ActivityKindMemcpy, CopyKind: cupti.ActivityMemcpyKindDtoh}, []string{"copy_kind"}, []string{"queued_ns", "submitted_ns"}},
		{"sync", Record{Kind: cupti.ActivityKindSynchronization, CudaEventSyncID: 1<<63 + 9}, []string{"synchronization_type", "cuda_event_id", "cuda_event_sync_id"}, []string{"queued_ns", "submitted_ns", "copy_kind"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := tt.record
			r.Start, r.End, r.Complete = 10, 20, true
			var b bytes.Buffer
			if err := WriteRecord(perfetto.NewWriter(&b), 1, 0, r); err != nil {
				t.Fatal(err)
			}
			for _, name := range tt.present {
				if !bytes.Contains(b.Bytes(), []byte(name)) {
					t.Errorf("missing %s", name)
				}
			}
			for _, name := range tt.absent {
				if bytes.Contains(b.Bytes(), []byte(name)) {
					t.Errorf("unexpected %s", name)
				}
			}
		})
	}
}

func TestTraceSynchronizationIdentity(t *testing.T) {
	var b bytes.Buffer
	r := Record{Kind: cupti.ActivityKindSynchronization, Start: 10, End: 20, Complete: true, CudaEventSyncID: 1<<63 + 9, ReturnValue: 600}
	if err := WriteRecord(perfetto.NewWriter(&b), 1, 0, r); err != nil {
		t.Fatal(err)
	}
	values := traceUintAnnotations(t, b.Bytes())
	if values["cuda_event_sync_id"] != r.CudaEventSyncID || values["return_value"] != 600 {
		t.Fatalf("annotations=%v", values)
	}
}

func TestTraceMemcpyIdentity(t *testing.T) {
	var b bytes.Buffer
	r := Record{Kind: cupti.ActivityKindMemcpy, Start: 10, End: 20, Complete: true, CorrelationID: 7, RuntimeCorrelationID: 19, CopyCount: 1<<63 + 9, SourceMemoryKind: 255, DestinationMemoryKind: 254}
	if err := WriteRecord(perfetto.NewWriter(&b), 1, 0, r); err != nil {
		t.Fatal(err)
	}
	values := traceUintAnnotations(t, b.Bytes())
	for name, want := range map[string]uint64{"correlation_id": 7, "runtime_correlation_id": 19, "copy_count": 1<<63 + 9, "source_memory_kind": 255, "destination_memory_kind": 254} {
		if got, ok := values[name]; !ok || got != want {
			t.Errorf("%s=%d present=%v, want %d", name, got, ok, want)
		}
	}
}

func traceUintAnnotations(t *testing.T, data []byte) map[string]uint64 {
	t.Helper()
	values := make(map[string]uint64)
	visitFields(t, data, func(field int, _ uint64, packet []byte) {
		if field != 1 {
			return
		}
		visitFields(t, packet, func(field int, _ uint64, event []byte) {
			if field != perfetto.FieldPacketTrackEvent {
				return
			}
			visitFields(t, event, func(field int, _ uint64, annotation []byte) {
				if field != perfetto.FieldTrackEventDebugAnnotations {
					return
				}
				var name string
				var value uint64
				var unsigned bool
				visitFields(t, annotation, func(field int, v uint64, data []byte) {
					switch field {
					case perfetto.FieldDebugAnnotationName:
						name = string(data)
					case perfetto.FieldDebugAnnotationUintValue:
						value, unsigned = v, true
					}
				})
				if unsigned {
					values[name] = value
				}
			})
		})
	})
	return values
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
