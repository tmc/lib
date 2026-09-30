package perfetto_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

func ExampleWriter_WriteFlowInstant() {
	var output bytes.Buffer
	w := perfetto.NewWriter(&output)
	w.WriteTrackDescriptor(1, "producer", 0)
	w.WriteTrackDescriptor(2, "consumer", 0)
	w.WriteFlowInstant(1, 100, "event recorded", "event_wait", nil, []uint64{7}, nil)
	w.WriteFlowInstant(2, 200, "event waited", "event_wait", nil, nil, []uint64{7})
	fmt.Println(output.Len() > 0)
	// Output: true
}

func TestFlowIDsUseFixed64(t *testing.T) {
	var output bytes.Buffer
	id := uint64(1<<54 + 3)
	if err := perfetto.NewWriter(&output).WriteFlowInstant(1, 2, "edge", "event_wait", nil, []uint64{id}, []uint64{id + 1}); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct{ field, id uint64 }{{47, id}, {48, id + 1}} {
		want := binary.AppendUvarint(nil, tt.field<<3|1)
		want = binary.LittleEndian.AppendUint64(want, tt.id)
		if !bytes.Contains(output.Bytes(), want) {
			t.Fatalf("missing fixed64 field %d", tt.field)
		}
	}
}

func TestFlowCannotOriginateAndTerminateTogether(t *testing.T) {
	var output bytes.Buffer
	err := perfetto.NewWriter(&output).WriteFlowInstant(1, 2, "invalid", "", nil, []uint64{7}, []uint64{7})
	if err == nil || output.Len() != 0 {
		t.Fatalf("error=%v bytes=%d", err, output.Len())
	}
}
