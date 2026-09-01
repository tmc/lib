package metalprof_test

import (
	"bytes"
	"compress/gzip"
	"context"
	"io"
	"runtime/pprof"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/apple/x/metal/metalprof"
)

func TestBuilderAndProtoEncode(t *testing.T) {
	b := metalprof.NewBuilder()
	fn1 := b.AddFunction("RMSNormKernel", "rmsnorm.metal", 10)
	loc1 := b.AddLocation(fn1, 24)

	fn2 := b.AddFunction("FastGEMVKernel", "gemv.metal", 50)
	loc2 := b.AddLocation(fn2, 65)

	b.AddSample([]uint64{loc1}, metalprof.SampleValues{Samples: 10, GPUTimeNs: 15000}, map[string]string{"layer": "0", "op": "norm"})
	b.AddSample([]uint64{loc2, loc1}, metalprof.SampleValues{Samples: 5, GPUTimeNs: 45000}, map[string]string{"layer": "0", "op": "gemv"})

	var buf bytes.Buffer
	if err := b.Encode(&buf); err != nil {
		t.Fatalf("Builder.Encode failed: %v", err)
	}

	if buf.Len() == 0 {
		t.Fatal("expected non-empty profile output")
	}

	// Verify gzip decompression
	gz, err := gzip.NewReader(&buf)
	if err != nil {
		t.Fatalf("gzip.NewReader failed: %v", err)
	}
	defer gz.Close()

	decompressed, err := io.ReadAll(gz)
	if err != nil {
		t.Fatalf("read decompressed profile failed: %v", err)
	}
	if len(decompressed) == 0 {
		t.Fatal("decompressed profile is empty")
	}
}

func TestFlightRecorderRingBuffer(t *testing.T) {
	fr := metalprof.NewFlightRecorder(4)

	// Record 6 events to test ring buffer wrap-around
	for i := 0; i < 6; i++ {
		fr.Record(
			"Kernel_"+string(rune('A'+i)),
			time.Duration(i+1)*time.Microsecond,
			[3]int{128, 1, 1},
			[3]int{32, 1, 1},
			map[string]string{"step": string(rune('0' + i))},
		)
	}

	events := fr.Events()
	if len(events) != 4 {
		t.Fatalf("expected 4 events, got %d", len(events))
	}

	// Chronological order should be C, D, E, F
	if events[0].Name != "Kernel_C" || events[3].Name != "Kernel_F" {
		t.Errorf("unexpected event ordering: first=%s, last=%s", events[0].Name, events[3].Name)
	}

	var buf bytes.Buffer
	if err := fr.SnapshotPprof(&buf); err != nil {
		t.Fatalf("SnapshotPprof failed: %v", err)
	}
	if buf.Len() == 0 {
		t.Fatal("expected non-empty snapshot profile")
	}

	summary := fr.SnapshotSummary()
	if !strings.Contains(summary, "Metal FlightRecorder") || !strings.Contains(summary, "Kernel_C") {
		t.Errorf("unexpected summary text: %s", summary)
	}
}

func TestLabelsPropagation(t *testing.T) {
	labels := pprof.Labels("layer", "12", "phase", "decode")
	metalprof.Do(context.Background(), labels, func(ctx context.Context) {
		extracted := metalprof.LabelsFromContext(ctx)
		if extracted["layer"] != "12" || extracted["phase"] != "decode" {
			t.Errorf("extracted labels mismatch: %v", extracted)
		}
		formatted := metalprof.FormatLabels(extracted)
		if formatted != "layer=12,phase=decode" {
			t.Errorf("formatted labels mismatch: %q", formatted)
		}
	})
}
