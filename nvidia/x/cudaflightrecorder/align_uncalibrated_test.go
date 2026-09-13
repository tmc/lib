package cudaflightrecorder_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func TestAlignSpansAllUncalibrated(t *testing.T) {
	goSpan, gpuSpan, overlap, reliable := cudaflightrecorder.AlignSpans(
		200, 100, 100, true,
		[]cudaflightrecorder.Record{{TimestampNs: 0, DurationNs: 50, Labels: map[string]string{"activity.uncalibrated": "true"}}},
		0,
	)
	if goSpan != 100 || gpuSpan != 0 || overlap != 0 || reliable {
		t.Fatalf("AlignSpans = (%d, %d, %d, %t), want (100, 0, 0, false)", goSpan, gpuSpan, overlap, reliable)
	}
}
