package main

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func TestSampleLossReport(t *testing.T) {
	for _, tt := range []struct {
		name   string
		report sampleReport
		period uint32
		want   string
	}{
		{"counts", sampleReport{PhaseReport: cuptipcsampling.PhaseReport{DroppedSamples: 7, RemainingPCs: 11}}, 16,
			"  lost: 7 samples dropped, 11 PC records reported remaining (summed over reads), drain limited false; raise the period above 2^16\n"},
		{"limited", sampleReport{PhaseReport: cuptipcsampling.PhaseReport{DrainLimited: true}}, 20,
			"  lost: 0 samples dropped, 0 PC records reported remaining (summed over reads), drain limited true; raise the period above 2^20\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := writeSampleLoss(&buf, tt.report, tt.period); err != nil {
				t.Fatal(err)
			}
			if got := buf.String(); got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
	err := errors.New("output failed")
	if got := writeSampleLoss(errorWriter{err}, sampleReport{}, 16); !errors.Is(got, err) {
		t.Fatalf("write error = %v", got)
	}
}

type errorWriter struct{ err error }

func (w errorWriter) Write([]byte) (int, error) { return 0, w.err }

func TestLaunchAndScalarContracts(t *testing.T) {
	// Describe and host scalar construction do not submit a kernel or load a library.
	l := cudalaunch.Kernel(driver.Function(1)).Grid(3, 2, 1).Block(128, 1, 1)
	d, err := l.Describe()
	if err != nil {
		t.Fatal(err)
	}
	if d.Grid != driver.Dims(3, 2, 1) || d.Block != driver.Dims(128, 1, 1) {
		t.Fatalf("launch geometry = %+v", d)
	}
	if cublas.Float32Scalar(0) == cublas.Float32Scalar(1) {
		t.Fatal("host scalar values lost")
	}
}
