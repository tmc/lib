package cudalmvet

import (
	"os"
	"reflect"
	"strings"
	"testing"
)

type raceProbeCounts struct {
	raceOutputs             int
	raceHashes              int
	controlOutputs          int
	instrumentedControlOut  int
	instrumentedControlHash int
}

func raceProbeFailures(counts raceProbeCounts) []string {
	var failures []string
	if counts.controlOutputs != 1 || counts.instrumentedControlOut != 1 || counts.instrumentedControlHash != 1 {
		failures = append(failures, "race-free control varied (broken control)")
	}
	if counts.raceOutputs <= 1 {
		failures = append(failures, "race did not vary uninstrumented (inconclusive)")
	}
	if counts.raceOutputs > 1 && counts.raceHashes == 1 {
		failures = append(failures, "race varied uninstrumented but gave one hash instrumented (suppressed)")
	}
	return failures
}

func TestRaceProbeFailureReportsBrokenControlFirst(t *testing.T) {
	tests := []struct {
		name   string
		counts raceProbeCounts
		want   []string
	}{
		{
			name: "healthy race probe",
			counts: raceProbeCounts{
				raceOutputs: 2, raceHashes: 2, controlOutputs: 1,
				instrumentedControlOut: 1, instrumentedControlHash: 1,
			},
		},
		{
			name: "broken control and inconclusive race",
			counts: raceProbeCounts{
				raceOutputs: 1, raceHashes: 1, controlOutputs: 2,
				instrumentedControlOut: 1, instrumentedControlHash: 1,
			},
			want: []string{
				"race-free control varied (broken control)",
				"race did not vary uninstrumented (inconclusive)",
			},
		},
		{"broken control alone", raceProbeCounts{2, 2, 2, 1, 1}, []string{"race-free control varied (broken control)"}},
		{"inconclusive alone", raceProbeCounts{1, 1, 1, 1, 1}, []string{"race did not vary uninstrumented (inconclusive)"}},
		{"suppressed alone", raceProbeCounts{2, 1, 1, 1, 1}, []string{"race varied uninstrumented but gave one hash instrumented (suppressed)"}},
		{"broken control with suppressed", raceProbeCounts{2, 1, 2, 1, 1}, []string{"race-free control varied (broken control)", "race varied uninstrumented but gave one hash instrumented (suppressed)"}},
		{"instrumented control hash alone", raceProbeCounts{2, 2, 1, 1, 2}, []string{"race-free control varied (broken control)"}},
		{"instrumented control output alone", raceProbeCounts{2, 2, 1, 2, 1}, []string{"race-free control varied (broken control)"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := raceProbeFailures(tt.counts); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("raceProbeFailures() = %v, want %v", got, tt.want)
			}
		})
	}
}

// The probe itself is native-only. Pin its classification/log order without
// running a device: an outcome cannot terminate the probe before this log.
func TestRaceProbeLogsFixedSeedBeforeOutcomeFailure(t *testing.T) {
	source, err := os.ReadFile("native_test.go")
	if err != nil {
		t.Fatal(err)
	}
	log := strings.Index(string(source), `t.Logf("race instrumented repeated fixed seed 0:`)
	classify := strings.Index(string(source), "raceProbeFailures(counts)")
	if log < 0 || classify <= log {
		t.Fatal("native outcome classification precedes fixed-seed log")
	}
}
