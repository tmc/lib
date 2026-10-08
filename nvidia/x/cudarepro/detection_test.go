package cudarepro

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func TestDetectionProbability_Table(t *testing.T) {
	tests := []struct {
		name          string
		reproductions int
		repeats       int
		attempts      int
		want          float64
		wantError     bool
	}{
		{name: "zero attempts", reproductions: 1, repeats: 2, attempts: 0, want: 0.0},
		{name: "zero reproductions", reproductions: 0, repeats: 5, attempts: 10, want: 0.0},
		{name: "deterministic one attempt", reproductions: 4, repeats: 4, attempts: 1, want: 1.0},
		{name: "deterministic multiple attempts", reproductions: 2, repeats: 2, attempts: 5, want: 1.0},
		{name: "half rate one attempt", reproductions: 1, repeats: 2, attempts: 1, want: 0.5},
		{name: "half rate two attempts", reproductions: 1, repeats: 2, attempts: 2, want: 0.75},
		{name: "half rate three attempts", reproductions: 1, repeats: 2, attempts: 3, want: 0.875},
		{name: "invalid zero repeats", reproductions: 0, repeats: 0, attempts: 1, wantError: true},
		{name: "invalid negative repeats", reproductions: 1, repeats: -1, attempts: 1, wantError: true},
		{name: "invalid negative reproductions", reproductions: -1, repeats: 2, attempts: 1, wantError: true},
		{name: "reproductions exceed repeats", reproductions: 3, repeats: 2, attempts: 1, wantError: true},
		{name: "negative attempts", reproductions: 1, repeats: 2, attempts: -1, wantError: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := DetectionProbability(tt.reproductions, tt.repeats, tt.attempts)
			if (err != nil) != tt.wantError {
				t.Fatalf("DetectionProbability(%d, %d, %d) err=%v, wantError=%v",
					tt.reproductions, tt.repeats, tt.attempts, err, tt.wantError)
			}
			if !tt.wantError && math.Abs(got-tt.want) > 1e-9 {
				t.Fatalf("DetectionProbability(%d, %d, %d) = %v, want %v",
					tt.reproductions, tt.repeats, tt.attempts, got, tt.want)
			}
		})
	}
}

func TestDetectionProbability_Monotonicity(t *testing.T) {
	prev := 0.0
	for attempts := 1; attempts <= 20; attempts++ {
		p, err := DetectionProbability(1, 4, attempts)
		if err != nil {
			t.Fatalf("attempts=%d: %v", attempts, err)
		}
		if p < prev {
			t.Fatalf("attempts=%d: p=%v < prev=%v", attempts, p, prev)
		}
		prev = p
	}
}

func TestDetectionAttempts_Table(t *testing.T) {
	identity := Identity{Index: 0, Policy: "exact-uint64"}
	makeRuns := func(reproductions, total int) []cudaschedule.RunRecord {
		runs := make([]cudaschedule.RunRecord, total)
		for i := 0; i < total; i++ {
			r := mismatchRecord(0)
			if i >= reproductions {
				r.Stdout = "PASS"
				r.Mismatch = nil
			}
			runs[i] = r
		}
		return runs
	}

	tests := []struct {
		name              string
		reproductions     int
		repeats           int
		targetProbability float64
		maxAttempts       int
		wantAttempts      int
		wantError         bool
	}{
		{
			name:              "deterministic finding needs 1 attempt",
			reproductions:     3,
			repeats:           3,
			targetProbability: 0.99,
			maxAttempts:       10,
			wantAttempts:      1,
		},
		{
			name:              "half rate 99 percent detection",
			reproductions:     1,
			repeats:           2,
			targetProbability: 0.99,
			maxAttempts:       20,
			// ceil(ln(0.01) / ln(0.5)) = ceil(-4.605 / -0.693) = ceil(6.64) = 7.
			wantAttempts: 7,
		},
		{
			name:              "half rate 99.9 percent detection",
			reproductions:     1,
			repeats:           2,
			targetProbability: 0.999,
			maxAttempts:       20,
			// ceil(ln(0.001) / ln(0.5)) = ceil(-6.907 / -0.693) = ceil(9.96) = 10.
			wantAttempts: 10,
		},
		{
			name:              "required attempts exceed limit",
			reproductions:     1,
			repeats:           2,
			targetProbability: 0.99,
			maxAttempts:       5,
			wantError:         true,
		},
		{
			name:              "target probability zero invalid",
			reproductions:     1,
			repeats:           2,
			targetProbability: 0.0,
			maxAttempts:       10,
			wantError:         true,
		},
		{
			name:              "target probability one invalid",
			reproductions:     1,
			repeats:           2,
			targetProbability: 1.0,
			maxAttempts:       10,
			wantError:         true,
		},
		{
			name:              "target probability NaN invalid",
			reproductions:     1,
			repeats:           2,
			targetProbability: math.NaN(),
			maxAttempts:       10,
			wantError:         true,
		},
		{
			name:              "max attempts non-positive invalid",
			reproductions:     1,
			repeats:           2,
			targetProbability: 0.99,
			maxAttempts:       0,
			wantError:         true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := Measurement{
				Reproductions: tt.reproductions,
				Runs:          makeRuns(tt.reproductions, tt.repeats),
			}
			got, err := DetectionAttempts(identity, m, tt.repeats, tt.targetProbability, tt.maxAttempts)
			if (err != nil) != tt.wantError {
				t.Fatalf("DetectionAttempts err=%v, wantError=%v", err, tt.wantError)
			}
			if !tt.wantError && got != tt.wantAttempts {
				t.Fatalf("DetectionAttempts = %d, want %d", got, tt.wantAttempts)
			}
		})
	}
}

func TestMeasurement_Methods(t *testing.T) {
	identity := Identity{Index: 0, Policy: "exact-uint64"}
	runs := []cudaschedule.RunRecord{mismatchRecord(0), mismatchRecord(0)}
	m := Measurement{Reproductions: 2, Runs: runs}

	prob, err := m.DetectionProbability(1)
	if err != nil || prob != 1.0 {
		t.Fatalf("m.DetectionProbability(1) = %v, %v; want 1.0, nil", prob, err)
	}

	attempts, err := m.DetectionAttempts(identity, 0.99, 10)
	if err != nil || attempts != 1 {
		t.Fatalf("m.DetectionAttempts = %d, %v; want 1, nil", attempts, err)
	}
}
