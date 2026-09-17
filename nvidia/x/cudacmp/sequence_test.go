package cudacmp_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudacmp"
)

func TestAnalyzeLaunchSequences(t *testing.T) {
	launches := []cudacensus.Launch{
		// Stream 1: Rapid sequence (candidate batching) with repeated pattern [gate, up]
		{Name: "gate", StreamID: 1, StartNs: 1000, EndNs: 2000},
		{Name: "up", StreamID: 1, StartNs: 2100, EndNs: 3000}, // gap = 100
		{Name: "gate", StreamID: 1, StartNs: 3100, EndNs: 4000}, // gap = 100
		{Name: "up", StreamID: 1, StartNs: 4100, EndNs: 5000}, // gap = 100
		{Name: "gate", StreamID: 1, StartNs: 5100, EndNs: 6000}, // gap = 100
		{Name: "up", StreamID: 1, StartNs: 6100, EndNs: 7000}, // gap = 100

		// Large gap on stream 1 before next launch
		{Name: "down", StreamID: 1, StartNs: 15000, EndNs: 18000}, // gap = 8000 ns (> 1000ns)

		// Stream 2: Two launches with small gap
		{Name: "norm", StreamID: 2, StartNs: 500, EndNs: 1200},
		{Name: "rope", StreamID: 2, StartNs: 3000, EndNs: 4000}, // gap = 1800 ns
	}

	analysis, err := cudacmp.AnalyzeLaunchSequences(launches, cudacmp.SequenceConfig{
		MinGapNs:                     1000,
		MinSequenceLen:               2,
		MinRepetitions:               2,
		CandidateBatchingMinLaunches: 3,
		CandidateBatchingMaxGapNs:    500,
	})
	if err != nil {
		t.Fatalf("AnalyzeLaunchSequences failed: %v", err)
	}

	// Verify streams
	if len(analysis.StreamIDs) != 2 || analysis.StreamIDs[0] != 1 || analysis.StreamIDs[1] != 2 {
		t.Errorf("StreamIDs = %v, want [1 2]", analysis.StreamIDs)
	}

	// Verify gaps
	// Expected gap on stream 1: [7000, 15000) = 8000ns
	// Expected gap on stream 2: [1200, 3000) = 1800ns
	if len(analysis.Gaps) != 2 {
		t.Fatalf("gaps count = %d, want 2: %+v", len(analysis.Gaps), analysis.Gaps)
	}
	if analysis.Gaps[0].StreamID != 1 || analysis.Gaps[0].GapNs != 8000 {
		t.Errorf("gap 0 = %+v, want stream 1 gap 8000ns", analysis.Gaps[0])
	}
	if len(analysis.Gaps[0].MissingEvidence) == 0 {
		t.Error("expected gap to report missing evidence")
	}

	// Verify repeated sequences on stream 1: [gate, up] repeated 3 times
	if len(analysis.RepeatedSequences) == 0 {
		t.Fatal("expected repeated sequence to be detected")
	}
	seq := analysis.RepeatedSequences[0]
	if seq.StreamID != 1 || seq.Repetitions != 3 || len(seq.Pattern) != 2 {
		t.Errorf("unexpected repeated sequence: %+v", seq)
	}
	if seq.Pattern[0] != "gate" || seq.Pattern[1] != "up" {
		t.Errorf("pattern = %v, want [gate up]", seq.Pattern)
	}
	if len(seq.MissingEvidence) == 0 {
		t.Error("expected repeated sequence to report missing evidence")
	}

	// Verify candidate batching region on stream 1: 6 launches with gap <= 500ns
	if len(analysis.BatchingCandidates) != 1 {
		t.Fatalf("batching candidates count = %d, want 1", len(analysis.BatchingCandidates))
	}
	cand := analysis.BatchingCandidates[0]
	if cand.StreamID != 1 || cand.LaunchCount != 6 {
		t.Errorf("batching candidate = %+v, want stream 1 launchCount 6", cand)
	}
	if len(cand.MissingEvidence) == 0 {
		t.Error("expected batching candidate to report missing evidence")
	}
}

func TestAnalyzeLaunchSequencesValidation(t *testing.T) {
	t.Run("end before start rejected", func(t *testing.T) {
		launches := []cudacensus.Launch{
			{Name: "bad", StreamID: 1, StartNs: 5000, EndNs: 4000},
		}
		_, err := cudacmp.AnalyzeLaunchSequences(launches, cudacmp.SequenceConfig{})
		if err == nil {
			t.Fatal("expected error for EndNs < StartNs, got nil")
		}
	})

	t.Run("negative timestamp rejected", func(t *testing.T) {
		launches := []cudacensus.Launch{
			{Name: "bad", StreamID: 1, StartNs: -100, EndNs: 200},
		}
		_, err := cudacmp.AnalyzeLaunchSequences(launches, cudacmp.SequenceConfig{})
		if err == nil {
			t.Fatal("expected error for negative timestamp, got nil")
		}
	})
}
