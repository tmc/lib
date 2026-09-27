package cudatma

import (
	"testing"
)

func TestAsyncCopyPipeline_InvalidParams(t *testing.T) {
	if _, err := NewAsyncCopyPipeline(0, 256); err == nil {
		t.Errorf("expected error on 0 stages")
	}
	if _, err := NewAsyncCopyPipeline(2, 0); err == nil {
		t.Errorf("expected error on 0 stage bytes")
	}
}

func TestAsyncCopyPipeline_FixedZeroDetectsHazard(t *testing.T) {
	sc, err := GenerateAsyncCopyScenario(false)
	if err != nil {
		t.Fatalf("GenerateAsyncCopyScenario(false): %v", err)
	}
	if !sc.HasHazards {
		t.Errorf("FIXED=0 scenario must detect hazard, got 0 hazards")
	}
	if len(sc.Hazards) == 0 {
		t.Fatalf("expected non-empty hazards slice")
	}
	if sc.Hazards[0].Kind != "RAW" {
		t.Errorf("expected RAW hazard, got %s", sc.Hazards[0].Kind)
	}
}

func TestAsyncCopyPipeline_FixedOneHasZeroHazards(t *testing.T) {
	sc, err := GenerateAsyncCopyScenario(true)
	if err != nil {
		t.Fatalf("GenerateAsyncCopyScenario(true): %v", err)
	}
	if sc.HasHazards {
		t.Errorf("FIXED=1 scenario must have zero hazards, got %d", len(sc.Hazards))
	}
}

func TestTMAScenario_FixedZeroDetectsHazard(t *testing.T) {
	sc, err := GenerateTMAScenario(false)
	if err != nil {
		t.Fatalf("GenerateTMAScenario(false): %v", err)
	}
	if !sc.HasHazards {
		t.Errorf("FIXED=0 TMA scenario must detect hazard, got 0 hazards")
	}
}

func TestTMAScenario_FixedOneHasZeroHazards(t *testing.T) {
	sc, err := GenerateTMAScenario(true)
	if err != nil {
		t.Fatalf("GenerateTMAScenario(true): %v", err)
	}
	if sc.HasHazards {
		t.Errorf("FIXED=1 TMA scenario must have zero hazards, got %d", len(sc.Hazards))
	}
}

func TestAsyncCopyPipeline_WAWHazard(t *testing.T) {
	pipe, err := NewAsyncCopyPipeline(2, 256)
	if err != nil {
		t.Fatalf("NewAsyncCopyPipeline: %v", err)
	}
	if err := pipe.IssueCopy(0, 128); err != nil {
		t.Fatalf("IssueCopy: %v", err)
	}
	// Issue second copy before retiring first
	if err := pipe.IssueCopy(0, 128); err != nil {
		t.Fatalf("IssueCopy second: %v", err)
	}
	hazards := pipe.CheckHazards()
	if len(hazards) == 0 || hazards[0].Kind != "WAW" {
		t.Errorf("expected WAW hazard on overlapping issue into same stage, got %v", hazards)
	}
}
