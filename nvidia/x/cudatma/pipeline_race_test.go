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

func TestAsyncCopyPipeline_WaitGroupKeepsNewest(t *testing.T) {
	p, _ := NewAsyncCopyPipeline(2, 256)
	_ = p.IssueCopy(0, 256)
	_ = p.CommitGroup()
	_ = p.IssueCopy(1, 256)
	_ = p.CommitGroup()
	if err := p.WaitGroup(1); err != nil {
		t.Fatal(err)
	}
	if err := p.ReadStage(1); err != nil {
		t.Fatal(err)
	}
	var raw bool
	for _, h := range p.CheckHazards() {
		raw = raw || h.Kind == "RAW"
	}
	if !raw {
		t.Fatal("WaitGroup(1) retired newest group; expected s1 read-after-write hazard")
	}
}

func TestAsyncCopyPipeline_WaitBeforeCompleteIsHazard(t *testing.T) {
	p, _ := NewAsyncCopyPipeline(1, 512)
	_ = p.ArriveExpectTx(0, 512)
	_ = p.WaitParity(0, 0)
	if len(p.CheckHazards()) == 0 || p.CheckHazards()[0].Kind != "WaitFailure" {
		t.Fatalf("wait before CompleteTx was not reported: %v", p.CheckHazards())
	}
}

func TestAsyncCopyPipeline_WriteInFlightIsHazard(t *testing.T) {
	p, _ := NewAsyncCopyPipeline(1, 512)
	_ = p.ArriveExpectTx(0, 512)
	_ = p.WriteStage(0)
	if len(p.CheckHazards()) == 0 || p.CheckHazards()[0].Kind != "WAR" {
		t.Fatalf("write in flight was not reported: %v", p.CheckHazards())
	}
}

func TestAsyncCopyPipeline_WrongParityDoesNotRetire(t *testing.T) {
	p, _ := NewAsyncCopyPipeline(1, 512)
	_ = p.ArriveExpectTx(0, 512)
	_ = p.CompleteTx(0, 512)
	_ = p.WaitParity(0, 1)
	_ = p.ReadStage(0)
	var failure, raw bool
	for _, h := range p.CheckHazards() {
		failure = failure || h.Kind == "WaitFailure"
		raw = raw || h.Kind == "RAW"
	}
	if !failure || !raw {
		t.Fatalf("wrong parity should fail without retiring stage: %v", p.CheckHazards())
	}
}

func TestTMASharedSourceIsWAR(t *testing.T) {
	bad, err := GenerateTMASharedReadWriteScenario(false)
	if err != nil {
		t.Fatal(err)
	}
	if len(bad.Hazards) != 1 || bad.Hazards[0].Kind != "WAR" {
		t.Fatalf("unfixed shared-source copy = %v, want one WAR", bad.Hazards)
	}
	good, err := GenerateTMASharedReadWriteScenario(true)
	if err != nil {
		t.Fatal(err)
	}
	if good.HasHazards {
		t.Fatalf("fixed shared-source copy still has hazards: %v", good.Hazards)
	}
}
