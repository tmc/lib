package cudaspec

import (
	"testing"
)

func TestNgramDrafter(t *testing.T) {
	drafter := NewNgramDrafter(3)

	// Feed prompt text tokens: "The capital of France is Paris. The capital of France is Paris."
	tokens := []int32{1, 10, 20, 30, 40, 50, 2, 1, 10, 20, 30, 40, 50, 2}
	drafter.Feed(tokens)

	// Context: "The capital of" -> should draft "France is Paris"
	context := []int32{1, 10, 20}
	draft := drafter.Draft(context, 3)

	if len(draft) != 3 {
		t.Fatalf("expected 3 draft tokens, got %d", len(draft))
	}
	if draft[0] != 30 || draft[1] != 40 || draft[2] != 50 {
		t.Errorf("unexpected draft tokens: %v", draft)
	}
}

func TestVerifyGreedyAllAccepted(t *testing.T) {
	draft := []int32{10, 20, 30}
	targetPredictions := []int32{10, 20, 30, 40}

	result := VerifyGreedy(targetPredictions, draft)
	if result.AcceptedCount != 3 {
		t.Errorf("expected 3 accepted tokens, got %d", result.AcceptedCount)
	}
	if result.BonusToken != 40 {
		t.Errorf("expected bonus token 40, got %d", result.BonusToken)
	}
	if result.TotalEmitted != 4 {
		t.Errorf("expected 4 total emitted tokens, got %d", result.TotalEmitted)
	}
}

func TestVerifyGreedyPartialReject(t *testing.T) {
	draft := []int32{10, 20, 99} // 3rd token is wrong
	targetPredictions := []int32{10, 20, 30, 40}

	result := VerifyGreedy(targetPredictions, draft)
	if result.AcceptedCount != 2 {
		t.Errorf("expected 2 accepted tokens, got %d", result.AcceptedCount)
	}
	if result.BonusToken != 30 {
		t.Errorf("expected correction token 30, got %d", result.BonusToken)
	}
	if result.TotalEmitted != 3 {
		t.Errorf("expected 3 total emitted tokens, got %d", result.TotalEmitted)
	}
}
