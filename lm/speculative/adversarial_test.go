package speculative

import "testing"

func TestNgramDrafterBoundsAndState(t *testing.T) {
	d := NewNgramDrafter(2)
	if got := d.Draft(nil, 3); got != nil {
		t.Fatalf("empty draft = %v, want nil", got)
	}
	d.Feed([]int32{1, 2, 3, 1, 2, 4})
	if got := d.Draft([]int32{1, 2}, 0); got != nil {
		t.Fatalf("zero bound draft = %v, want nil", got)
	}
	if got := d.Draft([]int32{1, 2}, 1); len(got) != 1 || got[0] != 4 {
		t.Fatalf("draft = %v, want most recent continuation [4]", got)
	}
	d.Feed([]int32{1, 2, 5})
	if got := d.Draft([]int32{1, 2}, 2); len(got) == 0 || got[0] != 5 {
		t.Fatalf("draft after feed = %v, want state update beginning with 5", got)
	}
}

func TestNgramDrafterFeedIndexesEachObservationOnce(t *testing.T) {
	d := NewNgramDrafter(2)
	d.Feed([]int32{1, 2, 3})
	d.Feed([]int32{4})
	if got := len(d.table[d.hashNgram([]int32{1, 2})]); got != 1 {
		t.Fatalf("first n-gram observations = %d, want 1", got)
	}
}

func TestVerifyStochasticNilRNG(t *testing.T) {
	got := VerifyStochastic([][]float32{{0.75, 0.25}}, [][]float32{{0.5, 0.5}}, []int32{0}, nil)
	if got.AcceptedCount != 1 {
		t.Fatalf("result = %+v", got)
	}
}

func TestVerifyGreedyEmptyAndImmediateRejection(t *testing.T) {
	if got := VerifyGreedy(nil, nil); got.TotalEmitted != 0 || got.BonusToken != -1 {
		t.Fatalf("empty verification = %+v", got)
	}
	got := VerifyGreedy([]int32{9, 8}, []int32{7})
	if got.AcceptedCount != 0 || len(got.AcceptedTokens) != 0 || got.BonusToken != 9 || got.TotalEmitted != 1 {
		t.Fatalf("immediate rejection = %+v", got)
	}
}

func TestDraftTreeRejectsInvalidParentAndShortLogits(t *testing.T) {
	tree := NewDraftTree(10)
	if got := tree.AddNode(-1, 11); got != nil {
		t.Fatalf("negative parent added node %+v", got)
	}
	if got := tree.AddNode(2, 11); got != nil {
		t.Fatalf("out-of-bounds parent added node %+v", got)
	}
	tree.AddNode(0, 11)
	got := VerifyTree(tree, [][]float32{{0, 1}})
	if got.NumAccepted != 1 || len(got.AcceptedPath) != 1 || got.AcceptedPath[0] != 10 {
		t.Fatalf("short logits verification = %+v", got)
	}
}
