package sampler

import "testing"

func TestThinkBudgetHardQueuesClose(t *testing.T) {
	b := ThinkBudget{Soft: 8, Hard: 3, Rank: 3, Close: []int{10, 11}}
	if id, ok := b.Force(3, true, nil); !ok || id != 10 {
		t.Fatalf("hard force = %d, %v", id, ok)
	}
	if id, ok := b.Force(2, true, nil); !ok || id != 11 {
		t.Fatalf("queued close = %d, %v", id, ok)
	}
	if _, ok := b.Force(2, true, nil); ok {
		t.Fatal("repeated close forced")
	}
}

func TestThinkBudgetSoftRank(t *testing.T) {
	b := ThinkBudget{Soft: 8, Hard: 3, Rank: 3, Close: []int{10}}
	rank := func(id, max int) int {
		if id == 10 && max == 3 {
			return 3
		}
		return 4
	}
	if id, ok := b.Force(8, true, rank); !ok || id != 10 {
		t.Fatalf("soft force = %d, %v", id, ok)
	}
	b = ThinkBudget{Soft: 8, Hard: 3, Rank: 3, Close: []int{10}}
	if _, ok := b.Force(8, true, func(int, int) int { return 4 }); ok {
		t.Fatal("soft force ignored rank")
	}
	if _, ok := b.Force(9, true, rank); ok {
		t.Fatal("soft force crossed limit")
	}
}

func TestThinkBudgetRequiresThinkingAndHardBoundary(t *testing.T) {
	b := ThinkBudget{Soft: 8, Hard: 3, Close: []int{10}}
	if _, ok := b.Force(3, false, nil); ok {
		t.Fatal("forced while not thinking")
	}
	if _, ok := b.Force(4, true, nil); ok {
		t.Fatal("hard force crossed boundary")
	}
	if _, ok := b.Force(3, true, nil); !ok {
		t.Fatal("hard boundary did not force")
	}
}
