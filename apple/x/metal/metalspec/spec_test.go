// SPDX-License-Identifier: Apache-2.0

package metalspec

import (
	"testing"
)

func TestNgramDrafter(t *testing.T) {
	drafter := NewNgramDrafter(2)
	history := []int32{1, 2, 3, 4, 1, 2}
	drafts, err := drafter.Draft(history, 2)
	if err != nil {
		t.Fatalf("Draft failed: %v", err)
	}
	if len(drafts) != 2 || drafts[0] != 3 || drafts[1] != 4 {
		t.Errorf("drafts = %v, want [3 4]", drafts)
	}
}

func TestDraftTreeCausalMask(t *testing.T) {
	tree := NewDraftTree(10)
	n1 := tree.AddNode(0, 20)
	tree.AddNode(0, 30)
	tree.AddNode(n1.ID, 25)

	mask := tree.BuildCausalTreeMask()
	if len(mask) != 4 || len(mask[0]) != 4 {
		t.Fatalf("mask dimensions = %dx%d, want 4x4", len(mask), len(mask[0]))
	}
	if !mask[3][0] || !mask[3][1] || !mask[3][3] {
		t.Errorf("Node 3 ancestor mask incorrect: %v", mask[3])
	}
}

func TestVerifyTree(t *testing.T) {
	tree := NewDraftTree(100)
	n1 := tree.AddNode(0, 101)
	tree.AddNode(0, 102)
	tree.AddNode(n1.ID, 103)

	logits := make([][]float32, 4)
	for i := range logits {
		logits[i] = make([]float32, 200)
	}
	logits[0][101] = 10.0
	logits[1][103] = 10.0
	logits[2][99] = 10.0
	logits[3][104] = 10.0

	res := VerifyTree(tree, logits)
	if res.BestPathLen != 3 {
		t.Errorf("BestPathLen = %d, want 3", res.BestPathLen)
	}
	if res.AcceptedPath[0] != 100 || res.AcceptedPath[1] != 101 || res.AcceptedPath[2] != 103 {
		t.Errorf("AcceptedPath = %v, want [100 101 103]", res.AcceptedPath)
	}
	if res.BonusToken != 104 {
		t.Errorf("BonusToken = %d, want 104", res.BonusToken)
	}
}
