// SPDX-License-Identifier: Apache-2.0

package speculative

import (
	"testing"
)

func TestDraftTreeCausalMask(t *testing.T) {
	// Build tree:
	//       Root (0, tok 10)
	//       /             \
	//   N1 (1, tok 20)   N2 (2, tok 30)
	//     |
	//   N3 (3, tok 25)
	tree := NewDraftTree(10)
	n1 := tree.AddNode(0, 20)
	tree.AddNode(0, 30)
	tree.AddNode(n1.ID, 25)

	mask := tree.BuildCausalTreeMask()
	if len(mask) != 4 || len(mask[0]) != 4 {
		t.Fatalf("mask dimensions = %dx%d, want 4x4", len(mask), len(mask[0]))
	}

	// Node 3 (child of Node 1, child of Node 0) should attend to 0, 1, 3 (not 2)
	if !mask[3][0] || !mask[3][1] || !mask[3][3] {
		t.Errorf("Node 3 ancestor mask incorrect: %v", mask[3])
	}
	if mask[3][2] {
		t.Errorf("Node 3 should NOT attend to sibling Node 2")
	}
}

func TestVerifyTree(t *testing.T) {
	tree := NewDraftTree(100)
	n1 := tree.AddNode(0, 101)
	tree.AddNode(0, 102)
	tree.AddNode(n1.ID, 103)

	// Target logits predicting path 100 -> 101 -> 103 -> bonus 104
	logits := make([][]float32, 4)
	for i := range logits {
		logits[i] = make([]float32, 200)
	}
	logits[0][101] = 10.0 // Root predicts N1 (101)
	logits[1][103] = 10.0 // N1 predicts N3 (103)
	logits[2][999%200] = 10.0
	logits[3][104] = 10.0 // N3 predicts bonus 104

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
