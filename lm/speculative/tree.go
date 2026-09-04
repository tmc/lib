// SPDX-License-Identifier: Apache-2.0

package speculative

// TreeNode represents a candidate token node within a speculative candidate tree.
type TreeNode struct {
	ID       int
	Token    int32
	ParentID int
	Children []*TreeNode
	Depth    int
}

// DraftTree holds a non-linear tree of candidate draft paths and its causal attention mask.
type DraftTree struct {
	Nodes []*TreeNode
	Root  *TreeNode
}

// NewDraftTree creates a new DraftTree rooted at rootToken.
func NewDraftTree(rootToken int32) *DraftTree {
	root := &TreeNode{
		ID:       0,
		Token:    rootToken,
		ParentID: -1,
		Depth:    0,
	}
	return &DraftTree{
		Nodes: []*TreeNode{root},
		Root:  root,
	}
}

// AddNode attaches a child candidate token to a parent node in the tree.
func (t *DraftTree) AddNode(parentID int, token int32) *TreeNode {
	if parentID < 0 || parentID >= len(t.Nodes) {
		return nil
	}
	parent := t.Nodes[parentID]
	node := &TreeNode{
		ID:       len(t.Nodes),
		Token:    token,
		ParentID: parentID,
		Depth:    parent.Depth + 1,
	}
	parent.Children = append(parent.Children, node)
	t.Nodes = append(t.Nodes, node)
	return node
}

// BuildCausalTreeMask generates the N x N causal 2D attention mask for single-pass verification.
// An element mask[i][j] is true (attendable) iff node j is an ancestor of node i or j == i.
func (t *DraftTree) BuildCausalTreeMask() [][]bool {
	n := len(t.Nodes)
	mask := make([][]bool, n)
	for i := 0; i < n; i++ {
		mask[i] = make([]bool, n)
		// Trace ancestor path from node i up to root
		curr := i
		for curr != -1 {
			mask[i][curr] = true
			curr = t.Nodes[curr].ParentID
		}
	}
	return mask
}

// TreeVerificationResult returns the longest accepted path in the tree and bonus token.
type TreeVerificationResult struct {
	AcceptedPath []int32
	AcceptedIDs  []int
	NumAccepted  int
	BonusToken   int32
	BestPathLen  int
}

func argmax(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
			maxIdx = i
		}
	}
	return int32(maxIdx)
}

// VerifyTree evaluates all tree candidates against target logits in a single pass.
func VerifyTree(tree *DraftTree, targetLogits [][]float32) TreeVerificationResult {
	if len(tree.Nodes) == 0 || len(targetLogits) == 0 {
		return TreeVerificationResult{}
	}

	// For each node in the tree, check if target predicts it
	nodeValid := make([]bool, len(tree.Nodes))
	nodeValid[0] = true // Root is guaranteed valid

	for i := 1; i < len(tree.Nodes); i++ {
		if i >= len(targetLogits) {
			break
		}
		parentID := tree.Nodes[i].ParentID
		if parentID >= 0 && parentID < len(targetLogits) && nodeValid[parentID] {
			pred := argmax(targetLogits[parentID])
			if pred == tree.Nodes[i].Token {
				nodeValid[i] = true
			}
		}
	}

	// Find the deepest valid leaf in the tree
	bestNodeID := 0
	maxDepth := 0
	for i, valid := range nodeValid {
		if valid && tree.Nodes[i].Depth > maxDepth {
			maxDepth = tree.Nodes[i].Depth
			bestNodeID = i
		}
	}

	// Trace back path from bestNodeID to root
	path := make([]int32, 0, maxDepth+1)
	pathIDs := make([]int, 0, maxDepth+1)
	curr := bestNodeID
	for curr != -1 {
		path = append(path, tree.Nodes[curr].Token)
		pathIDs = append(pathIDs, curr)
		curr = tree.Nodes[curr].ParentID
	}

	// Reverse path to root -> leaf order
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
		pathIDs[i], pathIDs[j] = pathIDs[j], pathIDs[i]
	}

	// Sample bonus token from the best node's prediction
	var bonus int32
	if bestNodeID < len(targetLogits) {
		bonus = argmax(targetLogits[bestNodeID])
	}

	return TreeVerificationResult{
		AcceptedPath: path,
		AcceptedIDs:  pathIDs,
		NumAccepted:  len(path),
		BonusToken:   bonus,
		BestPathLen:  len(path),
	}
}
