package cudaspec

import (
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

type failingKVRelease struct {
	allocator *cudakv.BlockAllocator
	failBlock int
	fail      bool
	calls     []int
}

func (r *failingKVRelease) Release(blocks []int) error {
	r.calls = append(r.calls, blocks...)
	for _, b := range blocks {
		if r.fail && b == r.failBlock {
			r.fail = false
			return errors.New("completion not established")
		}
	}
	return r.allocator.Release(blocks)
}
func allocateBranchBlock(t *testing.T, a *cudakv.BlockAllocator, sequence int64) int {
	t.Helper()
	if err := a.AllocSequence(sequence, nil, 1); err != nil {
		t.Fatal(err)
	}
	return a.Table(sequence)[0]
}
func TestRollbackReleaseFailurePreservesRetry(t *testing.T) {
	a := cudakv.NewBlockAllocator(1, 8)
	prefix := allocateBranchBlock(t, a, 1)
	private := allocateBranchBlock(t, a, 2)
	tree := NewBranchTree(1, []int{prefix}, 1)
	b, err := tree.ProposeBranch(0, []int32{2}, []int{private})
	if err != nil {
		t.Fatal(err)
	}
	release := &failingKVRelease{allocator: a, failBlock: private, fail: true}
	if err := tree.RollbackBranch(b.ID, release); err == nil {
		t.Fatal("release failure ignored")
	}
	if b.State != ProposalProposed || !reflect.DeepEqual(b.KVBlocks, []int{private}) {
		t.Fatalf("failed rollback lost ownership: %+v", b)
	}
	if holds, _ := a.Holders(private); holds != 1 {
		t.Fatal("released before completion", holds)
	}
	if err := tree.RollbackBranch(b.ID, release); err != nil {
		t.Fatal(err)
	}
	if err := tree.RollbackBranch(b.ID, release); err != nil {
		t.Fatal(err)
	}
	if len(release.calls) != 2 {
		t.Fatal("duplicate release", release.calls)
	}
	if holds, _ := a.Holders(private); holds != 0 {
		t.Fatal(holds)
	}
	if holds, _ := a.Holders(prefix); holds != 1 {
		t.Fatal("prefix released", holds)
	}
}
func TestRollbackDescendantsBeforeAncestor(t *testing.T) {
	a := cudakv.NewBlockAllocator(1, 8)
	tree := NewBranchTree(1, nil, 1)
	var branches []*Branch
	parent := BranchID(0)
	for i := 0; i < 3; i++ {
		block := allocateBranchBlock(t, a, int64(i+2))
		b, err := tree.ProposeBranch(parent, []int32{int32(i + 2)}, []int{block})
		if err != nil {
			t.Fatal(err)
		}
		branches = append(branches, b)
		parent = b.ID
	}
	release := &failingKVRelease{allocator: a, failBlock: branches[1].KVBlocks[0], fail: true}
	want := []int{branches[2].KVBlocks[0], branches[1].KVBlocks[0], branches[1].KVBlocks[0], branches[0].KVBlocks[0]}
	if err := tree.RollbackBranch(branches[0].ID, release); err == nil {
		t.Fatal("descendant error ignored")
	}
	if branches[0].State != ProposalProposed || branches[1].State != ProposalProposed || branches[2].State != ProposalRolledBack {
		t.Fatal("incorrect partial rollback states")
	}
	if err := tree.RollbackBranch(branches[0].ID, release); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(release.calls, want) {
		t.Fatal(release.calls, want)
	}
	for _, b := range branches {
		if b.State != ProposalRolledBack || len(b.KVBlocks) != 0 {
			t.Fatalf("not rolled back: %+v", b)
		}
	}
}
func TestCommitReleaseFailureDoesNotPublishAcceptance(t *testing.T) {
	a := cudakv.NewBlockAllocator(1, 8)
	tree := NewBranchTree(1, nil, 1)
	acceptedBlock, rejectedBlock := allocateBranchBlock(t, a, 2), allocateBranchBlock(t, a, 3)
	accepted, err := tree.ProposeBranch(0, []int32{2}, []int{acceptedBlock})
	if err != nil {
		t.Fatal(err)
	}
	rejected, err := tree.ProposeBranch(0, []int32{3}, []int{rejectedBlock})
	if err != nil {
		t.Fatal(err)
	}
	release := &failingKVRelease{allocator: a, failBlock: rejectedBlock, fail: true}
	if err := tree.CommitBranch(accepted.ID, []int32{2}, release); err == nil {
		t.Fatal("failure ignored")
	}
	if accepted.State != ProposalProposed || rejected.State != ProposalProposed || tree.ActiveBranch() != 0 {
		t.Fatal("published before release succeeded")
	}
	if err := tree.CommitBranch(accepted.ID, []int32{2}, release); err != nil {
		t.Fatal(err)
	}
	if accepted.State != ProposalCommitted || rejected.State != ProposalRolledBack || tree.ActiveBranch() != accepted.ID {
		t.Fatal("retry did not commit")
	}
	if holds, _ := a.Holders(acceptedBlock); holds != 1 {
		t.Fatal("accepted block released", holds)
	}
}
func TestRollbackCommittedDescendantRefuses(t *testing.T) {
	a := cudakv.NewBlockAllocator(1, 8)
	tree := NewBranchTree(1, nil, 1)
	parentBlock, childBlock := allocateBranchBlock(t, a, 2), allocateBranchBlock(t, a, 3)
	parent, _ := tree.ProposeBranch(0, []int32{2}, []int{parentBlock})
	child, _ := tree.ProposeBranch(parent.ID, []int32{3}, []int{childBlock})
	release := &failingKVRelease{allocator: a}
	if err := tree.CommitBranch(child.ID, []int32{3}, release); err != nil {
		t.Fatal(err)
	}
	if err := tree.RollbackBranch(parent.ID, release); !errors.Is(err, ErrBranchNotActive) {
		t.Fatal(err)
	}
	if len(release.calls) != 0 {
		t.Fatal("released live prefix", release.calls)
	}
}

type blockReleaseFunc func([]int) error

func (f blockReleaseFunc) Release(blocks []int) error { return f(blocks) }

func TestRollbackWaitsForCompletionBeforePublishing(t *testing.T) {
	a := cudakv.NewBlockAllocator(1, 4)
	block := allocateBranchBlock(t, a, 1)
	tree := NewBranchTree(0, nil, 1)
	branch, err := tree.ProposeBranch(0, []int32{2}, []int{block})
	if err != nil {
		t.Fatal(err)
	}
	entered, completed := make(chan struct{}), make(chan struct{})
	release := blockReleaseFunc(func(blocks []int) error { close(entered); <-completed; return a.Release(blocks) })
	done := make(chan error, 1)
	go func() { done <- tree.RollbackBranch(branch.ID, release) }()
	<-entered
	// Release is blocked, so the branch writer cannot advance; the channel
	// establishes ordering for this state observation without a data race.
	state := branch.State
	holders, holderErr := a.Holders(block)
	close(completed)
	rollbackErr := <-done
	if state != ProposalProposed || holders != 1 || holderErr != nil {
		t.Fatalf("premature completion: state=%v holders=%d error=%v", state, holders, holderErr)
	}
	if rollbackErr != nil {
		t.Fatal(rollbackErr)
	}
	if holders, _ := a.Holders(block); holders != 0 {
		t.Fatal(holders)
	}
}
