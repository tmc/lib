package cutensornet

import (
	"errors"
	"testing"
)

// The ownership rules are the point of the Workspace, Network and Plan types,
// so they are tested directly on the Go state machine. These tests build the
// structs by hand rather than through the library: the interesting cases are
// the ones that must be refused before any C call happens, and they must be
// checkable on a machine with no GPU and no libcutensornet.

// TestWorkspaceCloseRefusesWhilePlanLive checks that a workspace will not
// release its buffers while a plan the library may still read them through is
// alive. Releasing them there is a use-after-free inside the library.
func TestWorkspaceCloseRefusesWhilePlanLive(t *testing.T) {
	freed := false
	w := &Workspace{plans: 1, frees: []func(){func() { freed = true }}}
	if err := w.Close(); !errors.Is(err, ErrWorkspaceInUse) {
		t.Fatalf("Close with a live plan = %v, want ErrWorkspaceInUse", err)
	}
	if freed {
		t.Error("Close ran the deallocator despite refusing")
	}
	if w.closed {
		t.Error("Close marked the workspace closed despite refusing")
	}
}

// TestWorkspaceProvideRefusesWhilePlanLive checks that a workspace buffer
// cannot be swapped out from under a plan that was sized against it.
func TestWorkspaceProvideRefusesWhilePlanLive(t *testing.T) {
	w := &Workspace{plans: 1}
	err := w.Provide(MemspaceDevice, WorkspaceScratch, 0x1000, 4096, nil)
	if !errors.Is(err, ErrWorkspaceInUse) {
		t.Fatalf("Provide with a live plan = %v, want ErrWorkspaceInUse", err)
	}
}

// TestWorkspaceProvideRejectsInconsistentBuffer checks the argument combinations
// the library documents as invalid, which it reports only later and less
// clearly.
func TestWorkspaceProvideRejectsInconsistentBuffer(t *testing.T) {
	w := &Workspace{}
	if err := w.Provide(MemspaceDevice, WorkspaceScratch, 0, 4096, func() {}); err == nil {
		t.Error("Provide accepted a deallocator for a null buffer")
	}
	if err := w.Provide(MemspaceDevice, WorkspaceScratch, 0x1000, 0, nil); err == nil {
		t.Error("Provide accepted a non-null buffer with size 0")
	}
	if err := w.Provide(MemspaceDevice, WorkspaceScratch, 0x1000, -1, nil); err == nil {
		t.Error("Provide accepted a non-null buffer with a negative size")
	}
}

// TestWorkspaceClosedIsFinal checks that a closed workspace reports
// ErrWorkspaceClosed rather than calling into the library with a destroyed
// descriptor.
func TestWorkspaceClosedIsFinal(t *testing.T) {
	w := &Workspace{closed: true}
	if err := w.Close(); err != nil {
		t.Errorf("Close on a closed workspace = %v, want nil", err)
	}
	if err := w.Provide(MemspaceDevice, WorkspaceScratch, 0x1000, 4096, nil); !errors.Is(err, ErrWorkspaceClosed) {
		t.Errorf("Provide after Close = %v, want ErrWorkspaceClosed", err)
	}
	if _, err := w.Size(WorksizePrefMin, MemspaceDevice, WorkspaceScratch); !errors.Is(err, ErrWorkspaceClosed) {
		t.Errorf("Size after Close = %v, want ErrWorkspaceClosed", err)
	}
	if err := w.PurgeCache(MemspaceDevice); !errors.Is(err, ErrWorkspaceClosed) {
		t.Errorf("PurgeCache after Close = %v, want ErrWorkspaceClosed", err)
	}
	if _, err := w.NewPlan(&Network{}, 0); !errors.Is(err, ErrWorkspaceClosed) {
		t.Errorf("NewPlan after Close = %v, want ErrWorkspaceClosed", err)
	}
}

// TestNetworkCloseRefusesWhilePlanLive checks that the network descriptor, which
// the library reads during contraction, outlives every plan built from it.
func TestNetworkCloseRefusesWhilePlanLive(t *testing.T) {
	n := &Network{plans: 1}
	if err := n.Close(); !errors.Is(err, ErrNetworkInUse) {
		t.Fatalf("Close with a live plan = %v, want ErrNetworkInUse", err)
	}
	if n.closed {
		t.Error("Close marked the network closed despite refusing")
	}
}

// TestPlanCloseReleasesClaims checks that closing a plan is what makes the
// workspace and the network closeable again, and that closing twice does not
// release the claim twice.
func TestPlanCloseReleasesClaims(t *testing.T) {
	w := &Workspace{plans: 1}
	n := &Network{plans: 1}
	p := &Plan{ws: w, net: n}

	// Close destroys the plan handle through the library, which is not
	// available here, so exercise the claim release on its own and check the
	// bookkeeping. Destroying a real plan is a physical test.
	p.releaseClaims()
	if w.plans != 0 || n.plans != 0 {
		t.Fatalf("after plan close: workspace plans = %d, network plans = %d, want 0 and 0", w.plans, n.plans)
	}
	p.releaseClaims()
	if w.plans != 0 || n.plans != 0 {
		t.Fatalf("a second close released the claims twice: workspace plans = %d, network plans = %d", w.plans, n.plans)
	}
}

// TestPlanClosedRejectsUse checks that a closed plan does not reach the library
// with a destroyed handle.
func TestPlanClosedRejectsUse(t *testing.T) {
	p := &Plan{ws: &Workspace{}, net: &Network{numInputs: 2}, closed: true}
	if err := p.Contract([]uintptr{1, 2}, 3, false, 0, 0); !errors.Is(err, ErrPlanClosed) {
		t.Errorf("Contract after Close = %v, want ErrPlanClosed", err)
	}
	if err := p.Autotune([]uintptr{1, 2}, 3, 0, 0); !errors.Is(err, ErrPlanClosed) {
		t.Errorf("Autotune after Close = %v, want ErrPlanClosed", err)
	}
	if err := p.Close(); err != nil {
		t.Errorf("Close on a closed plan = %v, want nil", err)
	}
}

// TestPlanContractChecksOperands checks that the operand addresses are matched
// against the network the plan was built from. The library takes a bare pointer
// array with no length, so a short slice would be read past its end.
func TestPlanContractChecksOperands(t *testing.T) {
	p := &Plan{ws: &Workspace{}, net: &Network{numInputs: 3}}
	if err := p.Contract([]uintptr{1, 2}, 4, false, 0, 0); err == nil {
		t.Error("Contract accepted 2 addresses for a 3-input network")
	}
	if err := p.Contract([]uintptr{1, 2, 3, 4}, 5, false, 0, 0); err == nil {
		t.Error("Contract accepted 4 addresses for a 3-input network")
	}
	if err := p.Contract([]uintptr{1, 0, 3}, 4, false, 0, 0); err == nil {
		t.Error("Contract accepted a null input address")
	}
	if err := p.Contract([]uintptr{1, 2, 3}, 0, false, 0, 0); err == nil {
		t.Error("Contract accepted a null output address")
	}
	if err := p.Contract(nil, 4, false, 0, 0); err == nil {
		t.Error("Contract accepted a nil operand slice")
	}
}
