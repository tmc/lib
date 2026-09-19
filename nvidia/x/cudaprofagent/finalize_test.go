package cudaprofagent

import "testing"

// TestFinalizeReportsNoAgent covers the arm a test binary can reach: nothing
// injected this process, so there is nothing to finalize.
//
// The assertion worth having is the false, not the absence of a crash. A
// Finalize that returned true unconditionally would let a target report that
// it finalized an agent that was never there, which is exactly the confusion
// the boolean exists to prevent.
func TestFinalizeReportsNoAgent(t *testing.T) {
	if Finalize() {
		t.Fatal("Finalize reported an agent in a process with none attached")
	}
	if Finalize() {
		t.Fatal("second call reported an agent: the lookup is not stable")
	}
}
