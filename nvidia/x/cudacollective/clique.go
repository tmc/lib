package cudacollective

import (
	"fmt"
	"sync"
)

// A Clique is the failure authority shared by every Communicator of one
// local collective lineage. Create it once per intended clique with
// NewClique and pass the same pointer to every rank and every
// reconstruction. Two Cliques never share failures, even when their
// memberships are equal.
//
// A Clique records a failed-through generation floor. MarkFailed and
// ReportLostParticipant raise the floor to the reporting communicator's
// generation before they return. Communicators of that generation or an
// older one then refuse admission, and NewCommunicator refuses to create
// them, even with a fresh Transport. A caller may deliberately create a
// Communicator with a larger generation on the same Clique. The floor never
// lowers and nothing advances generations automatically.
//
// A Clique is a local host capability. It does not cross process boundaries,
// establish distributed agreement, recover native resources, or prove that
// work admitted before a failure is quiescent.
//
// The zero value is not usable; NewCommunicator refuses it.
type Clique struct {
	membership Membership

	mu    sync.RWMutex
	floor Generation // highest failed generation; 0 before any failure
	rank  int        // rank that raised floor
}

// NewClique returns a new Clique for the ranks in m.
// m must be a nonempty Membership returned by NewMembership.
func NewClique(m Membership) (*Clique, error) {
	if m.WorldSize() <= 0 {
		return nil, fmt.Errorf("%w: clique membership is empty", ErrInvalidWorldSize)
	}
	return &Clique{membership: m}, nil
}

// fail raises the floor to g on behalf of rank.
// The caller holds q.mu for writing.
func (q *Clique) fail(g Generation, rank int) {
	if g > q.floor {
		q.floor = g
		q.rank = rank
	}
}

// check reports whether generation g is at or below the floor.
// The caller holds q.mu.
func (q *Clique) check(g Generation) error {
	if g > q.floor {
		return nil
	}
	if q.floor == ^Generation(0) {
		return fmt.Errorf("%w: generation %d failed through %d by rank %d; clique generations exhausted", ErrCommunicatorGeneration, g, q.floor, q.rank)
	}
	return fmt.Errorf("%w: generation %d failed through %d by rank %d", ErrCommunicatorGeneration, g, q.floor, q.rank)
}
