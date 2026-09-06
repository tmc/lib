//go:build !windows

package nvlshm

// A Bootstrap coordinates the ranks of a distributed buffer during setup.
// It is the only external dependency nvlshm has; torch.distributed fills
// this role in the Python original.
//
// Implementations must provide the usual collective semantics: every rank
// calls the same methods in the same order. Barrier returns only after all
// ranks have entered it. BroadcastString returns root's value on every rank.
type Bootstrap interface {
	// Rank returns this process's rank in [0, WorldSize).
	Rank() int

	// WorldSize returns the number of ranks.
	WorldSize() int

	// BroadcastString sends s from root to all ranks and returns root's
	// value on every rank. Non-root callers pass the zero value.
	BroadcastString(s string, root int) (string, error)

	// Barrier blocks until all ranks have called Barrier.
	Barrier() error
}
