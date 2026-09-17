package egraph

import "fmt"

var collectiveOperations = map[string]bool{
	"allgather":     true,
	"allreduce":     true,
	"alltoall":      true,
	"broadcast":     true,
	"reducescatter": true,
}

// CollectiveIntent describes one rank-ordered communication operation without
// embedding runtime handles in an e-graph term.
type CollectiveIntent struct {
	Communicator string
	Generation   uint64
	Rank         int
	WorldSize    int
	Operation    string
	DType        string
	Count        uint64
	InPlace      bool
	Stream       uint64
	EffectToken  uint64
}

// Validate checks identity, participation, and payload invariants.
func (c CollectiveIntent) Validate() error {
	if c.Communicator == "" || c.Generation == 0 {
		return fmt.Errorf("egraph: communicator identity and generation required")
	}
	if c.WorldSize <= 0 || c.Rank < 0 || c.Rank >= c.WorldSize {
		return fmt.Errorf("egraph: invalid rank %d for world size %d", c.Rank, c.WorldSize)
	}
	if !collectiveOperations[c.Operation] {
		return fmt.Errorf("egraph: unsupported collective %q", c.Operation)
	}
	if c.DType == "" || c.Count == 0 || c.EffectToken == 0 {
		return fmt.Errorf("egraph: dtype, count, and effect token required")
	}
	return nil
}

// SameEpoch reports whether two intents can participate in one ordered epoch.
func (c CollectiveIntent) SameEpoch(other CollectiveIntent) bool {
	return c.Communicator == other.Communicator && c.Generation == other.Generation && c.WorldSize == other.WorldSize && c.Operation == other.Operation && c.DType == other.DType && c.Count == other.Count && c.EffectToken == other.EffectToken
}

// CanRewrite reports whether reordering preserves the communication effect.
func CanRewrite(before, after CollectiveIntent) bool {
	return before.Validate() == nil && after.Validate() == nil && before.SameEpoch(after) && before.Rank == after.Rank
}
