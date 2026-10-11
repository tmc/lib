package egraph

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
)

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

// Validate checks identity, participation, supported dtype, and payload size.
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
	dtype, ok := collectiveDataTypes[strings.ToLower(c.DType)]
	if !ok {
		return fmt.Errorf("egraph: unsupported dtype %q", c.DType)
	}
	if c.Count > uint64(^uint(0)>>1)/uint64(dtype.SizeBytes()) {
		return fmt.Errorf("egraph: collective payload overflows runtime int")
	}
	return nil
}

// OperationDescriptor converts a validated intent into the existing
// collective runtime contract. It does not select a transport or make the
// effectful operation eligible for algebraic reordering.
func (c CollectiveIntent) OperationDescriptor() (cudacollective.OperationDescriptor, error) {
	if err := c.Validate(); err != nil {
		return cudacollective.OperationDescriptor{}, err
	}
	kind, ok := collectiveKinds[c.Operation]
	if !ok {
		return cudacollective.OperationDescriptor{}, fmt.Errorf("egraph: unsupported collective %q", c.Operation)
	}
	dtype, ok := collectiveDataTypes[strings.ToLower(c.DType)]
	if !ok {
		return cudacollective.OperationDescriptor{}, fmt.Errorf("egraph: unsupported dtype %q", c.DType)
	}
	return cudacollective.OperationDescriptor{
		Kind:      kind,
		DataType:  dtype,
		Count:     int(c.Count),
		InPlace:   c.InPlace,
		Topology:  cudacollective.TopologyAny,
		Root:      0,
		Reduction: cudacollective.ReductionSum,
	}, nil
}

var collectiveKinds = map[string]cudacollective.Kind{
	"allgather":     cudacollective.KindAllGather,
	"allreduce":     cudacollective.KindAllReduce,
	"alltoall":      cudacollective.KindAllToAll,
	"broadcast":     cudacollective.KindBroadcast,
	"reducescatter": cudacollective.KindReduceScatter,
}

var collectiveDataTypes = map[string]cudair.DataType{
	"f32":      cudair.Float32,
	"float32":  cudair.Float32,
	"f16":      cudair.Float16,
	"float16":  cudair.Float16,
	"bf16":     cudair.BFloat16,
	"bfloat16": cudair.BFloat16,
	"i32":      cudair.Int32,
	"int32":    cudair.Int32,
	"i64":      cudair.Int64,
	"int64":    cudair.Int64,
	"u8":       cudair.Uint8,
	"uint8":    cudair.Uint8,
	"i8":       cudair.Int8,
	"int8":     cudair.Int8,
}

// SameEpoch reports whether two intents can participate in one ordered epoch.
func (c CollectiveIntent) SameEpoch(other CollectiveIntent) bool {
	return c.Communicator == other.Communicator && c.Generation == other.Generation && c.WorldSize == other.WorldSize && c.Operation == other.Operation && c.DType == other.DType && c.Count == other.Count && c.EffectToken == other.EffectToken
}

// CanRewrite reports whether reordering preserves the communication effect,
// including the participating rank, stream, and in-place buffer contract.
func CanRewrite(before, after CollectiveIntent) bool {
	return before.Validate() == nil && after.Validate() == nil && before.SameEpoch(after) &&
		before.Rank == after.Rank && before.Stream == after.Stream && before.InPlace == after.InPlace
}
