package gopccl

// SyncStrategy defines how shared state is reconciled between peers.
type SyncStrategy int

const (
	// EnforcePopular reconciles state to the majority revision.
	EnforcePopular SyncStrategy = iota
	// ReceiveOnly receives state from peers without sending local state.
	ReceiveOnly
	// SendOnly sends local state without receiving peer state.
	SendOnly
)

// StateTensor describes one named buffer in a shared state synchronization.
type StateTensor struct {
	// Name identifies the tensor in the synchronized state.
	Name string
	// Buffer points to the tensor data.
	Buffer Buffer
	// AllowInequality permits content differences for this tensor.
	AllowInequality bool
}

// SyncInfo reports shared state synchronization traffic.
type SyncInfo struct {
	TxBytes uint64
	RxBytes uint64
}
