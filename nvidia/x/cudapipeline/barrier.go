package cudapipeline

// BarrierState tracks the lifecycle and transaction state of an individual mbarrier object.
type BarrierState struct {
	Name             string         // symbol or register operand representing the barrier
	Initialized      bool           // true if mbarrier.init was executed
	Invalidated      bool           // true if mbarrier.inval was executed
	InitLine         int            // line number where initialized
	ExpectedArrivals int            // expected arrival tokens per phase
	PendingArrivals  int            // remaining arrival tokens needed for current phase
	Phase            int            // current primary phase index (0, 1, 2, ...)
	ExpectedTxBytes  int64          // transaction bytes expected in current phase
	CompletedTxBytes int64          // transaction bytes signaled as complete in current phase
	HasExpectedTx    bool           // true if an expect_tx operation was executed in current phase
	Arrived          bool           // true if at least one arrive operation was executed in current phase
	CopyCount        int            // number of async copy operations targeting this barrier in current phase
	Events           []BarrierEvent // chronological history of operations on this barrier
}

// Parity returns the current expected phase parity (0 or 1).
func (s *BarrierState) Parity() int {
	return s.Phase % 2
}

// TxPending returns the count of expected transaction bytes still uncompleted.
func (s *BarrierState) TxPending() int64 {
	return s.ExpectedTxBytes - s.CompletedTxBytes
}

// BarrierTracker tracks mbarrier states, arrival tokens, transaction byte counts,
// and phase parity across PTX instruction streams.
//
// The zero value is not usable; use NewBarrierTracker.
type BarrierTracker struct {
	barriers          map[string]*BarrierState
	tensorSizes       map[string]int64
	barrierArrivals   map[string]int
	regs              map[string]int64
	regSyms           map[string]string
	targetArch        string
	tmemAllocatedCols int
}

// NewBarrierTracker constructs an initialized BarrierTracker configured with opts.
func NewBarrierTracker(opts ...Option) *BarrierTracker {
	t := &BarrierTracker{
		barriers:        make(map[string]*BarrierState),
		tensorSizes:     make(map[string]int64),
		barrierArrivals: make(map[string]int),
		regs:            make(map[string]int64),
		regSyms:         make(map[string]string),
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// SetTensorSize associates a tensor-map parameter or register name with a known transfer byte size.
func (t *BarrierTracker) SetTensorSize(tensorMap string, bytes int64) {
	t.tensorSizes[tensorMap] = bytes
}

// Barrier returns the state of the named barrier, or nil if not encountered.
func (t *BarrierTracker) Barrier(name string) *BarrierState {
	name = cleanAddress(name)
	if resolved := t.resolveSymbol(name); resolved != "" {
		name = resolved
	}
	return t.barriers[name]
}

// Barriers returns a map of all tracked barriers keyed by name.
func (t *BarrierTracker) Barriers() map[string]*BarrierState {
	out := make(map[string]*BarrierState, len(t.barriers))
	for k, v := range t.barriers {
		out[k] = v
	}
	return out
}

// Reset clears the tracker's transient barrier states and registers while preserving configured tensor sizes.
func (t *BarrierTracker) Reset() {
	t.barriers = make(map[string]*BarrierState)
	t.regs = make(map[string]int64)
	t.regSyms = make(map[string]string)
	t.tmemAllocatedCols = 0
}

func (t *BarrierTracker) resolveSymbol(sym string) string {
	sym = cleanAddress(sym)
	if s, ok := t.regSyms[sym]; ok && s != "" {
		return s
	}
	return sym
}

func (t *BarrierTracker) getTensorSize(name string) (int64, bool) {
	name = cleanAddress(name)
	if sz, ok := t.tensorSizes[name]; ok {
		return sz, true
	}
	if resolved := t.resolveSymbol(name); resolved != "" && resolved != name {
		if sz, ok := t.tensorSizes[resolved]; ok {
			return sz, true
		}
	}
	return 0, false
}

func (t *BarrierTracker) getOrCreateBarrier(name string) *BarrierState {
	name = cleanAddress(name)
	if resolved := t.resolveSymbol(name); resolved != "" {
		name = resolved
	}
	b, ok := t.barriers[name]
	if !ok {
		b = &BarrierState{
			Name: name,
		}
		t.barriers[name] = b
	}
	return b
}
