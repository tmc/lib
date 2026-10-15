package egraph

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// Errors reported by collective planning and replay.
var (
	// ErrCollectiveUnsupported reports a collective contract outside the
	// bounded lowering: a non-sum reduction, a non-contiguous layout, a
	// partition that does not divide the count, a known operation other than
	// allreduce, a capturing or cancelled stream, a replay dtype other than
	// f32, or a transport Replay cannot drive one rank at a time.
	ErrCollectiveUnsupported = errors.New("egraph: unsupported collective contract")

	// ErrCollectiveMismatch reports ranks that disagree on the ordered
	// collective sequence or on a call's communicator, payload or reduction.
	ErrCollectiveMismatch = errors.New("egraph: rank-wide collective mismatch")

	// ErrCollectiveAlias reports buffer extents that do not match the payload
	// or send and receive ranges that violate the in-place contract.
	ErrCollectiveAlias = errors.New("egraph: collective buffer alias violation")

	// ErrCollectiveToken reports a missing completion token or a broken
	// predecessor chain on a rank's stream.
	ErrCollectiveToken = errors.New("egraph: collective completion token violation")

	// ErrCollectiveQuarantined reports a plan identity whose replay failed.
	ErrCollectiveQuarantined = errors.New("egraph: collective plan quarantined")
)

// CollectiveStreamState is the explicit stream state at a collective call.
// The zero value is invalid.
type CollectiveStreamState int

const (
	// CollectiveStreamOrdered is an ordinary stream-ordered launch.
	CollectiveStreamOrdered CollectiveStreamState = iota + 1
	// CollectiveStreamCapturing is a launch inside graph capture. Lowering
	// refuses it.
	CollectiveStreamCapturing
	// CollectiveStreamCancelled is a launch whose stream work was cancelled.
	// Lowering refuses it.
	CollectiveStreamCancelled
)

// CollectiveBuffer is a byte range in one rank's device address space.
// Address is the nonzero device address of the buffer; ranges of a call are
// compared as linear addresses. Host replay never dereferences it.
type CollectiveBuffer struct {
	Address uint64
	Offset  uint64
	Bytes   uint64
}

// CollectiveCall is one rank's collective intent with the payload, buffer,
// ordering and stream contract needed to lower it. The bounded lowering
// accepts only allreduce with a sum reduction over a contiguous layout.
type CollectiveCall struct {
	CollectiveIntent

	// Reduction names the reduction operator; only "sum" lowers.
	Reduction string
	// Root is the root rank passed to the runtime descriptor.
	Root int
	// Layout names the element layout; only "contiguous" lowers.
	Layout string
	// Partition is the number of equal contiguous chunks of the payload.
	// It must divide Count and match on every rank. The runtime descriptor
	// has no partition field, and the elementwise host sum does not depend
	// on it.
	Partition int

	Send, Recv CollectiveBuffer

	// Predecessor is the completion token of the previous collective on
	// this rank's stream. Completion is the nonzero token this call produces.
	Predecessor uint64
	Completion  uint64

	StreamState CollectiveStreamState

	// Estimate is a modeled duration. It is never compared with or copied
	// into replay observations.
	Estimate time.Duration
}

// Validate checks the call on its own rank.
func (c CollectiveCall) Validate() error {
	if err := c.CollectiveIntent.Validate(); err != nil {
		return err
	}
	if c.Operation != "allreduce" {
		return fmt.Errorf("%w: operation %q", ErrCollectiveUnsupported, c.Operation)
	}
	switch c.Reduction {
	case "":
		return errors.New("egraph: collective reduction required")
	case "sum":
	default:
		return fmt.Errorf("%w: reduction %q", ErrCollectiveUnsupported, c.Reduction)
	}
	switch c.Layout {
	case "":
		return errors.New("egraph: collective layout required")
	case "contiguous":
	default:
		return fmt.Errorf("%w: layout %q", ErrCollectiveUnsupported, c.Layout)
	}
	if c.Root < 0 || c.Root >= c.WorldSize {
		return fmt.Errorf("egraph: collective root %d outside world size %d", c.Root, c.WorldSize)
	}
	if c.Partition <= 0 || c.Count%uint64(c.Partition) != 0 {
		return fmt.Errorf("%w: partition %d does not divide count %d", ErrCollectiveUnsupported, c.Partition, c.Count)
	}
	switch c.StreamState {
	case CollectiveStreamOrdered:
	case CollectiveStreamCapturing:
		return fmt.Errorf("%w: collective inside graph capture", ErrCollectiveUnsupported)
	case CollectiveStreamCancelled:
		return fmt.Errorf("%w: collective on cancelled stream", ErrCollectiveUnsupported)
	default:
		return fmt.Errorf("egraph: invalid collective stream state %d", c.StreamState)
	}
	if c.Completion == 0 || c.Completion == c.Predecessor {
		return fmt.Errorf("%w: completion %d after predecessor %d", ErrCollectiveToken, c.Completion, c.Predecessor)
	}
	return c.checkBuffers()
}

func (c CollectiveCall) checkBuffers() error {
	bytes := c.Count * uint64(collectiveDataTypes[strings.ToLower(c.DType)].SizeBytes())
	for _, b := range []CollectiveBuffer{c.Send, c.Recv} {
		if b.Address == 0 {
			return fmt.Errorf("%w: nil buffer", ErrCollectiveAlias)
		}
		if b.Bytes != bytes {
			return fmt.Errorf("%w: extent %d bytes, payload %d bytes", ErrCollectiveAlias, b.Bytes, bytes)
		}
		if b.Address+b.Offset < b.Address || b.Address+b.Offset+b.Bytes < b.Address+b.Offset {
			return fmt.Errorf("%w: buffer range overflows", ErrCollectiveAlias)
		}
	}
	same := c.Send == c.Recv
	if c.InPlace != same {
		if c.InPlace {
			return fmt.Errorf("%w: in-place call with distinct send and receive ranges", ErrCollectiveAlias)
		}
		return fmt.Errorf("%w: out-of-place call with identical send and receive ranges", ErrCollectiveAlias)
	}
	if !c.InPlace && c.Send.overlaps(c.Recv) {
		return fmt.Errorf("%w: send and receive ranges overlap", ErrCollectiveAlias)
	}
	return nil
}

func (b CollectiveBuffer) overlaps(o CollectiveBuffer) bool {
	lo, hi := b.Address+b.Offset, b.Address+b.Offset+b.Bytes
	olo, ohi := o.Address+o.Offset, o.Address+o.Offset+o.Bytes
	return lo < ohi && olo < hi
}

// OperationDescriptor lowers the validated call to the runtime contract,
// carrying the call's explicit reduction and root.
func (c CollectiveCall) OperationDescriptor() (cudacollective.OperationDescriptor, error) {
	if err := c.Validate(); err != nil {
		return cudacollective.OperationDescriptor{}, err
	}
	return cudacollective.OperationDescriptor{
		Kind:      cudacollective.KindAllReduce,
		DataType:  collectiveDataTypes[strings.ToLower(c.DType)],
		Count:     int(c.Count),
		Root:      c.Root,
		Reduction: cudacollective.ReductionSum,
		Topology:  cudacollective.TopologyAny,
		InPlace:   c.InPlace,
	}, nil
}

// sameCollective reports whether two ranks' calls describe the same
// collective operation.
func (c CollectiveCall) sameCollective(o CollectiveCall) bool {
	return c.SameEpoch(o.CollectiveIntent) && c.Reduction == o.Reduction && c.Root == o.Root &&
		c.Layout == o.Layout && c.Partition == o.Partition
}

// CheckCollectiveRewrite reports whether after is an acceptable rewrite of
// before on one rank. A rewrite may change estimates only: it cannot drop,
// add, reorder or alter a collective call.
func CheckCollectiveRewrite(before, after []CollectiveCall) error {
	if len(after) != len(before) {
		return fmt.Errorf("%w: rewrite changed %d collectives to %d", ErrCollectiveMismatch, len(before), len(after))
	}
	for i := range before {
		b, a := before[i], after[i]
		b.Estimate, a.Estimate = 0, 0
		if b != a {
			return fmt.Errorf("%w: rewrite changed collective %d", ErrCollectiveMismatch, i)
		}
	}
	return nil
}

// CollectiveQuarantine records plan identities whose replay failed.
// The zero value is an empty quarantine ready to use.
type CollectiveQuarantine struct {
	mu     sync.Mutex
	failed map[string]error
}

// Err returns a non-nil error wrapping ErrCollectiveQuarantined if id failed.
func (q *CollectiveQuarantine) Err(id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if err, ok := q.failed[id]; ok {
		return fmt.Errorf("%w: %s: %w", ErrCollectiveQuarantined, id, err)
	}
	return nil
}

func (q *CollectiveQuarantine) add(id string, err error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.failed == nil {
		q.failed = make(map[string]error)
	}
	if _, ok := q.failed[id]; !ok {
		q.failed[id] = err
	}
}

// CollectiveSchedule is a rank-wide collective schedule to lower.
// Calls[r] is rank r's ordered call sequence and Communicators[r] is rank
// r's runtime communicator.
type CollectiveSchedule struct {
	Communicators []*cudacollective.Communicator
	Calls         [][]CollectiveCall
	// Quarantine records failed plans. It must be non-nil.
	Quarantine *CollectiveQuarantine
	// Now is the clock for replay observations. Nil means time.Now.
	Now func() time.Time
}

// CollectivePlan is a schedule validated across ranks and lowered onto one
// cudacollective.Plan per rank and collective. The zero value is not usable;
// create one with CollectiveSchedule.Plan.
type CollectivePlan struct {
	mu         sync.Mutex // serializes Replay and Close
	id         string
	calls      [][]CollectiveCall
	comms      []*cudacollective.Communicator
	runtime    [][]*cudacollective.Plan // [collective][rank]
	quarantine *CollectiveQuarantine
	now        func() time.Time
	closed     bool
}

// Plan validates s and lowers it. Every rank must hold the same ordered
// sequence of collectives with matching communicator, generation, world
// size, payload, reduction, root, layout and partition; each rank's calls
// must form one acyclic predecessor/completion chain on one stream; and each
// communicator must be alive, hold rank r of the call's generation and
// world size, share rank 0's membership and transport type, and have an
// available transport of type cudacollective.TransportTypeControlPlaneTest,
// the only type Replay can drive one rank at a time. Ranks may use distinct
// transport instances of that type.
func (s CollectiveSchedule) Plan() (*CollectivePlan, error) {
	if s.Quarantine == nil {
		return nil, errors.New("egraph: collective schedule needs a quarantine")
	}
	if err := s.validate(); err != nil {
		return nil, err
	}
	id := collectiveIdentity(s.Calls)
	if err := s.Quarantine.Err(id); err != nil {
		return nil, err
	}
	p := &CollectivePlan{
		id:         id,
		comms:      append([]*cudacollective.Communicator(nil), s.Communicators...),
		quarantine: s.Quarantine,
		now:        s.Now,
	}
	if p.now == nil {
		p.now = time.Now
	}
	for _, calls := range s.Calls {
		p.calls = append(p.calls, append([]CollectiveCall(nil), calls...))
	}
	for i := range p.calls[0] {
		var step []*cudacollective.Plan
		for r, comm := range p.comms {
			c := p.calls[r][i]
			rp, err := cudacollective.NewPlan(comm, nil)
			if err != nil {
				p.closeRuntime()
				return nil, fmt.Errorf("egraph: rank %d plan: %w", r, err)
			}
			step = append(step, rp)
			desc, err := c.OperationDescriptor()
			if err == nil {
				send := driver.DevicePtr(c.Send.Address + c.Send.Offset)
				recv := driver.DevicePtr(c.Recv.Address + c.Recv.Offset)
				err = rp.AddCollectiveStep(fmt.Sprintf("effect-%d", c.EffectToken), desc, send, recv)
			}
			if err != nil {
				p.runtime = append(p.runtime, step)
				p.closeRuntime()
				return nil, fmt.Errorf("egraph: rank %d effect %d: %w", r, c.EffectToken, err)
			}
		}
		p.runtime = append(p.runtime, step)
	}
	return p, nil
}

func (s CollectiveSchedule) validate() error {
	if len(s.Calls) == 0 || len(s.Calls[0]) == 0 {
		return errors.New("egraph: empty collective schedule")
	}
	first := s.Calls[0][0]
	if len(s.Calls) != first.WorldSize || len(s.Communicators) != first.WorldSize {
		return fmt.Errorf("%w: %d call sequences and %d communicators for world size %d",
			ErrCollectiveMismatch, len(s.Calls), len(s.Communicators), first.WorldSize)
	}
	completions := make(map[uint64]bool)
	for _, calls := range s.Calls {
		for _, c := range calls {
			completions[c.Completion] = true
		}
	}
	for r, calls := range s.Calls {
		if len(calls) != len(s.Calls[0]) {
			return fmt.Errorf("%w: rank %d has %d collectives, rank 0 has %d", ErrCollectiveMismatch, r, len(calls), len(s.Calls[0]))
		}
		if err := checkCommunicator(s.Communicators[r], s.Communicators[0], r, first); err != nil {
			return err
		}
		// The first call waits on work outside the schedule; waiting on a
		// completion inside it would close a cycle.
		if completions[calls[0].Predecessor] {
			return fmt.Errorf("%w: rank %d first collective waits on schedule completion %d",
				ErrCollectiveToken, r, calls[0].Predecessor)
		}
		seen, completed := make(map[uint64]bool), make(map[uint64]bool)
		for i, c := range calls {
			if err := c.Validate(); err != nil {
				return fmt.Errorf("egraph: rank %d collective %d: %w", r, i, err)
			}
			if c.Rank != r {
				return fmt.Errorf("%w: rank %d holds a call for rank %d", ErrCollectiveMismatch, r, c.Rank)
			}
			if c.Communicator != first.Communicator || c.Generation != first.Generation || c.WorldSize != first.WorldSize {
				return fmt.Errorf("%w: rank %d collective %d changes communicator epoch", ErrCollectiveMismatch, r, i)
			}
			if !c.sameCollective(s.Calls[0][i]) {
				return fmt.Errorf("%w: rank %d collective %d differs from rank 0", ErrCollectiveMismatch, r, i)
			}
			if seen[c.EffectToken] {
				return fmt.Errorf("%w: rank %d repeats effect %d", ErrCollectiveMismatch, r, c.EffectToken)
			}
			seen[c.EffectToken] = true
			if completed[c.Completion] {
				return fmt.Errorf("%w: rank %d repeats completion %d", ErrCollectiveToken, r, c.Completion)
			}
			completed[c.Completion] = true
			if i > 0 {
				prev := calls[i-1]
				if c.Stream != prev.Stream {
					return fmt.Errorf("%w: rank %d collective %d changes stream", ErrCollectiveToken, r, i)
				}
				if c.Predecessor != prev.Completion {
					return fmt.Errorf("%w: rank %d collective %d waits on %d, predecessor completes %d",
						ErrCollectiveToken, r, i, c.Predecessor, prev.Completion)
				}
			}
		}
	}
	return nil
}

func checkCommunicator(comm, comm0 *cudacollective.Communicator, rank int, c CollectiveCall) error {
	if comm == nil || comm0 == nil {
		return fmt.Errorf("%w: rank %d has no communicator", ErrCollectiveMismatch, rank)
	}
	if comm.Rank() != rank || comm.Membership().WorldSize() != c.WorldSize {
		return fmt.Errorf("%w: communicator is rank %d of %d, want rank %d of %d",
			ErrCollectiveMismatch, comm.Rank(), comm.Membership().WorldSize(), rank, c.WorldSize)
	}
	if !slices.Equal(comm.Membership().Ranks(), comm0.Membership().Ranks()) ||
		comm.Transport().Type() != comm0.Transport().Type() {
		return fmt.Errorf("%w: rank %d communicator membership or transport differs from rank 0", ErrCollectiveMismatch, rank)
	}
	if uint64(comm.Generation()) != c.Generation {
		return fmt.Errorf("%w: communicator generation %d, call generation %d",
			ErrCollectiveMismatch, comm.Generation(), c.Generation)
	}
	if err := comm.CheckAlive(); err != nil {
		return fmt.Errorf("egraph: rank %d communicator: %w", rank, err)
	}
	if !comm.Transport().Available(comm.Membership()) {
		return fmt.Errorf("egraph: rank %d: %w: transport %v unavailable",
			rank, cudacollective.ErrExecutionUnavailable, comm.Transport().Type())
	}
	// Replay drives ranks one after another on one goroutine, which only a
	// transport that never waits for peers allows. NCCL, for one, blocks.
	if comm.Transport().Type() != cudacollective.TransportTypeControlPlaneTest {
		return fmt.Errorf("%w: rank %d transport %v cannot be driven one rank at a time",
			ErrCollectiveUnsupported, rank, comm.Transport().Type())
	}
	return nil
}

func collectiveIdentity(calls [][]CollectiveCall) string {
	h := sha256.New()
	for r, rank := range calls {
		for _, c := range rank {
			c.Estimate = 0
			fmt.Fprintf(h, "%d:%+v\n", r, c)
		}
	}
	return hex.EncodeToString(h.Sum(nil)[:8])
}

// ID returns the plan identity, a digest of every rank's calls excluding
// estimates.
func (p *CollectivePlan) ID() string { return p.id }

// Close closes the runtime plans. It does not close communicators.
// Replay after Close returns an error without failing communicators.
func (p *CollectivePlan) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	return p.closeRuntime()
}

func (p *CollectivePlan) closeRuntime() error {
	var errs []error
	for _, step := range p.runtime {
		for _, rp := range step {
			errs = append(errs, rp.Close())
		}
	}
	return errors.Join(errs...)
}

// CollectiveStepRecord is the recorded result of one collective step.
// Outputs are copies owned by the record.
type CollectiveStepRecord struct {
	EffectToken uint64
	Completions []uint64    // by rank
	Outputs     [][]float32 // by rank
}

// CollectiveObservation is the measured duration of one rank's runtime
// execution of one collective during replay.
type CollectiveObservation struct {
	Step    int
	Rank    int
	Elapsed time.Duration
}

// CollectiveReceipt records a replay. Estimated is the sum of rank 0's call
// estimates; Observations are measured with the schedule's clock and never
// derived from estimates.
type CollectiveReceipt struct {
	PlanID       string
	Transport    cudacollective.TransportType
	Steps        []CollectiveStepRecord
	Estimated    time.Duration
	Observations []CollectiveObservation
}

// Replay executes the plan and records the result. inputs[i][r] is rank r's
// f32 payload for collective i.
//
// Collectives run in order. For each one, Replay holds every rank's send and
// receive buffers through its communicator, executes each rank's
// cudacollective.Plan through the communicator's transport in rank order on
// one goroutine, then sums the ranks' inputs on the host in ascending rank
// order and releases the buffers. The transport does not move the data.
// In-place calls write the sum into the rank's input slice.
//
// Replay refuses without side effects a closed plan, a communicator that is
// closed or failed before replay starts, a buffer held elsewhere, and a
// context cancelled before a collective starts. It marks every communicator
// failed and quarantines the plan identity only for a failure of its own
// execution: when some ranks executed a collective and another rank did not,
// or when the runtime records a failed generation or lost participant while
// executing a collective. A communicator that fails or closes concurrently
// with that execution is attributed to the plan. Cancellation inside a
// transport is the transport's to report. In-place inputs of collectives
// that completed before an error are already overwritten.
//
// Concurrent calls of Replay on one plan run one at a time. Plans that
// share communicators and buffers may refuse each other's held buffers.
func (p *CollectivePlan) Replay(ctx context.Context, inputs [][][]float32) (*CollectiveReceipt, error) {
	if p.quarantine == nil {
		return nil, errors.New("egraph: collective plan not created by CollectiveSchedule.Plan")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil, fmt.Errorf("egraph: replay of closed plan: %w", cudacollective.ErrCommunicatorClosed)
	}
	if err := p.quarantine.Err(p.id); err != nil {
		return nil, err
	}
	if err := p.checkInputs(inputs); err != nil {
		return nil, err
	}
	for r, comm := range p.comms {
		if err := comm.CheckAlive(); err != nil {
			return nil, fmt.Errorf("egraph: rank %d communicator before replay: %w", r, err)
		}
	}
	rec := &CollectiveReceipt{PlanID: p.id, Transport: p.comms[0].Transport().Type()}
	for _, c := range p.calls[0] {
		rec.Estimated += c.Estimate
	}
	for i := range p.calls[0] {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("egraph: replay cancelled before collective %d: %w", i, err)
		}
		release, err := p.hold(i)
		if err != nil {
			return nil, err
		}
		ran, err := p.execute(ctx, i, rec)
		if err != nil {
			release()
			// Ranks that ran a collective their peers did not have
			// diverged; a failure recorded during execution is the plan's.
			if ran > 0 || p.runtimeFailed() {
				return nil, p.fail(err)
			}
			return nil, err
		}
		rec.Steps = append(rec.Steps, p.reduce(i, inputs[i]))
		release()
	}
	return rec, nil
}

func (p *CollectivePlan) checkInputs(inputs [][][]float32) error {
	if len(inputs) != len(p.calls[0]) {
		return fmt.Errorf("%w: %d input steps for %d collectives", ErrCollectiveMismatch, len(inputs), len(p.calls[0]))
	}
	for i, step := range inputs {
		c := p.calls[0][i]
		if collectiveDataTypes[strings.ToLower(c.DType)] != cudair.Float32 {
			return fmt.Errorf("%w: host replay of dtype %q", ErrCollectiveUnsupported, c.DType)
		}
		if len(step) != len(p.calls) {
			return fmt.Errorf("%w: collective %d has %d rank inputs, want %d", ErrCollectiveMismatch, i, len(step), len(p.calls))
		}
		for r, in := range step {
			if uint64(len(in)) != c.Count {
				return fmt.Errorf("%w: rank %d collective %d has %d elements, want %d",
					ErrCollectiveMismatch, r, i, len(in), c.Count)
			}
		}
	}
	return nil
}

// hold acquires every rank's buffers for collective i through its
// communicator and returns a function releasing them.
func (p *CollectivePlan) hold(i int) (release func(), err error) {
	var held []func()
	release = func() {
		for _, rel := range held {
			rel()
		}
	}
	for r, comm := range p.comms {
		c := p.calls[r][i]
		bufs := []CollectiveBuffer{c.Send}
		if !c.InPlace {
			bufs = append(bufs, c.Recv)
		}
		for _, b := range bufs {
			tr, err := comm.AcquireTransferBuffer(uintptr(b.Address+b.Offset), int(b.Bytes), 0)
			if err != nil {
				release()
				return nil, fmt.Errorf("egraph: rank %d effect %d: %w", r, c.EffectToken, err)
			}
			held = append(held, func() { comm.ReleaseTransferBuffer(tr, nil) })
		}
	}
	return release, nil
}

// execute runs collective i on each rank in rank order and reports how many
// ranks completed it.
func (p *CollectivePlan) execute(ctx context.Context, i int, rec *CollectiveReceipt) (ran int, err error) {
	for r, rp := range p.runtime[i] {
		start := p.now()
		if err := rp.ExecuteSteps(ctx); err != nil {
			return r, fmt.Errorf("egraph: rank %d effect %d: %w", r, p.calls[r][i].EffectToken, err)
		}
		rec.Observations = append(rec.Observations, CollectiveObservation{Step: i, Rank: r, Elapsed: p.now().Sub(start)})
	}
	return len(p.runtime[i]), nil
}

// reduce sums collective i's inputs and writes each rank's output.
func (p *CollectivePlan) reduce(i int, in [][]float32) CollectiveStepRecord {
	c0 := p.calls[0][i]
	sum := make([]float32, c0.Count)
	for j := range sum {
		var s float32
		for r := range in {
			s += in[r][j]
		}
		sum[j] = s
	}
	step := CollectiveStepRecord{EffectToken: c0.EffectToken}
	for r := range in {
		if p.calls[r][i].InPlace {
			copy(in[r], sum)
		}
		step.Outputs = append(step.Outputs, slices.Clone(sum))
		step.Completions = append(step.Completions, p.calls[r][i].Completion)
	}
	return step
}

// runtimeFailed reports whether the runtime recorded a communicator failure:
// a failed generation or a lost participant. A closed communicator is not one.
func (p *CollectivePlan) runtimeFailed() bool {
	for _, comm := range p.comms {
		err := comm.CheckAlive()
		if errors.Is(err, cudacollective.ErrCommunicatorGeneration) || errors.Is(err, cudacollective.ErrLostParticipant) {
			return true
		}
	}
	return false
}

// fail marks every communicator failed and quarantines the plan.
func (p *CollectivePlan) fail(err error) error {
	for _, comm := range p.comms {
		comm.MarkFailed(err)
	}
	p.quarantine.add(p.id, err)
	return err
}
