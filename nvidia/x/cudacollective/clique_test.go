package cudacollective

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
)

func newTestClique(t *testing.T, world int) *Clique {
	t.Helper()
	ranks := make([]Rank, world)
	for i := range ranks {
		ranks[i] = Rank{ID: i, HostID: "h", DeviceID: i}
	}
	m, err := NewMembership(ranks)
	if err != nil {
		t.Fatal(err)
	}
	q, err := NewClique(m)
	if err != nil {
		t.Fatal(err)
	}
	return q
}

func newTestComm(t *testing.T, q *Clique, rank int, g Generation) *Communicator {
	t.Helper()
	tr := NewDeterministicTestTransport(q.membership.WorldSize())
	c, err := NewCommunicator(CommunicatorConfig{Rank: rank, Clique: q, Generation: g, Transport: tr})
	if err != nil {
		t.Fatalf("rank %d generation %d: %v", rank, g, err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

// refused checks that c refuses both admission paths with ErrCommunicatorGeneration.
func refused(t *testing.T, c *Communicator) {
	t.Helper()
	if err := c.CheckAlive(); !errors.Is(err, ErrCommunicatorGeneration) {
		t.Fatalf("rank %d generation %d CheckAlive: %v, want ErrCommunicatorGeneration", c.Rank(), c.Generation(), err)
	}
	if tr, err := c.AcquireTransferBuffer(0x1000, 8, 0); !errors.Is(err, ErrCommunicatorGeneration) {
		c.ReleaseTransferBuffer(tr, nil)
		t.Fatalf("rank %d generation %d AcquireTransferBuffer: %v, want ErrCommunicatorGeneration", c.Rank(), c.Generation(), err)
	}
}

// admitted checks that c accepts both admission paths.
func admitted(t *testing.T, c *Communicator) {
	t.Helper()
	if err := c.CheckAlive(); err != nil {
		t.Fatalf("rank %d generation %d CheckAlive: %v", c.Rank(), c.Generation(), err)
	}
	tr, err := c.AcquireTransferBuffer(0x1000, 8, 0)
	if err != nil {
		t.Fatalf("rank %d generation %d AcquireTransferBuffer: %v", c.Rank(), c.Generation(), err)
	}
	c.ReleaseTransferBuffer(tr, nil)
}

func TestCliqueFailureRefusesAllRanks(t *testing.T) {
	cause := errors.New("link down")
	for _, tt := range []struct {
		name string
		fail func(c0, c1 *Communicator)
	}{
		{"one rank", func(c0, c1 *Communicator) { c0.MarkFailed(cause) }},
		{"both ranks", func(c0, c1 *Communicator) { c0.MarkFailed(cause); c1.MarkFailed(errors.New("peer")) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			q := newTestClique(t, 2)
			c0, c1 := newTestComm(t, q, 0, 1), newTestComm(t, q, 1, 1)
			tt.fail(c0, c1)
			refused(t, c0)
			refused(t, c1)
			// The reporter keeps its own cause; a sibling never receives it.
			if err := c0.CheckAlive(); !errors.Is(err, cause) {
				t.Fatalf("reporter lost cause: %v", err)
			}
			if _, err := c0.AcquireTransferBuffer(0x1000, 8, 0); !errors.Is(err, cause) {
				t.Fatalf("reporter acquire lost cause: %v", err)
			}
			if err := c1.CheckAlive(); errors.Is(err, cause) {
				t.Fatalf("sibling received foreign cause: %v", err)
			}
		})
	}
}

func TestCliqueRefusesReconstruction(t *testing.T) {
	q := newTestClique(t, 2)
	newTestComm(t, q, 0, 1)
	c := newTestComm(t, q, 0, 2)
	c.MarkFailed(errors.New("abort"))
	for _, g := range []Generation{1, 2} {
		for rank := range 2 {
			fresh := NewDeterministicTestTransport(2)
			_, err := NewCommunicator(CommunicatorConfig{Rank: rank, Clique: q, Generation: g, Transport: fresh})
			if !errors.Is(err, ErrCommunicatorGeneration) {
				t.Fatalf("rank %d generation %d with fresh transport: %v, want ErrCommunicatorGeneration", rank, g, err)
			}
			if fresh.closed {
				t.Fatal("refused constructor closed caller transport")
			}
		}
	}
}

func TestCliqueUnrelatedIndependent(t *testing.T) {
	a, b := newTestClique(t, 2), newTestClique(t, 2)
	ca, cb := newTestComm(t, a, 0, 1), newTestComm(t, b, 0, 1)
	if !slices.Equal(ca.Membership().Ranks(), cb.Membership().Ranks()) {
		t.Fatal("fixture memberships differ")
	}
	ca.MarkFailed(errors.New("abort"))
	refused(t, ca)
	admitted(t, cb)
	admitted(t, newTestComm(t, b, 1, 1))
}

func TestCliqueHigherGenerationAdmits(t *testing.T) {
	q := newTestClique(t, 2)
	old := newTestComm(t, q, 0, 3)
	old.ReportLostParticipant(1)
	next := newTestComm(t, q, 0, 4)
	admitted(t, next)
	admitted(t, newTestComm(t, q, 1, 4))
	refused(t, old)
	for _, g := range []Generation{1, 3} {
		_, err := NewCommunicator(CommunicatorConfig{Rank: 1, Clique: q, Generation: g, Transport: NewDeterministicTestTransport(2)})
		if !errors.Is(err, ErrCommunicatorGeneration) {
			t.Fatalf("generation %d after admitting 4: %v", g, err)
		}
	}
	if next.Generation() != 4 {
		t.Fatalf("generation advanced to %d", next.Generation())
	}
}

func TestCliqueDelayedOlderFailure(t *testing.T) {
	q := newTestClique(t, 2)
	old := newTestComm(t, q, 0, 1)
	next := newTestComm(t, q, 0, 2)
	old.MarkFailed(errors.New("late generation 1 failure"))
	refused(t, old)
	admitted(t, next)
	admitted(t, newTestComm(t, q, 1, 2))
	if _, err := NewCommunicator(CommunicatorConfig{Rank: 1, Clique: q, Generation: 1, Transport: NewDeterministicTestTransport(2)}); !errors.Is(err, ErrCommunicatorGeneration) {
		t.Fatalf("generation 1 after its failure: %v", err)
	}
	// A late failure of an older generation never lowers the floor.
	high := newTestComm(t, q, 0, 4)
	high.MarkFailed(errors.New("generation 4 failure"))
	old.MarkFailed(errors.New("later generation 1 failure"))
	for _, g := range []Generation{2, 3, 4} {
		if _, err := NewCommunicator(CommunicatorConfig{Rank: 1, Clique: q, Generation: g, Transport: NewDeterministicTestTransport(2)}); !errors.Is(err, ErrCommunicatorGeneration) {
			t.Fatalf("generation %d after a late older failure: %v", g, err)
		}
	}
	refused(t, next)
	admitted(t, newTestComm(t, q, 1, 5))
}

func TestCliqueMaxGeneration(t *testing.T) {
	const top = ^Generation(0)
	q := newTestClique(t, 1)
	c := newTestComm(t, q, 0, top)
	admitted(t, c)
	c.MarkFailed(errors.New("abort"))
	for _, g := range []Generation{1, top - 1, top} {
		_, err := NewCommunicator(CommunicatorConfig{Rank: 0, Clique: q, Generation: g, Transport: NewDeterministicTestTransport(1)})
		if !errors.Is(err, ErrCommunicatorGeneration) || !strings.Contains(err.Error(), "exhausted") {
			t.Fatalf("generation %d after failing max: %v", g, err)
		}
	}
}

func TestCliqueInvalidOwner(t *testing.T) {
	tr := NewDeterministicTestTransport(1)
	if _, err := NewClique(Membership{}); !errors.Is(err, ErrInvalidWorldSize) {
		t.Fatalf("NewClique(Membership{}): %v", err)
	}
	if _, err := NewCommunicator(CommunicatorConfig{Rank: 0, Generation: 1, Transport: tr}); err == nil {
		t.Fatal("nil clique admitted")
	}
	if _, err := NewCommunicator(CommunicatorConfig{Rank: 0, Clique: new(Clique), Generation: 1, Transport: tr}); !errors.Is(err, ErrInvalidWorldSize) {
		t.Fatalf("zero clique: %v", err)
	}
	q := newTestClique(t, 1)
	if _, err := NewCommunicator(CommunicatorConfig{Rank: 0, Clique: q, Generation: 0, Transport: tr}); !errors.Is(err, ErrCommunicatorGeneration) {
		t.Fatalf("generation 0: %v", err)
	}
	// Refusals record no failure.
	admitted(t, newTestComm(t, q, 0, 1))
}

func TestCliqueNilCause(t *testing.T) {
	q := newTestClique(t, 2)
	c0, c1 := newTestComm(t, q, 0, 1), newTestComm(t, q, 1, 1)
	c0.MarkFailed(nil)
	refused(t, c0)
	refused(t, c1)
	if err := c0.CheckAlive(); !errors.Is(err, errUnspecifiedFailure) {
		t.Fatalf("nil cause: %v", err)
	}
}

func TestCliqueLostParticipantStillCloses(t *testing.T) {
	q := newTestClique(t, 2)
	c0, c1 := newTestComm(t, q, 0, 1), newTestComm(t, q, 1, 1)
	held, err := c1.AcquireTransferBuffer(0x2000, 8, 0)
	if err != nil {
		t.Fatal(err)
	}
	c0.ReportLostParticipant(1)
	if err := c0.CheckAlive(); !errors.Is(err, ErrLostParticipant) || !errors.Is(err, ErrCommunicatorGeneration) {
		t.Fatalf("reporter: %v", err)
	}
	refused(t, c1)
	select {
	case <-held.Done:
		t.Fatal("failure released an admitted reservation")
	default:
	}
	if err := c1.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-held.Done:
	default:
		t.Fatal("close did not retire reservation")
	}
	if !errors.Is(held.Err, ErrCommunicatorClosed) {
		t.Fatalf("retired reservation: %v", held.Err)
	}
	if err := c1.Close(); err != nil {
		t.Fatal(err)
	}
	if err := c0.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestCliqueExplicitClosedFailurePublishes checks that MarkFailed publishes
// whatever its cause.
func TestCliqueExplicitClosedFailurePublishes(t *testing.T) {
	q := newTestClique(t, 2)
	c0, c1 := newTestComm(t, q, 0, 1), newTestComm(t, q, 1, 1)
	c0.MarkFailed(ErrCommunicatorClosed)
	refused(t, c0)
	refused(t, c1)
}

// cliqueResultTransport returns err from every Execute.
type cliqueResultTransport struct{ err error }

func (*cliqueResultTransport) Type() TransportType       { return TransportTypeControlPlaneTest }
func (*cliqueResultTransport) Available(Membership) bool { return true }
func (x *cliqueResultTransport) Execute(context.Context, *Communicator, OperationDescriptor, driver.DevicePtr, driver.DevicePtr) error {
	return x.err
}
func (*cliqueResultTransport) Close() error { return nil }

// TestCliquePlanLocalClose checks that a plan step failing only because a
// communicator closed does not fail the Clique, while any other failure
// joined with the close does.
func TestCliquePlanLocalClose(t *testing.T) {
	linkDown := errors.New("link down")
	for _, tt := range []struct {
		name    string
		err     error
		publish bool
	}{
		{"closed", ErrCommunicatorClosed, false},
		{"wrapped closed", fmt.Errorf("transport: %w", ErrCommunicatorClosed), false},
		{"closed and failure", errors.Join(ErrCommunicatorClosed, linkDown), true},
		{"wrapped closed and failure", fmt.Errorf("transport: %w", errors.Join(linkDown, ErrCommunicatorClosed)), true},
		{"closed with two %w", fmt.Errorf("%w: %w", ErrCommunicatorClosed, linkDown), true},
	} {
		for _, pipelined := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/pipelined=%v", tt.name, pipelined), func(t *testing.T) {
				q := newTestClique(t, 2)
				x := &cliqueResultTransport{err: tt.err}
				c0, err := NewCommunicator(CommunicatorConfig{Rank: 0, Clique: q, Generation: 1, Transport: x})
				if err != nil {
					t.Fatal(err)
				}
				defer c0.Close()
				c1 := newTestComm(t, q, 1, 1)
				var pipe *PipelineManager
				if pipelined {
					if pipe, err = NewPipelineManager(c0, DoubleBuffer{Slots: [2]driver.DevicePtr{0x1000, 0x2000}, Bytes: 8}, 2); err != nil {
						t.Fatal(err)
					}
				}
				plan, err := NewPlan(c0, pipe)
				if err != nil {
					t.Fatal(err)
				}
				if err := plan.AddCollectiveStep("step", OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}, 0x1000, 0x1004); err != nil {
					t.Fatal(err)
				}
				err = plan.ExecuteSteps(context.Background())
				if !errors.Is(err, ErrCommunicatorClosed) || tt.publish && !errors.Is(err, linkDown) {
					t.Fatalf("ExecuteSteps: %v, want %v", err, tt.err)
				}
				if tt.publish {
					refused(t, c0)
					refused(t, c1)
				} else {
					admitted(t, c0)
					admitted(t, c1)
				}
				if err := plan.Close(); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

// TestCliqueConstructorBeforeFailure checks that a communicator admitted
// before a failure of its generation refuses afterward.
func TestCliqueConstructorBeforeFailure(t *testing.T) {
	q := newTestClique(t, 2)
	c0 := newTestComm(t, q, 0, 1)
	c1 := newTestComm(t, q, 1, 1)
	admitted(t, c1)
	c0.MarkFailed(errors.New("abort"))
	refused(t, c1)
}

// TestCliqueConstructorDuringFailure stages a constructor behind an
// in-progress publication and checks that it is refused.
func TestCliqueConstructorDuringFailure(t *testing.T) {
	q := newTestClique(t, 2)
	c0 := newTestComm(t, q, 0, 1)
	c0.mu.Lock() // stall MarkFailed after it publishes
	marked := make(chan struct{})
	go func() { c0.MarkFailed(errors.New("abort")); close(marked) }()
	for deadline := time.Now().Add(10 * time.Second); q.mu.TryRLock(); runtime.Gosched() {
		q.mu.RUnlock()
		if time.Now().After(deadline) {
			c0.mu.Unlock()
			t.Fatal("MarkFailed never took the Clique lock")
		}
	}
	built := make(chan error, 1)
	go func() {
		_, err := NewCommunicator(CommunicatorConfig{Rank: 1, Clique: q, Generation: 1, Transport: NewDeterministicTestTransport(2)})
		built <- err
	}()
	select {
	case err := <-built:
		c0.mu.Unlock()
		t.Fatalf("constructor completed during publication: %v", err)
	default:
	}
	c0.mu.Unlock()
	<-marked
	if err := <-built; !errors.Is(err, ErrCommunicatorGeneration) {
		t.Fatalf("constructor during publication: %v", err)
	}
}

// TestCliqueReservationOrdersWithFailure stages a reservation that holds
// Clique admission and checks that publication waits for it, then refuses
// every later reservation. The admitted reservation remains owned.
func TestCliqueReservationOrdersWithFailure(t *testing.T) {
	q := newTestClique(t, 2)
	c0, c1 := newTestComm(t, q, 0, 1), newTestComm(t, q, 1, 1)
	c1.inFlightMu.Lock() // stall the reservation before insertion
	acquired := make(chan error, 1)
	var held *InFlightTransfer
	go func() {
		var err error
		held, err = c1.AcquireTransferBuffer(0x1000, 8, 0)
		acquired <- err
	}()
	for deadline := time.Now().Add(10 * time.Second); q.mu.TryLock(); runtime.Gosched() {
		q.mu.Unlock()
		if time.Now().After(deadline) {
			c1.inFlightMu.Unlock()
			t.Fatal("reservation never held Clique admission")
		}
	}
	marked := make(chan struct{})
	go func() { c0.MarkFailed(errors.New("abort")); close(marked) }()
	select {
	case <-marked:
		c1.inFlightMu.Unlock()
		t.Fatal("failure published while a reservation held admission")
	default:
	}
	c1.inFlightMu.Unlock()
	if err := <-acquired; err != nil {
		t.Fatalf("reservation admitted before publication: %v", err)
	}
	<-marked
	refused(t, c1)
	select {
	case <-held.Done:
		t.Fatal("publication released admitted reservation")
	default:
	}
	c1.ReleaseTransferBuffer(held, nil)
}

// ExampleNewClique shares one Clique between two ranks. A failure on one
// rank refuses its sibling and any reconstruction at that generation; a
// deliberately larger generation is admitted.
func ExampleNewClique() {
	m, _ := NewMembership([]Rank{{ID: 0}, {ID: 1, DeviceID: 1}})
	clique, _ := NewClique(m)
	tr := NewDeterministicTestTransport(2)
	rank0, _ := NewCommunicator(CommunicatorConfig{Rank: 0, Clique: clique, Generation: 1, Transport: tr})
	rank1, _ := NewCommunicator(CommunicatorConfig{Rank: 1, Clique: clique, Generation: 1, Transport: tr})

	rank0.MarkFailed(errors.New("link down"))
	fmt.Println(errors.Is(rank1.CheckAlive(), ErrCommunicatorGeneration))

	_, err := NewCommunicator(CommunicatorConfig{Rank: 1, Clique: clique, Generation: 1, Transport: NewDeterministicTestTransport(2)})
	fmt.Println(errors.Is(err, ErrCommunicatorGeneration))

	next, err := NewCommunicator(CommunicatorConfig{Rank: 1, Clique: clique, Generation: 2, Transport: NewDeterministicTestTransport(2)})
	fmt.Println(err, next.CheckAlive())
	// Output:
	// true
	// true
	// <nil> <nil>
}

// ExampleClique shows that Cliques with equal memberships are independent.
func ExampleClique() {
	m, _ := NewMembership([]Rank{{ID: 0}})
	a, _ := NewClique(m)
	b, _ := NewClique(m)
	ca, _ := NewCommunicator(CommunicatorConfig{Rank: 0, Clique: a, Generation: 1, Transport: NewDeterministicTestTransport(1)})
	cb, _ := NewCommunicator(CommunicatorConfig{Rank: 0, Clique: b, Generation: 1, Transport: NewDeterministicTestTransport(1)})
	ca.MarkFailed(errors.New("abort"))
	fmt.Println(ca.CheckAlive() != nil, cb.CheckAlive())
	// Output:
	// true <nil>
}
