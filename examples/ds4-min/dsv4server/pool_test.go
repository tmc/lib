package dsv4server

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestSessionPoolUsesIndependentSlots(t *testing.T) {
	p, err := newSessionPool(testModel{}, 2)
	if err != nil {
		t.Fatal(err)
	}
	a, err := p.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	b, err := p.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if a.session == b.session {
		t.Fatal("pool handed out the same session twice")
	}
	a.Release()
	b.Release()
}

func TestSessionPoolFIFO(t *testing.T) {
	p, err := newSessionPool(testModel{}, 1)
	if err != nil {
		t.Fatal(err)
	}
	first, err := p.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	type result struct {
		lease *sessionLease
		err   error
	}
	one := make(chan result, 1)
	two := make(chan result, 1)
	go func() {
		lease, err := p.acquire(context.Background())
		one <- result{lease, err}
	}()
	waitForWaiters(t, p, 1)
	go func() {
		lease, err := p.acquire(context.Background())
		two <- result{lease, err}
	}()
	waitForWaiters(t, p, 2)

	first.Release()
	oneResult := <-one
	if oneResult.err != nil {
		t.Fatal(oneResult.err)
	}
	select {
	case <-two:
		t.Fatal("second waiter bypassed FIFO order")
	default:
	}
	oneResult.lease.Release()
	twoResult := <-two
	if twoResult.err != nil {
		t.Fatal(twoResult.err)
	}
	twoResult.lease.Release()
}

func TestSessionPoolAcquireHonorsCancellation(t *testing.T) {
	p, err := newSessionPool(testModel{}, 1)
	if err != nil {
		t.Fatal(err)
	}
	lease, err := p.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.acquire(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("acquire error = %v, want context canceled", err)
	}
	lease.Release()
}

func TestSessionPoolRetainsTimelineForPrefixReuse(t *testing.T) {
	p, err := newSessionPool(testModel{}, 1)
	if err != nil {
		t.Fatal(err)
	}
	first, err := p.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := first.session.Sync([]int{1, 2}); err != nil {
		t.Fatal(err)
	}
	first.Release()
	second, err := p.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer second.Release()
	if got := second.session.Tokens(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("reused session tokens = %v, want [1 2]", got)
	}
}

func TestNewRejectsUnprovenMultipleSessions(t *testing.T) {
	if _, err := New(struct{ Model }{Model: testModelWithoutIndependent{}}, Config{SessionSlots: 2}); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("New error = %v, want unsupported", err)
	}
}

type testModelWithoutIndependent struct{}

func (testModelWithoutIndependent) Render([]Message) ([]int, error) { return []int{1}, nil }
func (testModelWithoutIndependent) Decode([]int) (string, error)    { return "", nil }
func (testModelWithoutIndependent) NewSession() (*dsv4.Session, error) {
	return dsv4.NewSession(testBackend{}, 16, dsv4.NewStopSet(9))
}

func waitForWaiters(t *testing.T, p *sessionPool, want int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		p.mu.Lock()
		got := len(p.waiters)
		p.mu.Unlock()
		if got == want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("waiter count did not reach %d", want)
}
