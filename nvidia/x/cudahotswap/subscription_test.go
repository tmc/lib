package cudahotswap

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func TestServiceSubscriptionsDropResumeAndStamp(t *testing.T) {
	s, leases := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	registerService(t, s)
	state, err := s.SetCompileFlags("kernel", nil, "settings snapshot")
	if err != nil {
		t.Fatal(err)
	}
	state, err = s.SetGeometry("kernel", nil, "settings snapshot")
	if err != nil || state.ConfigRevision == state.Generation {
		t.Fatal("control needs distinct config and generation revisions")
	}
	sub, err := s.Subscribe(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	report := &Report{Outcome: OutcomeWrongAnswer, Generation: 999, ConfigRevision: 999, Reason: "actual producer refusal"}
	for i := 1; i <= 66; i++ {
		seq, err := s.publishReport(state, report)
		if err != nil || seq != uint64(i) {
			t.Fatalf("report %d: %d, %v", i, seq, err)
		}
	}
	if report.Generation != 999 || report.ConfigRevision != 999 || !errors.Is(sub.Err(), ErrSlowSubscriber) {
		t.Fatal("publication changed caller report or concealed subscriber drop")
	}
	var last uint64
	for e := range sub.Events() {
		last++
		if e.Sequence != last || e.Report.Generation != state.Generation || e.Report.ConfigRevision != state.ConfigRevision || e.Generation != state.Generation || e.ConfigRevision != state.ConfigRevision || e.Label != state.Label || e.Report.Outcome != OutcomeWrongAnswer {
			t.Fatalf("unbound or missing report: %+v", e)
		}
		e.Report.Reason = "caller mutation"
	}
	if last != 65 {
		t.Fatalf("queued reports = %d", last)
	}
	resumed, err := s.Subscribe(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	gap := <-resumed.Events()
	if gap.Sequence != 2 || gap.Report != nil || gap.Gap == nil || *gap.Gap != (Gap{First: 1, Last: 2}) {
		t.Fatalf("missing explicit lost range: %+v", gap)
	}
	for seq := uint64(3); seq <= 66; seq++ {
		e := <-resumed.Events()
		if e.Sequence != seq || e.Gap != nil || e.Report.Reason != "actual producer refusal" {
			t.Fatalf("retained report aliases subscriber or has wrong cursor: %+v", e)
		}
	}
	if len(*leases) != 1 || (*leases)[0].calls != 1 {
		t.Fatal("report subscription or publication acquires a GPU lease")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	assertSubscriptionClosed(t, resumed, io.ErrClosedPipe)
}

func TestServiceSubscriptionOwnershipAndRefusal(t *testing.T) {
	s, leases := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	if _, err := s.Subscribe(nil, 0); err == nil {
		t.Fatal("nil context admitted")
	}
	if _, err := s.Subscribe(context.Background(), 1); err == nil {
		t.Fatal("future cursor admitted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	sub, err := s.Subscribe(ctx, 0)
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case _, ok := <-sub.Events():
		if ok || !errors.Is(sub.Err(), context.Canceled) {
			t.Fatal("cancellation did not preserve its cause")
		}
	case <-time.After(time.Second):
		t.Fatal("canceled subscription remains open")
	}
	var subscriptions []*Subscription
	for i := 0; i < 32; i++ {
		sub, err := s.Subscribe(context.Background(), 0)
		if err != nil {
			t.Fatal(err)
		}
		subscriptions = append(subscriptions, sub)
	}
	if _, err := s.Subscribe(context.Background(), 0); err == nil {
		t.Fatal("subscriber limit ignored")
	}
	if err := subscriptions[0].Close(); err != nil || subscriptions[0].Err() != nil {
		t.Fatal("explicit subscription close failed")
	}
	if err := subscriptions[0].Close(); err != nil {
		t.Fatal(err)
	}
	replacement, err := s.Subscribe(context.Background(), 0)
	if err != nil {
		t.Fatal("explicit close did not release subscriber slot")
	}
	if err := replacement.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Subscribe(context.Background(), 0); err == nil || len(*leases) != 0 {
		t.Fatal("idle subscription closes with GPU work or admits after shutdown")
	}
	for _, sub := range []*Subscription{nil, new(Subscription)} {
		if _, ok := <-sub.Events(); ok || sub.Err() != nil || sub.Close() != nil {
			t.Fatal("zero subscription is not closed")
		}
	}
}

func TestServiceSubscriptionFailedShutdownAndPoison(t *testing.T) {
	k := &serviceKernel{raw: 1, waitErr: context.Canceled}
	s, _ := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	sub, err := s.Subscribe(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Launch(context.Background(), "kernel", cudalaunch.Config{}); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(context.Background()); !errors.Is(err, context.Canceled) {
		t.Fatalf("drain error concealed: %v", err)
	}
	assertSubscriptionClosed(t, sub, io.ErrClosedPipe)
	if k.calls != 0 {
		t.Fatal("failed shutdown unloads incomplete work")
	}
	state, _ := s.Status("kernel")
	if _, err := s.publishReport(state, &Report{Outcome: OutcomeOK}); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatal("publication admitted after shutdown begins")
	}
	k.waitErr = nil
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}

	p, _ := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	registerService(t, p)
	state, _ = p.Status("kernel")
	p.mu.Lock()
	p.poison = errors.New("terminal fault")
	p.mu.Unlock()
	if _, err := p.Subscribe(context.Background(), 0); err == nil {
		t.Fatal("poisoned service admitted new subscriber")
	}
	if _, err := p.publishReport(state, &Report{Outcome: OutcomeHarnessError, ContextPoisoned: true}); err != nil {
		t.Fatal("terminal fault report could not be retained")
	}
	if err := p.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func assertSubscriptionClosed(t *testing.T, sub *Subscription, cause error) {
	t.Helper()
	select {
	case _, open := <-sub.Events():
		if open || !errors.Is(sub.Err(), cause) {
			t.Fatalf("subscription remains open or has wrong cause: %v", sub.Err())
		}
	default:
		t.Fatal("subscription remains open after synchronous shutdown")
	}
}

func TestServiceReportBudgetAndConcurrentShutdown(t *testing.T) {
	s, _ := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	registerService(t, s)
	state, _ := s.Status("kernel")
	if seq, err := s.publishReport(state, &Report{Reason: strings.Repeat("x", 64<<10)}); seq != 0 || err == nil || s.reports.sequence != 0 {
		t.Fatal("oversized report consumed sequence or retention")
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 8; j++ {
				sub, err := s.Subscribe(context.Background(), 0)
				if err == nil {
					_ = sub.Close()
				}
				_, _ = s.publishReport(state, &Report{Outcome: OutcomeOK})
			}
		}()
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
	s.reports.mu.Lock()
	defer s.reports.mu.Unlock()
	if len(s.reports.subs) != 0 {
		t.Fatal("shutdown left report subscriber owners")
	}
}
