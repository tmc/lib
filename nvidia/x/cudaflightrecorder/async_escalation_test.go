package cudaflightrecorder

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestAsyncEscalationBoundsAndStopsAllJobs(t *testing.T) {
	rec, err := New(Config{Capacity: 10, MaxQueuedJobs: 1})
	if err != nil {
		t.Fatal(err)
	}
	budgets := QualificationBudgets()
	budgets.HysteresisCount = 1
	sampler := &blockingUsage{entered: make(chan struct{}), release: make(chan struct{})}
	coord, err := NewEscalationCoordinator(rec, budgets, WithResourceSampler(sampler))
	if err != nil {
		t.Fatal(err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatal(err)
	}
	first, err := coord.TriggerAsync(context.Background(), TriggerRequest{Reason: "first"})
	if err != nil {
		t.Fatal(err)
	}
	<-sampler.entered
	details := map[string]string{"source": "original"}
	second, err := coord.TriggerAsync(context.Background(), TriggerRequest{Reason: "second", Details: details})
	if err != nil {
		t.Fatal(err)
	}
	details["source"] = "caller changed"
	if _, err := coord.TriggerAsync(context.Background(), TriggerRequest{}); !errors.Is(err, ErrEscalationQueueFull) {
		t.Fatalf("queue err=%v", err)
	}
	if err := coord.Stop(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := coord.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	firstResult := <-first
	if !errors.Is(firstResult.Err, context.Canceled) {
		t.Fatalf("first=%+v", firstResult)
	}
	if _, open := <-first; open {
		t.Fatal("first channel not closed")
	}
	secondResult := <-second
	if !errors.Is(secondResult.Err, ErrEscalationRefused) || secondResult.Receipt.RefusalReason != RefusalStopped || secondResult.Receipt.Trigger.Details["source"] != "original" {
		t.Fatalf("second=%+v", secondResult)
	}
	if _, open := <-second; open {
		t.Fatal("second channel not closed")
	}
	if _, err := coord.TriggerAsync(context.Background(), TriggerRequest{}); !errors.Is(err, ErrEscalationRefused) {
		t.Fatalf("post-stop=%v", err)
	}
	if coord.State() != StateStopped {
		t.Fatalf("state=%s", coord.State())
	}
}

func TestAsyncEscalationUnreadResultDoesNotBlockShutdown(t *testing.T) {
	rec, err := New(Config{Capacity: 10})
	if err != nil {
		t.Fatal(err)
	}
	coord, err := NewEscalationCoordinator(rec, QualificationBudgets(), WithResourceSampler(&FakeResourceSampler{}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := coord.TriggerAsync(context.Background(), TriggerRequest{}); err != nil {
		t.Fatal(err)
	}
	if err := coord.Stop(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := coord.Wait(ctx); err != nil {
		t.Fatal(err)
	}
}

func TestEscalationCopiesHistoryMetadata(t *testing.T) {
	rec, err := New(Config{Capacity: 10})
	if err != nil {
		t.Fatal(err)
	}
	coord, err := NewEscalationCoordinator(rec, QualificationBudgets())
	if err != nil {
		t.Fatal(err)
	}
	defer coord.Stop()
	receipt, _ := coord.Trigger(context.Background(), TriggerRequest{Details: map[string]string{"source": "original"}})
	receipt.Trigger.Details["source"] = "changed receipt"
	history := coord.History()
	if history[0].Trigger.Details["source"] != "original" {
		t.Fatal("receipt mutated history")
	}
	history[0].Trigger.Details["source"] = "changed history copy"
	if coord.History()[0].Trigger.Details["source"] != "original" {
		t.Fatal("history copy mutated coordinator")
	}
}

func TestEscalationMetadataAdmissionBounds(t *testing.T) {
	rec, err := New(Config{Capacity: 10})
	if err != nil {
		t.Fatal(err)
	}
	coord, err := NewEscalationCoordinator(rec, QualificationBudgets())
	if err != nil {
		t.Fatal(err)
	}
	defer coord.Stop()
	req := TriggerRequest{TenantID: strings.Repeat("t", 1025)}
	if _, err := coord.TriggerAsync(context.Background(), req); err == nil {
		t.Fatal("oversized queued metadata admitted")
	}
	if _, err := coord.Trigger(context.Background(), req); err == nil {
		t.Fatal("oversized direct metadata admitted")
	}
	if len(coord.History()) != 0 {
		t.Fatal("oversized metadata retained")
	}
}
