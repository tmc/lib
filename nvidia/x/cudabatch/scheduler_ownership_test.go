package cudabatch

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"
)

func ownershipRequest(id string) *Request {
	return &Request{ID: id, PromptTokens: []int32{1}, MaxTokens: 3}
}

func TestSchedulerOwnsRequests(t *testing.T) {
	s := NewBatchScheduler(SchedulerConfig{}, nil)
	r := ownershipRequest("a")
	if err := s.Submit(r); err != nil {
		t.Fatal(err)
	}
	r.ID = "changed"
	r.PromptTokens[0] = 99
	r.MaxTokens = 0
	step, err := s.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	if step.AllocatedRows[0].SeqID != "a" || !reflect.DeepEqual(step.AllocatedRows[0].Tokens, []int32{1}) {
		t.Fatal(step)
	}
	if err := s.CompleteStep(step, map[string]int32{"a": 7}, nil); err != nil {
		t.Fatal(err)
	}
	view, _ := s.Request("a")
	view.PromptTokens[0] = 44
	view.GeneratedTokens[0] = 55
	step, err = s.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(step.AllocatedRows[0].Tokens, []int32{7}) {
		t.Fatal("snapshot aliases state", step)
	}
}

func TestSchedulerAdmissionOrder(t *testing.T) {
	for _, policy := range []FairnessPolicy{PolicyFCFS, PolicyFairShare, PolicyPriorityDecode} {
		t.Run(fmt.Sprint(policy), func(t *testing.T) {
			s := NewBatchScheduler(SchedulerConfig{MaxBatchSize: 4, MaxBudgetTokens: 4, ChunkSize: 1, FairnessPolicy: policy}, nil)
			for _, id := range []string{"z", "a", "m"} {
				r := ownershipRequest(id)
				r.CreatedAt = time.Unix(1, 0)
				if err := s.Submit(r); err != nil {
					t.Fatal(err)
				}
			}
			for i := 0; i < 3; i++ {
				step, err := s.Schedule()
				if err != nil {
					t.Fatal(err)
				}
				var got []string
				out := map[string]int32{}
				for _, row := range step.AllocatedRows {
					got = append(got, row.SeqID)
					out[row.SeqID] = 7
				}
				if !reflect.DeepEqual(got, []string{"z", "a", "m"}) {
					t.Fatalf("order %v", got)
				}
				if err := s.CompleteStep(step, out, nil); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func TestSchedulerBoundsAndRetirement(t *testing.T) {
	s := NewBatchScheduler(SchedulerConfig{MaxWaitingRequests: 1, MaxFinishedRequests: 1}, nil)
	if err := s.Submit(ownershipRequest("old")); err != nil {
		t.Fatal(err)
	}
	if err := s.Submit(ownershipRequest("overflow")); !errors.Is(err, ErrCapacityExhausted) {
		t.Fatal(err)
	}
	if _, ok := s.Request("overflow"); ok {
		t.Fatal("overcapacity admitted")
	}
	if err := s.Cancel("old"); err != nil {
		t.Fatal(err)
	}
	if err := s.Submit(ownershipRequest("active")); err != nil {
		t.Fatal(err)
	}
	step, err := s.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Submit(ownershipRequest("waiting")); err != nil {
		t.Fatal(err)
	}
	if err := s.Cancel("waiting"); err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Request("old"); ok {
		t.Fatal("oldest terminal not retired")
	}
	released := 0
	s.RegisterResourceRelease("old", func() { released++ })
	if released != 1 {
		t.Fatal("retired ID retained a cleanup callback")
	}
	if _, ok := s.Request("active"); !ok {
		t.Fatal("active retired")
	}
	if _, err := s.Schedule(); !errors.Is(err, ErrExecutionInFlight) {
		t.Fatal(err)
	}
	if err := s.CompleteStep(&ScheduledIteration{StepID: step.StepID - 1}, nil, nil); !errors.Is(err, ErrInvalidRequest) {
		t.Fatal(err)
	}
	if err := s.Cancel("active"); err != nil {
		t.Fatal(err)
	}
	if err := s.CompleteStep(step, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := s.Submit(ownershipRequest("old")); err != nil {
		t.Fatal(err)
	}
	if s.requests["old"].admissionOrdinal <= s.requests["active"].admissionOrdinal {
		t.Fatal("admission generation reset")
	}
}

func TestSchedulerWaitingExpiryRetainsActive(t *testing.T) {
	s := NewBatchScheduler(SchedulerConfig{MaxBatchSize: 1, MaxWaitingTimeout: time.Second, MaxFinishedRequests: 1}, nil)
	var finished []*Request
	r := ownershipRequest("active")
	r.OnFinish = func(r *Request) { finished = append(finished, r) }
	if err := s.Submit(r); err != nil {
		t.Fatal(err)
	}
	step, err := s.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	w := ownershipRequest("waiting")
	w.OnFinish = func(r *Request) { finished = append(finished, r) }
	if err := s.Submit(w); err != nil {
		t.Fatal(err)
	}
	s.requests["waiting"].admittedAt = time.Now().Add(-2 * time.Second)
	if _, err := s.Schedule(); !errors.Is(err, ErrExecutionInFlight) {
		t.Fatal(err)
	}
	if len(finished) != 1 || finished[0].ID != "waiting" || !errors.Is(finished[0].Err(), ErrWaitingTimeout) {
		t.Fatal(finished)
	}
	if !s.requests["active"].activeExecution {
		t.Fatal("expiry released active state")
	}
	if err := s.Cancel("active"); err != nil {
		t.Fatal(err)
	}
	if len(finished) != 1 {
		t.Fatal("inflight callback ran early")
	}
	if err := s.CompleteStep(step, nil, nil); err != nil {
		t.Fatal(err)
	}
	if len(finished) != 2 || !finished[1].IsFinished() {
		t.Fatal(finished)
	}
}

func TestSchedulerAdmissionValidation(t *testing.T) {
	for _, edit := range []func(*Request){func(r *Request) { r.MaxTokens = 0 }, func(r *Request) { r.MaxTokens = -1 }, func(r *Request) { r.Visibility = 99 }, func(r *Request) { r.State = -1 }} {
		s := NewBatchScheduler(SchedulerConfig{}, nil)
		r := ownershipRequest("a")
		edit(r)
		if err := s.Submit(r); !errors.Is(err, ErrInvalidRequest) {
			t.Fatal(err)
		}
	}
	for _, cfg := range []SchedulerConfig{{MaxWaitingRequests: -1}, {MaxFinishedRequests: -1}, {MaxWaitingTimeout: -1}, {FairnessPolicy: 99}} {
		if err := NewBatchScheduler(cfg, nil).Submit(ownershipRequest("a")); !errors.Is(err, ErrInvalidRequest) {
			t.Fatal(err)
		}
	}
	s := NewBatchScheduler(SchedulerConfig{}, nil)
	if s.cfg.MaxWaitingRequests != 1024 || s.cfg.MaxFinishedRequests != 1024 {
		t.Fatal(s.cfg)
	}
}

func ExampleRequest_Err() {
	s := NewBatchScheduler(SchedulerConfig{}, nil)
	_ = s.Submit(&Request{ID: "a", PromptTokens: []int32{1}, MaxTokens: 1})
	_ = s.Cancel("a")
	r, _ := s.Request("a")
	fmt.Println(r.Err())
	// Output: context canceled
}
