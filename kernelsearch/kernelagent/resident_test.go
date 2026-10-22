package kernelagent

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudasearch"
)

func hostPlan(t *testing.T, opts ...cudaplan.PlanOption) *cudaplan.ExecutablePlan {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	y := g.AddUnary("relu", x)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	p, err := cudaplan.Compile(s, opts...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { p.Close() })
	return p
}

func TestPromotionReplacesPlanForNextLaunchOnly(t *testing.T) {
	f, l := setup(t)
	result := run(t, l, f)
	oldPlan, newPlan := hostPlan(t), hostPlan(t)
	r, err := NewResident(HostStub, 1, PlanBinding{f.cfg.Baseline.Identity(), oldPlan})
	if err != nil {
		t.Fatal(err)
	}
	alreadyBorrowed, gen := r.Next()
	if gen != 1 {
		t.Fatal("wrong generation")
	}
	if err := r.Promote(result.Promotion, PlanBinding{result.Search.Winner.ID, newPlan}); err != nil {
		t.Fatal(err)
	}
	next, gen := r.Next()
	if next.Plan != newPlan || alreadyBorrowed.Plan != oldPlan || gen != 2 {
		t.Fatal("existing borrowed plan changed")
	}
	if err := r.Promote(result.Promotion, PlanBinding{result.Search.Winner.ID, newPlan}); !errors.Is(err, cudasearch.ErrStaleGeneration) {
		t.Fatal("stale switch accepted")
	}
}

func TestExecutingGraphMutationNotAttempted(t *testing.T) {
	f, l := setup(t)
	result := run(t, l, f)
	p, q := hostPlan(t, cudaplan.WithFiniteMega(true)), hostPlan(t, cudaplan.WithFiniteMega(true))
	before := p.Steps()
	r, _ := NewResident(HostStub, 1, PlanBinding{f.cfg.Baseline.Identity(), p})
	if err := r.Promote(result.Promotion, PlanBinding{result.Search.Winner.ID, q}); err != nil {
		t.Fatal(err)
	}
	after := p.Steps()
	if len(before) == 0 || len(before) != len(after) {
		t.Fatal("previous plan mutated")
	}
	for i := range before {
		if before[i].Artifact == nil || after[i].Artifact == nil {
			t.Fatal("generated plan has no artifact")
		}
		if before[i].Artifact == after[i].Artifact {
			t.Fatal("plan readback did not detach artifact wrappers")
		}
		if before[i].NodeID != after[i].NodeID || before[i].NodeName != after[i].NodeName || *before[i].Artifact != *after[i].Artifact {
			t.Fatal("graph node changed")
		}
	}
	// This exercises the actual owner refusal, not a successful host execution.
	if comp, err := r.Execute(nil, cudaplan.NewBindings()); err == nil || comp != nil {
		t.Fatal("host fixture pretended GPU execution")
	}
}

func TestRegressionUnderRemeasurementRollsBack(t *testing.T) {
	f, l := setup(t)
	result := run(t, l, f)
	p, q := hostPlan(t), hostPlan(t)
	r, _ := NewResident(HostStub, 1, PlanBinding{f.cfg.Baseline.Identity(), p})
	if err := r.Promote(result.Promotion, PlanBinding{result.Search.Winner.ID, q}); err != nil {
		t.Fatal(err)
	}
	f.fastDuration = 180 * time.Nanosecond
	candidate, err := l.Remeasure(context.Background(), "remeasure", r, result)
	if err != nil || candidate == nil || improves(candidate) {
		t.Fatalf("remeasure=%+v %v", candidate, err)
	}
	defer candidate.Close()
	next, generation := r.Next()
	if next.Plan != p || generation != 3 {
		t.Fatal("regression did not restore previous next plan")
	}
	last := f.archive.Records()
	if last[len(last)-1].Kind != "rollback" {
		t.Fatal("rollback not durable")
	}
}

func TestNativeUnverifiedPlanAssociationRefused(t *testing.T) {
	f, l := setup(t)
	result := run(t, l, f)
	p, q := hostPlan(t), hostPlan(t)
	r, _ := NewResident(Native, 1, PlanBinding{f.cfg.Baseline.Identity(), p})
	if err := r.Promote(result.Promotion, PlanBinding{result.Search.Winner.ID, q}); !errors.Is(err, ErrPlanAssociationUnavailable) {
		t.Fatalf("unverified association accepted: %v", err)
	}
	next, generation := r.Next()
	if next.Plan != p || generation != 1 {
		t.Fatal("refused association changed resident")
	}
}

func TestRegressionRollsBackEvenWhenArchiveFull(t *testing.T) {
	f, l := setup(t)
	result := run(t, l, f)
	p, q := hostPlan(t), hostPlan(t)
	r, _ := NewResident(HostStub, 1, PlanBinding{f.cfg.Baseline.Identity(), p})
	if err := r.Promote(result.Promotion, PlanBinding{result.Search.Winner.ID, q}); err != nil {
		t.Fatal(err)
	}
	f.fastDuration = 180 * time.Nanosecond
	f.archive.maxRecords = len(f.archive.Records()) + 3 // epoch, begin, outcome fit; rollback append fails
	candidate, err := l.Remeasure(context.Background(), "full-archive", r, result)
	if err == nil || candidate != nil {
		t.Fatal("failed durability was acknowledged")
	}
	next, _ := r.Next()
	if next.Plan != p {
		t.Fatal("archive failure blocked known regression rollback")
	}
}

func TestResidentGenerationExhaustionRefused(t *testing.T) {
	f, l := setup(t)
	result := run(t, l, f)
	p, q := hostPlan(t), hostPlan(t)
	r, _ := NewResident(HostStub, math.MaxInt-1, PlanBinding{f.cfg.Baseline.Identity(), p})
	result.Promotion.Generation = math.MaxInt - 1
	if err := r.Promote(result.Promotion, PlanBinding{result.Search.Winner.ID, q}); err == nil {
		t.Fatal("generation overflow accepted")
	}
	next, generation := r.Next()
	if next.Plan != p || generation != math.MaxInt-1 {
		t.Fatal("refused exhausted generation mutated resident")
	}
}
