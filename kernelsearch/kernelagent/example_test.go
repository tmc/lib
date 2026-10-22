package kernelagent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

func exampleSearch() (*fixture, *Loop, *Result, func()) {
	dir, err := os.MkdirTemp("", "kernelagent-example-")
	if err != nil {
		panic(err)
	}
	f, err := newFixture(dir)
	if err != nil {
		panic(err)
	}
	l, err := NewLoop(f.archive, f.cfg)
	if err != nil {
		panic(err)
	}
	r, err := l.RunEpoch(context.Background(), "example", f.state, nil, nil)
	if err != nil {
		panic(err)
	}
	return f, l, r, func() { r.Close(); l.Close(); f.archive.Close(); os.RemoveAll(dir) }
}

func examplePlan() *cudaplan.ExecutablePlan {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	y := g.AddUnary("relu", x)
	if err := g.SetOutputs(y); err != nil {
		panic(err)
	}
	s, err := g.Snapshot()
	if err != nil {
		panic(err)
	}
	p, err := cudaplan.Compile(s)
	if err != nil {
		panic(err)
	}
	return p
}

func Example() {
	_, _, result, close := exampleSearch()
	defer close()
	fmt.Println(result.Search.Strategy, result.Promotion.CandidateID)
	// Output: beam host-stub:fast:tool-fixture-v1
}

func ExampleOpenArchive() {
	dir, _ := os.MkdirTemp("", "kernelagent-archive-")
	defer os.RemoveAll(dir)
	a, err := OpenArchive(dir, 1<<20, 100)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer a.Close()
	fmt.Println(len(a.Records()))
	// Output: 0
}

func ExampleNewLoop() {
	f, _, _, close := exampleSearch()
	defer close()
	l, err := NewLoop(f.archive, f.cfg)
	defer l.Close()
	fmt.Println(err)
	// Output: <nil>
}

func ExampleLoop_RunEpoch() {
	_, _, result, close := exampleSearch()
	defer close()
	fmt.Println(result.Search.Winner.Validated, result.Confirmation.ConfirmationOf == result.Search.Winner.ID)
	// Output: true true
}

func ExampleLoop_Close() {
	_, l, _, close := exampleSearch()
	defer close()
	fmt.Println(l.Close())
	// Output: <nil>
}

func ExampleResult_Close() {
	_, _, result, close := exampleSearch()
	defer close()
	winner := result.Search.Winner
	fmt.Println(winner.Expr != nil, result.Promotion != nil)
	result.Close()
	fmt.Println(winner.Expr == nil, result.Search == nil, result.Confirmation == nil, result.Promotion != nil)
	// Output:
	// true true
	// true true true true
}

func ExampleArchive_Records() {
	f, _, _, close := exampleSearch()
	defer close()
	fmt.Println(f.archive.Records()[0].Kind)
	// Output: epoch
}

func ExampleArchive_CandidateRecords() {
	f, _, result, close := exampleSearch()
	defer close()
	records := f.archive.CandidateRecords(result.Search.Winner.ID)
	fmt.Println(len(records) > 0, records[0].CandidateID == result.Search.Winner.ID)
	// Output: true true
}

func ExampleArchive_Close() {
	dir, _ := os.MkdirTemp("", "kernelagent-archive-")
	defer os.RemoveAll(dir)
	a, _ := OpenArchive(dir, 4096, 10)
	fmt.Println(a.Close(), a.Close())
	// Output: <nil> <nil>
}

func ExampleFixedEnumerator_Propose() {
	f := FixedEnumerator{Edits: []json.RawMessage{json.RawMessage(`{"rules":["fast"]}`)}}
	reply, err := f.Propose(context.Background(), Request{})
	fmt.Println(reply.Model.Kind, string(reply.Edit), err)
	// Output: fixed {"rules":["fast"]} <nil>
}

func ExampleSummarize() {
	s, _ := Summarize(nil, "incumbent", "workload", "sm_fixture", []string{"memory"})
	fmt.Println(s.Available, s.Counters[0].Available, s.Inferred)
	// Output: false false true
}

func ExampleNewResident() {
	p := examplePlan()
	defer p.Close()
	r, err := NewResident(HostStub, 1, PlanBinding{"declared-host-stub", p})
	_, generation := r.Next()
	fmt.Println(generation, err)
	// Output: 1 <nil>
}

func ExampleResident_Next() {
	p := examplePlan()
	defer p.Close()
	r, _ := NewResident(HostStub, 1, PlanBinding{"declared-host-stub", p})
	next, generation := r.Next()
	fmt.Println(next.Plan == p, generation)
	// Output: true 1
}

func ExampleResident_Promote() {
	f, _, result, close := exampleSearch()
	defer close()
	p, q := examplePlan(), examplePlan()
	defer p.Close()
	defer q.Close()
	r, _ := NewResident(HostStub, 1, PlanBinding{f.cfg.Baseline.Identity(), p})
	err := r.Promote(result.Promotion, PlanBinding{result.Promotion.CandidateID, q})
	next, generation := r.Next()
	fmt.Println(next.Plan == q, generation, err)
	// Output: true 2 <nil>
}

func ExampleResident_Execute() {
	p := examplePlan()
	defer p.Close()
	r, _ := NewResident(HostStub, 1, PlanBinding{"declared-host-stub", p})
	completion, err := r.Execute(nil, cudaplan.NewBindings())
	fmt.Println(completion == nil, err != nil)
	// Output: true true
}

func ExampleLoop_Remeasure() {
	f, l, result, close := exampleSearch()
	defer close()
	p, q := examplePlan(), examplePlan()
	defer p.Close()
	defer q.Close()
	r, _ := NewResident(HostStub, 1, PlanBinding{f.cfg.Baseline.Identity(), p})
	if err := r.Promote(result.Promotion, PlanBinding{result.Promotion.CandidateID, q}); err != nil {
		fmt.Println(err)
		return
	}
	confirmed, err := l.Remeasure(context.Background(), "remeasure", r, result)
	defer confirmed.Close()
	next, _ := r.Next()
	fmt.Println(next.Plan == q, confirmed.ConfirmationOf == result.Search.Winner.ID, err)
	// Output: true true <nil>
}
