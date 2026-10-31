package cudamega_test

import (
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func TestExecutorReleaseRequiresDrain(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "p", Scope: cudataskgraph.ScopeDevice}})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name       string
		aborted    bool
		abortAfter bool
	}{
		{"normal", false, false},
		{"aborted", true, false},
		{"abort after refusal", false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, err := cudamega.NewExecutor(g, cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 32, SharedBytes: 1}, 1)
			if err != nil {
				t.Fatal(err)
			}
			if tc.aborted {
				e.Abort()
			}
			if err := e.Release("absent"); err == nil || !strings.Contains(err.Error(), "not claimed") {
				t.Fatalf("unknown release = %v; want not claimed", err)
			}
			err = e.Release("p")
			if err == nil || !strings.Contains(err.Error(), "has not been drained") {
				t.Fatalf("undrained release = %v; want drain refusal", err)
			}
			if tc.aborted && !strings.Contains(err.Error(), "abort is outstanding") {
				t.Fatalf("aborted release lost its reason: %v", err)
			}
			if got := e.Epoch().Retained(); !reflect.DeepEqual(got, []string{"p"}) {
				t.Fatalf("retained after refusal = %v; want [p]", got)
			}
			if tc.abortAfter {
				if got := e.Abort(); !reflect.DeepEqual(got, []string{"p"}) {
					t.Fatalf("abort after refused release retained %v; want [p]", got)
				}
			}
			e.Drain()
			if err := e.Release("p"); err != nil {
				t.Fatalf("drained release: %v", err)
			}
			if len(e.Epoch().Retained()) != 0 {
				t.Fatal("drained release retained storage")
			}
			if err := e.Release("p"); err == nil || !strings.Contains(err.Error(), "not claimed") {
				t.Fatalf("duplicate release = %v; want not claimed", err)
			}
		})
	}
}

func TestExecutorEpochSnapshot(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "p", Scope: cudataskgraph.ScopeDevice}})
	if err != nil {
		t.Fatal(err)
	}
	e, err := cudamega.NewExecutor(g, cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 32, SharedBytes: 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := e.Epoch()
	if err := snapshot.Publish("p", g); err != nil {
		t.Fatal(err)
	}
	snapshot.Drain()
	if err := snapshot.Release("p"); err != nil {
		t.Fatal(err)
	}
	snapshot.Abort()
	snapshot.Number = 42
	fresh := e.Epoch()
	if fresh.Number != 1 || fresh.Aborted() || fresh.Drained() || fresh.Published("p") || !reflect.DeepEqual(fresh.Retained(), []string{"p"}) {
		t.Fatal("snapshot mutation changed the executor")
	}
	if err := e.Release("p"); err == nil {
		t.Fatal("snapshot drain admitted executor release")
	}
	if got := e.Abort(); !reflect.DeepEqual(got, []string{"p"}) {
		t.Fatalf("snapshot release changed retained outputs: %v", got)
	}
	e.Drain()
	if !e.Epoch().Aborted() || !e.Epoch().Drained() || fresh.Aborted() || fresh.Drained() {
		t.Fatal("old snapshot changed with the executor")
	}
	if err := e.Release("p"); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fresh.Retained(), []string{"p"}) {
		t.Fatal("executor release changed an earlier snapshot")
	}
}

func TestExecutorCopiedLifecycle(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "p", Scope: cudataskgraph.ScopeDevice}})
	if err != nil {
		t.Fatal(err)
	}
	e, err := cudamega.NewExecutor(g, cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 32, SharedBytes: 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	copy := *e
	start := make(chan struct{})
	var wg sync.WaitGroup
	for _, executor := range []*cudamega.Executor{e, &copy} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for i := 0; i < 100; i++ {
				snapshot := executor.Epoch()
				snapshot.Drain()
				if err := snapshot.Release("p"); err != nil {
					t.Errorf("snapshot release: %v", err)
				}
				if err := executor.Release("p"); err == nil {
					t.Error("released before executor drain")
				}
				if got := executor.Abort(); !reflect.DeepEqual(got, []string{"p"}) {
					t.Errorf("abort retained %v; want [p]", got)
				}
			}
		}()
	}
	close(start)
	wg.Wait()
	copy.Drain()
	if !e.Epoch().Drained() {
		t.Fatal("value copy has independent drain state")
	}
	results := make(chan error, 2)
	for _, executor := range []*cudamega.Executor{e, &copy} {
		go func() { results <- executor.Release("p") }()
	}
	successes := 0
	for i := 0; i < 2; i++ {
		if err := <-results; err == nil {
			successes++
		} else if !strings.Contains(err.Error(), "not claimed") {
			t.Errorf("duplicate release: %v", err)
		}
	}
	if successes != 1 || len(e.Epoch().Retained()) != 0 || len(copy.Epoch().Retained()) != 0 {
		t.Fatalf("concurrent release successes = %d; want 1 and no retained output", successes)
	}
}

func TestZeroExecutorLifecycle(t *testing.T) {
	var e cudamega.Executor
	if e.Epoch() != nil || e.Abort() != nil {
		t.Fatal("zero executor has lifecycle state")
	}
	e.Drain()
	if err := e.Release("p"); err == nil || err.Error() != "nil epoch" {
		t.Fatalf("zero executor release = %v; want nil epoch", err)
	}
}
