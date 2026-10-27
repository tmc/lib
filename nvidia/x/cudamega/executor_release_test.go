package cudamega_test

import (
	"reflect"
	"strings"
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

func TestExecutorEpochReleaseRemainsCallerManaged(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "p", Scope: cudataskgraph.ScopeDevice}})
	if err != nil {
		t.Fatal(err)
	}
	e, err := cudamega.NewExecutor(g, cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 32, SharedBytes: 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Epoch().Release("p"); err != nil {
		t.Fatalf("caller-managed epoch release changed: %v", err)
	}
	if e.Epoch().Drained() || len(e.Epoch().Retained()) != 0 {
		t.Fatal("direct epoch release changed the drain contract")
	}
}
