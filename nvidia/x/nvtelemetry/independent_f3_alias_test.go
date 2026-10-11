package nvtelemetry_test

import (
	n "github.com/tmc/lib/nvidia/x/nvtelemetry"
	"sync"
	"testing"
)

func TestIndependentConcurrentResultMutation(t *testing.T) {
	s := independentSnapshot()
	tab, _ := n.NewJobAttributionTable(1, 500)
	r, e := tab.Refresh(s, 200)
	if e != nil {
		t.Fatal(e)
	}
	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < 1000; i++ {
			r[0].Assignment.Provenance[0].Revision = "caller-owned"
			r[0].Assignment.Process.NamespacePIDs[0] = uint32(i)
		}
	}()
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < 1000; i++ {
			_ = tab.Resolve(s.Processes[0].Process, "GPU-1", 210)
		}
	}()
	close(start)
	wg.Wait()
}

func TestIndependentBoundsAndFailedRefresh(t *testing.T) {
	s := independentSnapshot()
	tab, _ := n.NewJobAttributionTable(1, 500)
	if _, e := tab.Refresh(s, 200); e != nil {
		t.Fatal(e)
	}
	bad := independentSnapshot()
	bad.Mappings[0].Device.UUID = ""
	if _, e := tab.Refresh(bad, 210); e == nil {
		t.Fatal("invalid mapping admitted")
	}
	got := tab.Resolve(s.Processes[0].Process, "GPU-1", 220)
	if got.Assignment.State != "assigned" || got.Assignment.Owner.AllocationID != "alloc-1" {
		t.Fatal("simple invalid snapshot damaged valid cache")
	}
	s.Processes = make([]n.ProcEntityFacts, 4097)
	if _, e := n.JoinSlurmAttribution(s.Allocations, s.Mappings, s.Processes, s.Executions, 220); e == nil {
		t.Fatal("input record bound not enforced")
	}
}
