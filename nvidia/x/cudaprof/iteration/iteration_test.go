package iteration

import (
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Driver callback ids used by the fixtures.
const (
	cbAlloc       = 243 // cuMemAlloc_v2
	cbFree        = 245 // cuMemFree_v2
	cbSync        = 126 // cuStreamSynchronize
	cbAddNode     = 502 // cuGraphAddKernelNode
	cbInstantiate = 513 // cuGraphInstantiate
	cbExecUpdate  = 561 // cuGraphExecUpdate
	cbRtMalloc    = 20  // runtime cudaMalloc
)

func kernel(name string, start, end int64) timeline.Activity {
	return timeline.Activity{Subtype: "kernel", Name: name, Start: start, End: end}
}

func memcpy(start, end int64, bytes uint64) timeline.Activity {
	return timeline.Activity{Subtype: "memcpy", Start: start, End: end, Bytes: bytes}
}

func call(cbid uint32, start, end int64) timeline.Call {
	return timeline.Call{Domain: "driver", CallbackID: cbid, Thread: 1, Start: start, End: end}
}

const gather = "_ZN3mlx4core2cu6gatherIfiLi1ELi2EiEEvPKT_PS3_"

// steps builds n iterations of 100ns, each beginning with the marker kernel.
// Iteration i runs gather at [100i, 100i+10), matmul at [100i+20, 100i+60)
// and a memcpy at [100i+50, 100i+70), which the matmul overlaps by 10.
func steps(n int) timeline.Records {
	var r timeline.Records
	for i := range int64(n) {
		t := 100 * i
		r.Activities = append(r.Activities,
			kernel(gather, t, t+10),
			kernel("matmul", t+20, t+60),
			memcpy(t+50, t+70, 64),
		)
	}
	return r
}

func TestAnalyze(t *testing.T) {
	r, err := Analyze(steps(5), Options{Marker: regexp.MustCompile(`^mlx::core::cu::gather<float, int, 1, 2`), Skip: 1})
	if err != nil {
		t.Fatal(err)
	}
	if r.Markers != 5 || len(r.Iterations) != 4 {
		t.Fatalf("markers %d, iterations %d, want 5, 4", r.Markers, len(r.Iterations))
	}
	if !r.Gate.Pass {
		t.Errorf("gate failed: %v", r.Gate.Failures)
	}
	if r.Iterations[0].Steady || !r.Iterations[1].Steady {
		t.Errorf("iteration 0 steady %v, 1 steady %v; want false, true", r.Iterations[0].Steady, r.Iterations[1].Steady)
	}
	it := r.Iterations[2]
	want := Totals{
		Iterations: 1, WindowNanos: 100, BusyNanos: 60, IdleNanos: 40,
		KernelNanos: 50, MemcpyNanos: 10, Launches: 2, Memcpys: 1, MemcpyBytes: 64,
	}
	if !equalTotals(it.Totals, want) {
		t.Errorf("iteration 2 = %+v, want %+v", it.Totals, want)
	}
	s := r.Steady
	if s.Iterations != 3 || s.WindowNanos != 300 || s.BusyNanos != 180 || s.Launches != 6 {
		t.Errorf("steady = %+v, want 3 iterations, window 300, busy 180, launches 6", s)
	}
	groups := map[string]Group{}
	for _, g := range r.Groups {
		groups[g.Group] = g
	}
	if g := groups["matmul"]; g.Launches != 3 || g.Exclusive != 120 || g.Summed != 120 {
		t.Errorf("matmul group = %+v, want 3 launches, 120 exclusive, 120 summed", g)
	}
	if g := groups["memcpy"]; g.Launches != 3 || g.Exclusive != 30 || g.Summed != 60 {
		t.Errorf("memcpy group = %+v, want 3 launches, 30 exclusive, 60 summed", g)
	}
	if r.Kernels[0].Kernel != "matmul" || r.Kernels[1].Kernel != "mlx::core::cu::gather<float, int, 1, 2, int>" {
		t.Errorf("kernels = %+v", r.Kernels)
	}
}

func TestAnalyzeClipsToIteration(t *testing.T) {
	// A kernel that straddles a marker is split between the iterations it
	// overlaps, and launched in the one it starts in.
	var r timeline.Records
	r.Activities = []timeline.Activity{
		kernel("m", 0, 10),
		kernel("long", 80, 130),
		kernel("m", 100, 110),
		kernel("m", 200, 210),
	}
	rep, err := Analyze(r, Options{Marker: regexp.MustCompile(`^m$`)})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct{ busy, launches int64 }{{30, 2}, {30, 1}}
	for i, tt := range tests {
		it := rep.Iterations[i]
		if it.BusyNanos != tt.busy || int64(it.Launches) != tt.launches {
			t.Errorf("iteration %d: busy %d, launches %d; want %d, %d", i, it.BusyNanos, it.Launches, tt.busy, tt.launches)
		}
	}
}

func TestAnalyzeNVTX(t *testing.T) {
	r := steps(4)
	r.Markers = []timeline.Marker{
		{Name: "step", Flags: timeline.MarkerStart, Time: 0},
		{Flags: timeline.MarkerEnd, Time: 90},
		{Name: "other", Flags: timeline.MarkerInstantaneous, Time: 150},
		{Name: "step", Flags: timeline.MarkerStart, Time: 200},
		{Name: "step", Flags: timeline.MarkerInstantaneous, Time: 300},
	}
	rep, err := Analyze(r, Options{Marker: regexp.MustCompile(`^step$`), NVTX: true})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Markers != 3 || len(rep.Iterations) != 2 {
		t.Fatalf("markers %d, iterations %d, want 3, 2", rep.Markers, len(rep.Iterations))
	}
	if w := rep.Iterations[0].WindowNanos; w != 200 {
		t.Errorf("iteration 0 window %d, want 200", w)
	}
}

func TestGate(t *testing.T) {
	uneven := steps(5)
	uneven.Activities = append(uneven.Activities, kernel("extra", 330, 340))
	tests := []struct {
		name    string
		records timeline.Records
		opts    Options
		want    string // substring of the failure, or "" to pass
	}{
		{"pass", steps(5), Options{Skip: 1, Expect: 5}, ""},
		{"expect", steps(5), Options{Skip: 1, Expect: 6}, "fired 5 times, want 6"},
		{"too few", steps(2), Options{Skip: 1}, "none left after skipping 1"},
		{"none", timeline.Records{}, Options{}, "fired 0 times"},
		{"uneven launches", uneven, Options{Skip: 1}, "2 in iteration 1-2; 3 in iteration 3"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.opts.Marker = regexp.MustCompile(`gather`)
			r, err := Analyze(tt.records, tt.opts)
			if err != nil {
				t.Fatal(err)
			}
			got := strings.Join(r.Gate.Failures, "\n")
			if tt.want == "" {
				if !r.Gate.Pass {
					t.Errorf("gate failed: %s", got)
				}
				return
			}
			if r.Gate.Pass || !strings.Contains(got, tt.want) {
				t.Errorf("gate pass %v, failures %q; want failure containing %q", r.Gate.Pass, got, tt.want)
			}
		})
	}
}

func TestClassify(t *testing.T) {
	tests := []struct {
		name string
		want Class
	}{
		{"cuMemAlloc_v2", Alloc},
		{"cudaMalloc", Alloc},
		{"cuMemAllocAsync", Alloc},
		{"cuMemFree_v2", Free},
		{"cudaFree", Free},
		{"cuMemPoolCreate", ""},
		{"cuStreamBeginCapture_v2", GraphBuild},
		{"cuGraphAddKernelNode", GraphBuild},
		{"cuGraphInstantiateWithFlags", GraphInstantiate},
		{"cuGraphExecUpdate_v2", GraphUpdate},
		{"cuGraphExecKernelNodeSetParams", GraphUpdate},
		{"cuGraphExecDestroy", ""},
		{"cuGraphLaunch", GraphLaunch},
		{"cuStreamSynchronize", Sync},
		{"cudaDeviceSynchronize", Sync},
		{"cuLaunchKernel", ""},
	}
	for _, tt := range tests {
		if got := Classify(tt.name); got != tt.want {
			t.Errorf("Classify(%q) = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestAdvise(t *testing.T) {
	// withCalls adds calls at the given offsets into each of 5 iterations.
	withCalls := func(calls ...timeline.Call) timeline.Records {
		r := steps(5)
		for i := range int64(5) {
			for _, c := range calls {
				c.Start += 100 * i
				c.End += 100 * i
				r.Calls = append(r.Calls, c)
			}
		}
		return r
	}
	tests := []struct {
		name    string
		records timeline.Records
		want    []string // rules, in order
	}{
		{"idle only", steps(5), []string{"idle"}},
		{"alloc", withCalls(call(cbAlloc, 1, 50), call(cbFree, 60, 65)), []string{"steady-alloc", "idle"}},
		{
			// cudaMalloc and the cuMemAlloc it makes are one allocation.
			"nested alloc",
			withCalls(
				timeline.Call{Domain: "runtime", CallbackID: cbRtMalloc, Thread: 1, Start: 1, End: 30},
				call(cbAlloc, 2, 29),
			),
			[]string{"idle", "steady-alloc"},
		},
		{"graph rebuild", withCalls(call(cbAddNode, 1, 20), call(cbInstantiate, 21, 90)), []string{"graph-rebuild", "idle"}},
		{"graph update", withCalls(call(cbExecUpdate, 1, 90)), []string{"idle"}},
		{"sync", withCalls(call(cbSync, 70, 99)), []string{"idle", "host-sync"}},
		{"sync while busy", withCalls(call(cbSync, 30, 45)), []string{"idle"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := Analyze(tt.records, Options{Marker: regexp.MustCompile(`gather`), Skip: 1})
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			for _, f := range r.Advise() {
				got = append(got, f.Rule)
			}
			if strings.Join(got, ",") != strings.Join(tt.want, ",") {
				t.Errorf("rules = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAdviseNestedCount(t *testing.T) {
	r := steps(3)
	r.Calls = []timeline.Call{
		{Domain: "runtime", CallbackID: cbRtMalloc, Thread: 1, Start: 101, End: 130},
		call(cbAlloc, 102, 129),
	}
	rep, err := Analyze(r, Options{Marker: regexp.MustCompile(`gather`), Skip: 1})
	if err != nil {
		t.Fatal(err)
	}
	if s := rep.Steady.Classes[Alloc]; s.Count != 1 || s.Nanos != 29 {
		t.Errorf("alloc = %+v, want 1 call, 29ns", s)
	}
	if rep.Steady.Calls != 2 {
		t.Errorf("calls = %d, want 2", rep.Steady.Calls)
	}
}

func TestKernelName(t *testing.T) {
	tests := []struct{ in, want string }{
		{gather, "mlx::core::cu::gather<float, int, 1, 2, int>"},
		{"Embed", "Embed"},
		{"  matmul ", "matmul"},
	}
	for _, tt := range tests {
		if got := KernelName(tt.in); got != tt.want {
			t.Errorf("KernelName(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func equalTotals(a, b Totals) bool {
	a.Classes, b.Classes = nil, nil
	return reflect.DeepEqual(a, b)
}

func TestSyncIdle(t *testing.T) {
	// Each iteration is busy over [0,10) and [20,70) and idle over [10,20)
	// and [70,100).
	tests := []struct {
		name       string
		start, end int64
		want       int64
	}{
		{"returns into drain", 60, 80, 30},
		{"returns into first gap", 5, 15, 10},
		{"returns while busy", 30, 40, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := steps(3)
			r.Calls = []timeline.Call{call(cbSync, 100+tt.start, 100+tt.end)}
			rep, err := Analyze(r, Options{Marker: regexp.MustCompile(`gather`), Skip: 1})
			if err != nil {
				t.Fatal(err)
			}
			if got := rep.Steady.SyncIdleNanos; got != tt.want {
				t.Errorf("sync idle = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestSyncIdleShared(t *testing.T) {
	// Two syncs returning into the drain gap [70,100) count it once.
	r := steps(3)
	r.Calls = []timeline.Call{call(cbSync, 160, 175), call(cbSync, 176, 180)}
	rep, err := Analyze(r, Options{Marker: regexp.MustCompile(`gather`), Skip: 1})
	if err != nil {
		t.Fatal(err)
	}
	if got := rep.Steady.SyncIdleNanos; got != 30 {
		t.Errorf("sync idle = %d, want 30", got)
	}
}
