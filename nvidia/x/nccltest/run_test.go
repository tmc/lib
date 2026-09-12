package nccltest

import (
	"context"
	"errors"
	"github.com/tmc/lib/nvidia/nccl"
	"slices"
	"strings"
	"testing"
)

func TestRunRejectsInvalidOptionsBeforeCUDA(t *testing.T) {
	for _, tt := range []struct {
		name    string
		options Options
		want    string
	}{
		{"negative graphs", Options{Iterations: 1, Checks: 1, GraphLaunches: -1}, "invalid graph configuration"},
		{"default stream graphs", Options{Iterations: 1, Checks: 1, GraphLaunches: 1, DefaultStream: true}, "invalid graph configuration"},
		{"graph operations overflow", Options{Iterations: 2, Checks: 1, GraphLaunches: int(^uint(0) >> 1)}, "overflowing graph operation count"},
		{"graph checks overflow", Options{Iterations: 1, Checks: int(^uint(0) >> 1), GraphLaunches: 1}, "overflowing graph check count"},
		{"negative device", Options{Devices: []int32{-1}, Iterations: 1, Checks: 1}, "invalid or repeated device"},
		{"repeated device", Options{Devices: []int32{0, 0}, Iterations: 1, Checks: 1}, "invalid or repeated device"},
		{"timing", Options{Devices: []int32{0}, Iterations: 1, Checks: 1, Timing: "bogus"}, "unknown timing selection"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Run(Case{Operation: AllReduce, Bytes: 1024}, tt.options)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v; want %q", err, tt.want)
			}
		})
	}
}

func TestAbortPartial(t *testing.T) {
	failure := errors.New("abort failed")
	var ranks []int
	states := make([]rankState, 4)
	err := abortPartial(states, []nccl.Comm{0, 11, 0, 22}, func(rank int, comm nccl.Comm) error {
		ranks = append(ranks, rank)
		if rank == 1 {
			return failure
		}
		if comm != 22 {
			t.Fatalf("unexpected communicator %v", comm)
		}
		return nil
	})
	if !errors.Is(err, failure) || !slices.Equal(ranks, []int{1, 3}) {
		t.Fatalf("ranks=%v err=%v", ranks, err)
	}
	for i, s := range states {
		if !s.retainResources {
			t.Fatalf("rank %d not retained after failed abort", i)
		}
	}
	if err := abortPartial(nil, nil, func(int, nccl.Comm) error { t.Fatal("empty cleanup invoked"); return nil }); err != nil {
		t.Fatal(err)
	}
}

func TestSingleRankInPlaceIdentity(t *testing.T) {
	c := Case{Operation: AllReduce, Bytes: 65536, InPlace: true}
	data := inputBuffer(c, 1, 0, c.Bytes/4)
	n, wrong, err := CheckBytes(c, 1, 0, data)
	if err != nil || n != 16384 || wrong != 0 {
		t.Fatalf("identity count=%d wrong=%d err=%v", n, wrong, err)
	}
	// Disjoint receive poison provides the non-identity omission control.
	n, wrong, err = CheckBytes(c, 1, 0, poisonBuffer(c, 16384))
	if err != nil || n != 16384 || wrong != n {
		t.Fatalf("poison count=%d wrong=%d err=%v", n, wrong, err)
	}
}

func TestBufferAllocation(t *testing.T) {
	limit := int(^uint(0) >> 1)
	for _, tt := range []struct {
		name                       string
		size, offset, bytes, shift int
		ok                         bool
	}{
		{"aligned", 4, 0, 64, 0, true},
		{"element offset", 4, 3, 76, 12, true},
		{"byte elements", 1, 1, 17, 1, true},
		{"negative", 4, -1, 0, 0, false},
		{"overflow shift", 8, limit, 0, 0, false},
		{"overflow total", 1, limit - 15, 0, 0, false},
		{"last representable", 1, limit - 16, limit, limit - 16, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			bytes, shift, err := bufferAllocation(Layout{SendCount: 8, RecvCount: 16}, tt.size, tt.offset)
			if (err == nil) != tt.ok || bytes != tt.bytes || shift != tt.shift {
				t.Fatalf("got %d,%d,%v", bytes, shift, err)
			}
		})
	}
}

func TestCycleFailuresStopSequence(t *testing.T) {
	failure := errors.New("cycle failure")
	for _, where := range []string{"none", "cycle", "yield"} {
		t.Run(where, func(t *testing.T) {
			calls, yields := 0, 0
			_, err := repeatCycles(3, func() (Result, error) {
				calls++
				if where == "cycle" && calls == 2 {
					return Result{}, failure
				}
				return Result{Comparisons: calls}, nil
			}, func(r Result) error {
				yields++
				if r.Comparisons != yields {
					t.Fatal("out-of-order cycle")
				}
				if where == "yield" && yields == 2 {
					return failure
				}
				return nil
			})
			if where == "none" {
				if err != nil || calls != 3 || yields != 3 {
					t.Fatalf("calls=%d yields=%d err=%v", calls, yields, err)
				}
			} else if !errors.Is(err, failure) || calls != 2 || (where == "cycle" && yields != 1) || (where == "yield" && yields != 2) {
				t.Fatalf("calls=%d yields=%d err=%v", calls, yields, err)
			}
		})
	}
}

func TestCyclesValidateBeforeCUDA(t *testing.T) {
	if err := RunCycles(Case{}, Options{}, 1, nil); err == nil {
		t.Fatal("nil callback accepted")
	}
	for _, count := range []int{0, -1} {
		if err := RunCycles(Case{}, Options{}, count, func(Result) error { t.Fatal("callback called"); return nil }); err == nil {
			t.Fatal("invalid count accepted")
		}
	}
}

func TestComparisonCountOverflow(t *testing.T) {
	for _, tt := range []struct {
		op                            Operation
		elements, ranks, passes, want int
	}{
		{AllReduce, 8, 3, 2, 48}, {Reduce, 8, 3, 2, 16}, {Gather, 8, 3, 4, 32},
		{AllReduce, int(^uint(0) >> 1), 2, 1, 0}, {AllReduce, 8, 2, int(^uint(0) >> 1), 0},
	} {
		n, err := comparisonCount(Case{Operation: tt.op}, Layout{RecvCount: tt.elements}, tt.ranks, tt.passes)
		if n != tt.want || (err == nil) != (tt.want > 0) {
			t.Fatalf("%+v: got %d %v", tt, n, err)
		}
	}
	_, err := Run(Case{Operation: AllReduce, Bytes: 1024}, Options{Devices: []int32{0}, Iterations: 1, Checks: int(^uint(0) >> 1)})
	if err == nil || !strings.Contains(err.Error(), "overflowing comparison count") {
		t.Fatalf("pre-CUDA check: %v", err)
	}
}

func TestSweepOrderAndStop(t *testing.T) {
	stop := errors.New("stop sweep")
	for _, fail := range []bool{false, true} {
		var seen []int
		_, err := repeatSweep(2, []int{1024, 16, 64}, func(bytes int) (Result, error) {
			seen = append(seen, bytes)
			if fail && len(seen) == 4 {
				return Result{}, stop
			}
			return Result{Case: Case{Bytes: bytes}}, nil
		}, nil)
		want := []int{1024, 16, 64, 1024, 16, 64}
		if fail {
			want = want[:4]
		}
		if !slices.Equal(seen, want) || (fail && !errors.Is(err, stop)) || (!fail && err != nil) {
			t.Fatalf("seen=%v err=%v", seen, err)
		}
	}
}

func TestSweepValidatesEverySizeBeforeCUDA(t *testing.T) {
	err := RunSweep(Case{Operation: AllReduce}, Options{Devices: []int32{0}, Iterations: 1, Checks: 1}, []int{1024, 0}, 1, func(Result) error { t.Fatal("callback before validation"); return nil })
	if err == nil || !strings.Contains(err.Error(), "size must contain") {
		t.Fatalf("got %v", err)
	}
}

func TestSynchronizedIterationOrder(t *testing.T) {
	var calls []string
	action := func(name string) func() error { return func() error { calls = append(calls, name); return nil } }
	err := timedIterationsSync(2, 2, action("submit"), action("start"), action("end"), action("sync"))
	want := []string{"start", "submit", "submit", "end", "sync", "start", "submit", "submit", "end", "sync"}
	if err != nil || !slices.Equal(calls, want) {
		t.Fatalf("calls=%v err=%v", calls, err)
	}
	stop := errors.New("sync failed")
	calls = nil
	err = timedIterationsSync(3, 1, action("submit"), action("start"), action("end"), func() error { return stop })
	if !errors.Is(err, stop) || !slices.Equal(calls, []string{"submit"}) {
		t.Fatalf("calls=%v err=%v", calls, err)
	}
}

func TestSyncGraphsRejectedBeforeCUDA(t *testing.T) {
	_, err := Run(Case{}, Options{Iterations: 1, SyncIterations: true, GraphLaunches: 1})
	if err == nil || !strings.Contains(err.Error(), "synchronized iterations cannot use graphs") {
		t.Fatalf("got %v", err)
	}
}

func TestSynchronizedIterationFailure(t *testing.T) {
	failure := errors.New("iteration failed")
	for _, tt := range []struct {
		name string
		want []string
	}{
		{"start", []string{"start"}},
		{"submit", []string{"start", "submit", "end"}},
		{"end", []string{"start", "submit", "submit", "end"}},
		{"sync", []string{"start", "submit", "submit", "end", "sync"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			action := func(name string) func() error {
				return func() error {
					calls = append(calls, name)
					if name == tt.name {
						return failure
					}
					return nil
				}
			}
			err := timedIterationsSync(3, 2, action("submit"), action("start"), action("end"), action("sync"))
			if !errors.Is(err, failure) || !slices.Equal(calls, tt.want) {
				t.Fatalf("calls=%v, want %v; error=%v", calls, tt.want, err)
			}
		})
	}
}

func TestSweepContextCancellation(t *testing.T) {
	for _, count := range []int{0, 2} {
		for _, cancelAt := range []int{0, 1, 4} {
			ctx, cancel := context.WithCancel(context.Background())
			var seen []int
			yields := 0
			if cancelAt == 0 {
				cancel()
			}
			_, err := repeatSweepContext(ctx, count, []int{16, 64}, func(bytes int) (Result, error) {
				seen = append(seen, bytes)
				return Result{}, nil
			}, func(Result) error {
				yields++
				if yields == cancelAt {
					cancel()
				}
				return nil
			})
			cancel()
			if !errors.Is(err, context.Canceled) || yields != cancelAt || !slices.Equal(seen, []int{16, 64, 16, 64}[:cancelAt]) {
				t.Fatalf("count=%d cancelAt=%d seen=%v yields=%d error=%v", count, cancelAt, seen, yields, err)
			}
		}
	}
}

func TestSweepContextFinishesCurrentSample(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls, yields := 0, 0
	_, err := repeatSweepContext(ctx, 0, []int{16, 64}, func(int) (Result, error) {
		calls++
		cancel()
		return Result{Comparisons: 4}, nil
	}, func(r Result) error {
		yields++
		if r.Comparisons != 4 {
			t.Fatal("lost completed sample")
		}
		return nil
	})
	if !errors.Is(err, context.Canceled) || calls != 1 || yields != 1 {
		t.Fatalf("calls=%d yields=%d error=%v", calls, yields, err)
	}
}

func TestSweepContextBeforeCUDA(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	yield := func(Result) error { t.Fatal("callback called"); return nil }
	if err := RunSweepContext(ctx, Case{}, Options{}, []int{16}, 0, yield); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled context: %v", err)
	}
	if err := RunSweepContext(nil, Case{}, Options{}, []int{16}, 0, yield); err == nil {
		t.Fatal("nil context accepted")
	}
	if err := RunSweepContext(context.Background(), Case{}, Options{}, []int{16}, -1, yield); err == nil {
		t.Fatal("negative count accepted")
	}
	if err := RunSweepContext(context.Background(), Case{}, Options{}, []int{16}, 0, nil); err == nil {
		t.Fatal("nil callback accepted")
	}
}

func TestAbortPartialSuccessReleasesRanks(t *testing.T) {
	states := make([]rankState, 2)
	err := abortPartial(states, []nccl.Comm{11, 22}, func(int, nccl.Comm) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range states {
		if s.retainResources {
			t.Fatalf("rank %d retained after successful abort", i)
		}
	}
}
