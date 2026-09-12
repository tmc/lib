package main

import (
	"encoding/json"
	"fmt"
	"github.com/tmc/lib/nvidia/x/nccltest"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCheckEnvironment(t *testing.T) {
	for _, tt := range []struct {
		name string
		env  []string
		want string
	}{
		{"empty", nil, ""},
		{"shared library controls", []string{"NCCL_DEBUG=WARN", "CUDA_VISIBLE_DEVICES=0"}, ""},
		{"split", []string{"NCCL_TESTS_SPLIT=MOD 2"}, "NCCL_TESTS_SPLIT"},
		{"device", []string{"NCCL_TESTS_DEVICE=1"}, "NCCL_TESTS_DEVICE"},
		{"traffic", []string{"NCCL_TESTS_ALLTOALLV_SPREAD=0"}, "NCCL_TESTS_ALLTOALLV_SPREAD"},
		{"empty override", []string{"NCCL_TESTS_ALLTOALLV_MATRIX_FILE="}, "NCCL_TESTS_ALLTOALLV_MATRIX_FILE"},
		{"future override", []string{"NCCL_TESTS_NEW_MODE=private-value"}, "NCCL_TESTS_NEW_MODE"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := checkEnvironment(tt.env)
			if tt.want == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v, want rejection naming %s", err, tt.want)
			}
			if strings.Contains(err.Error(), "private-value") {
				t.Fatal("error exposes environment value")
			}
		})
	}
}

func TestSelectSizes(t *testing.T) {
	for _, tt := range []struct {
		name, list, want          string
		explicit                  bool
		first, last, step, factor int
	}{
		{"list", "16,32", "[16 32]", true, 0, 0, 0, 0},
		{"linear", "ignored", "[16 32 48]", false, 16, 50, 16, 0},
		{"geometric", "ignored", "[16 32 64]", false, 16, 64, 0, 2},
		{"mixed", "16", "", true, 16, 64, 16, 0},
		{"two modes", "", "", false, 16, 64, 16, 2},
		{"missing mode", "", "", false, 16, 64, 0, 0},
		{"missing first", "", "", false, 0, 64, 16, 0},
		{"duplicate", "16,16", "", true, 0, 0, 0, 0},
		{"zero", "0", "", true, 0, 0, 0, 0},
		{"malformed", "16,no", "", true, 0, 0, 0, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := selectSizes(tt.list, tt.explicit, tt.first, tt.last, tt.step, tt.factor)
			if tt.want == "" {
				if err == nil {
					t.Fatalf("accepted invalid sweep: %v", got)
				}
			} else if err != nil || fmt.Sprint(got) != tt.want {
				t.Fatalf("got %v, %v; want %s", got, err, tt.want)
			}
		})
	}
}

func TestPrepareWorkloads(t *testing.T) {
	for _, tt := range []struct {
		name, operations, reductions string
		ranks, bytes, want           int
	}{
		{"valid", "all_reduce,gather", "sum,avg", 4, 1024, 3},
		{"unknown operation", "bogus", "sum", 1, 1024, 0},
		{"unknown reduction", "all_reduce", "bogus", 1, 1024, 0},
		{"duplicate operation", "gather,gather", "sum", 1, 1024, 0},
		{"duplicate reduction", "reduce", "sum,sum", 1, 1024, 0},
		{"empty operation", "", "sum", 1, 1024, 0},
		{"zero work", "gather", "sum", 4, 16, 0},
		{"hypercube ranks", "hypercube", "sum", 3, 1024, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := prepareWorkloads(tt.operations, tt.reductions, []int{tt.bytes}, nccltest.Float16, tt.ranks, 0)
			if (err != nil) != (tt.want == 0) || len(got) != tt.want {
				t.Fatalf("got %v, %v; want %d workloads", got, err, tt.want)
			}
		})
	}
}

func TestReferenceTiming(t *testing.T) {
	for _, tt := range []struct{ mode, want string }{
		{"rank0", "0"}, {"mean", "1"}, {"min", "2"}, {"max", "3"},
	} {
		got, err := referenceTiming(tt.mode)
		if err != nil || got != tt.want {
			t.Fatalf("%s: got %q, %v; want %q", tt.mode, got, err, tt.want)
		}
	}
	for _, mode := range []string{"", "average", "4"} {
		if _, err := referenceTiming(mode); err == nil {
			t.Fatalf("accepted %q", mode)
		}
	}
}

func TestReceiptCheckpoint(t *testing.T) {
	dir := t.TempDir()
	initial := receipt{Ranks: 1}
	if err := writeReceipt(dir, initial); err != nil {
		t.Fatal(err)
	}
	read := func() receipt {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(dir, "comparison.json"))
		if err != nil {
			t.Fatal(err)
		}
		var r receipt
		if err := json.Unmarshal(data, &r); err != nil {
			t.Fatal(err)
		}
		return r
	}
	if read().Complete {
		t.Fatal("initial checkpoint claims completion")
	}
	if err := os.WriteFile(filepath.Join(dir, "comparison.json.next"), []byte("truncated"), 0644); err != nil {
		t.Fatal(err)
	}
	if read().Complete {
		t.Fatal("partial write changed checkpoint")
	}
	initial.Complete = true
	if err := writeReceipt(dir, initial); err != nil {
		t.Fatal(err)
	}
	if !read().Complete {
		t.Fatal("final receipt is incomplete")
	}
}

func TestValidTiming(t *testing.T) {
	for _, tt := range []struct {
		name string
		r    nccltest.Result
		want bool
	}{
		{"zero bandwidth is valid for one-rank routing", nccltest.Result{GPUTime: time.Microsecond}, true},
		{"positive", nccltest.Result{GPUTime: time.Microsecond, AlgorithmGBs: 1, BusGBs: 2}, true},
		{"missing time", nccltest.Result{}, false},
		{"negative time", nccltest.Result{GPUTime: -1}, false},
		{"negative bandwidth", nccltest.Result{GPUTime: 1, BusGBs: -1}, false},
		{"nan", nccltest.Result{GPUTime: 1, AlgorithmGBs: math.NaN()}, false},
		{"infinite", nccltest.Result{GPUTime: 1, BusGBs: math.Inf(1)}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := validTiming(tt.r); got != tt.want {
				t.Fatalf("got %v want %v", got, tt.want)
			}
		})
	}
}

func TestGraphReceiptRejectsMissingReplay(t *testing.T) {
	good := nccltest.Result{GraphLaunches: 3, GraphComparisons: 32, Comparisons: 64}
	for _, tt := range []struct {
		name   string
		change func(*nccltest.Result)
		want   bool
	}{
		{"valid", func(r *nccltest.Result) {}, true},
		{"ordinary only", func(r *nccltest.Result) { r.GraphComparisons = 0 }, false},
		{"wrong replay", func(r *nccltest.Result) { r.GraphWrong = 1 }, false},
		{"short replay", func(r *nccltest.Result) { r.GraphComparisons-- }, false},
		{"wrong launches", func(r *nccltest.Result) { r.GraphLaunches-- }, false},
		{"odd total", func(r *nccltest.Result) { r.Comparisons++ }, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := good
			tt.change(&r)
			if got := validGraphResult(r, 3); got != tt.want {
				t.Fatalf("got %v; want %v", got, tt.want)
			}
		})
	}
	if validGraphResult(good, 0) {
		t.Fatal("replay accepted for ordinary request")
	}
}

func TestComparisonCount(t *testing.T) {
	for _, tt := range []struct {
		name                           string
		op                             nccltest.Operation
		ranks, checks, launches, count int
		want                           bool
	}{
		{"all ranks", nccltest.AllReduce, 3, 2, 0, 48, true},
		{"missing rank", nccltest.AllReduce, 3, 2, 0, 32, false},
		{"missing pass", nccltest.AllReduce, 3, 2, 0, 24, false},
		{"root reduce", nccltest.Reduce, 3, 2, 0, 16, true},
		{"root gather", nccltest.Gather, 3, 2, 0, 16, true},
		{"counted undefined outputs", nccltest.Reduce, 3, 2, 0, 48, false},
		{"graph", nccltest.AllReduce, 3, 2, 4, 96, true},
		{"missing graph pass", nccltest.AllReduce, 3, 2, 4, 48, false},
		{"overflow fields", nccltest.AllReduce, int(^uint(0) >> 1), 2, 4, 16, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := nccltest.Result{Case: nccltest.Case{Operation: tt.op}, Ranks: tt.ranks, Checks: tt.checks, GraphLaunches: tt.launches, Comparisons: tt.count}
			r.Layout.RecvCount = 8
			if got := validComparisonCount(r); got != tt.want {
				t.Fatalf("got %v; want %v", got, tt.want)
			}
		})
	}
}

func TestOffsetSizePreflight(t *testing.T) {
	limit := int(^uint(0) >> 1)
	for _, tt := range []struct {
		sizes  []int
		offset int
		ok     bool
	}{
		{[]int{16, 1024}, 3, true},
		{[]int{limit - 24}, 3, true},
		{[]int{16, limit - 23}, 3, false},
		{[]int{limit}, 0, true},
		{[]int{16}, -1, false},
		{[]int{16}, (1<<31-1)/8 + 1, false},
	} {
		if err := validateOffsetSizes(tt.sizes, tt.offset); (err == nil) != tt.ok {
			t.Fatalf("sizes=%v offset=%d: %v", tt.sizes, tt.offset, err)
		}
	}
}

func TestSameCaseSpread(t *testing.T) {
	left, right := 0.5, 0.5
	a := nccltest.Case{Operation: nccltest.AllToAllV, Spread: &left}
	b := nccltest.Case{Operation: nccltest.AllToAllV, Spread: &right}
	if !sameCase(a, b) {
		t.Fatal("equal decoded values rejected")
	}
	right = 1
	if sameCase(a, b) {
		t.Fatal("changed spread accepted")
	}
	b.Spread = nil
	if sameCase(a, b) {
		t.Fatal("missing spread accepted")
	}
	a.Spread = nil
	if !sameCase(a, b) {
		t.Fatal("default spread rejected")
	}
	b.Bytes = 16
	if sameCase(a, b) {
		t.Fatal("different case accepted")
	}
}
