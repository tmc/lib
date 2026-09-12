package nccltest

import (
	"errors"
	"reflect"
	"testing"
)

func TestTimedIterations(t *testing.T) {
	for _, tc := range []struct {
		name        string
		aggregation int
		want        string
	}{{"ungrouped", 1, "cc"}, {"grouped", 3, "scccesccce"}} {
		t.Run(tc.name, func(t *testing.T) {
			var trace string
			err := timedIterations(2, tc.aggregation, func() error { trace += "c"; return nil }, func() error { trace += "s"; return nil }, func() error { trace += "e"; return nil })
			if err != nil || trace != tc.want {
				t.Fatalf("got %q, %v; want %q", trace, err, tc.want)
			}
		})
	}
}
func TestTimedIterationFailure(t *testing.T) {
	callErr, endErr := errors.New("submit"), errors.New("end")
	calls, ends := 0, 0
	err := timedIterations(4, 3, func() error { calls++; return callErr }, func() error { return nil }, func() error { ends++; return endErr })
	if calls != 1 || ends != 1 || !errors.Is(err, callErr) || !errors.Is(err, endErr) {
		t.Fatalf("calls=%d ends=%d err=%v", calls, ends, err)
	}
	err = timedIterations(1, 3, func() error { t.Fatal("submitted after failed start"); return nil }, func() error { return callErr }, func() error { t.Fatal("ended an unopened group"); return nil })
	if !errors.Is(err, callErr) {
		t.Fatal(err)
	}
}
func TestIterationCounts(t *testing.T) {
	a, n, err := iterationCounts(20, 0)
	if err != nil || !reflect.DeepEqual([]int{a, n}, []int{1, 20}) {
		t.Fatalf("%d %d %v", a, n, err)
	}
	for _, counts := range [][2]int{{0, 1}, {1, -1}, {int(^uint(0) >> 1), 2}} {
		if _, _, err := iterationCounts(counts[0], counts[1]); err == nil {
			t.Fatalf("accepted %v", counts)
		}
	}
}
