package main

import (
	"errors"
	"slices"
	"testing"
)

func TestAfmoeGenerationBoundary(t *testing.T) {
	var positions []int
	var inputs []int32
	var samples []bool
	drains := 0
	ids, _, intervals, err := generateAfmoe([]int32{1, 2}, 3, 4,
		func() error { drains++; return nil },
		func(token int32, pos int, sample bool) (int32, error) {
			inputs = append(inputs, token)
			positions = append(positions, pos)
			samples = append(samples, sample)
			return token + 10, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, []int32{12, 22, 32}) || !slices.Equal(inputs, []int32{1, 2, 12, 22}) || !slices.Equal(positions, []int{0, 1, 2, 3}) || !slices.Equal(samples, []bool{false, true, true, true}) || len(intervals) != 2 || drains != 3 {
		t.Fatalf("wrong sample/forward boundary: ids=%v inputs=%v pos=%v samples=%v intervals=%d drains=%d", ids, inputs, positions, samples, len(intervals), drains)
	}
}

func TestAfmoeGenerationStopsOnError(t *testing.T) {
	failure := errors.New("forward failed")
	calls := 0
	ids, _, intervals, err := generateAfmoe([]int32{2}, 128, 128,
		func() error { return nil }, func(token int32, pos int, sample bool) (int32, error) {
			calls++
			if pos == 2 {
				return 0, failure
			}
			return token + 1, nil
		})
	if !errors.Is(err, failure) || calls != 3 || !slices.Equal(ids, []int32{3, 4}) || len(intervals) != 1 {
		t.Fatalf("error boundary: calls=%d ids=%v intervals=%d error=%v", calls, ids, len(intervals), err)
	}
}

func TestAfmoeGenerationContext(t *testing.T) {
	calls := 0
	_, _, _, err := generateAfmoe([]int32{1, 2}, 128, 128,
		func() error { calls++; return nil }, func(int32, int, bool) (int32, error) { calls++; return 0, nil })
	if err == nil || calls != 0 {
		t.Fatalf("invalid context entered execution: calls=%d error=%v", calls, err)
	}
}
