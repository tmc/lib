package cudamem

import (
	"errors"
	"strings"
	"testing"
)

func TestArenaFreeReverseOrder(t *testing.T) {
	a := NewArena(nil)
	var order []int
	for i := range 3 {
		a.frees = append(a.frees, func() error {
			order = append(order, i)
			return nil
		})
	}

	if err := a.Free(); err != nil {
		t.Fatalf("Free() = %v, want nil", err)
	}
	want := []int{2, 1, 0}
	if len(order) != len(want) {
		t.Fatalf("released %v, want %v", order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("released %v, want %v", order, want)
		}
	}

	// A second Free has nothing left to release.
	order = nil
	if err := a.Free(); err != nil {
		t.Fatalf("second Free() = %v, want nil", err)
	}
	if len(order) != 0 {
		t.Errorf("second Free() released %v, want nothing", order)
	}
}

func TestArenaFreeReportsErrors(t *testing.T) {
	errFirst := errors.New("first")
	errSecond := errors.New("second")

	a := NewArena(nil)
	a.frees = append(a.frees,
		func() error { return errFirst },
		func() error { return nil },
		func() error { return errSecond },
	)

	err := a.Free()
	if !errors.Is(err, errFirst) || !errors.Is(err, errSecond) {
		t.Errorf("Free() = %v, want both release errors", err)
	}
}

func TestArenaRecordsFirstError(t *testing.T) {
	// A zero Device has no context, so every allocation fails.
	a := NewArena(nil)

	x := AllocIn[float32](a, "x", 16)
	if x.Ptr() != 0 || x.Len() != 0 {
		t.Errorf("AllocIn on a failing arena = %+v, want the zero Slice", x)
	}
	first := a.Err()
	if first == nil {
		t.Fatal("Err() = nil after a failed allocation, want an error")
	}
	if !strings.Contains(first.Error(), "arena alloc x") {
		t.Errorf("Err() = %v, want it to name allocation x", first)
	}

	if y := AllocIn[int32](a, "y", 1); y.Ptr() != 0 {
		t.Errorf("AllocIn after a failure = %+v, want the zero Slice", y)
	}
	if a.Err() != first {
		t.Errorf("Err() = %v, want the first error %v", a.Err(), first)
	}
	if err := a.Free(); err != nil {
		t.Errorf("Free() = %v, want nil: nothing was allocated", err)
	}
}
