package firstelem

import (
	"errors"

	"github.com/tmc/lib/nvidia/driver"
)

func mismatched(opts []driver.JITOption, vals []uintptr) (driver.LinkState, error) {
	if len(opts) == 0 || len(vals) == 0 {
		return 0, errors.New("no options")
	}
	return driver.LinkCreate(uint32(len(vals)), &opts[0], &vals[0]) // want `count is len\(vals\), but the buffer after it is opts`
}

func untested(opts []driver.JITOption, vals []uintptr) (driver.LinkState, error) {
	return driver.LinkCreate(uint32(len(opts)), &opts[0], &vals[0]) // want `&opts\[0\] panics if opts is empty; test len\(opts\) first` `&vals\[0\] panics if vals is empty; test len\(vals\) first`
}

func fixed() (driver.LinkState, error) {
	opts := []driver.JITOption{driver.JITInfoLogBufferSizeBytes}
	vals := []uintptr{0}
	return driver.LinkCreate(uint32(len(opts)), &opts[0], &vals[0])
}

func tested(opts []driver.JITOption, vals []uintptr) (driver.LinkState, error) {
	if len(opts) != len(vals) || len(opts) == 0 {
		return 0, errors.New("bad options")
	}
	return driver.LinkCreate(uint32(len(opts)), &opts[0], &vals[0])
}

// countAfter passes len(ids) after &ids[0]; it counts ids, not values.
func countAfter(ids []int32) error {
	if len(ids) == 0 {
		return nil
	}
	values := make([]float64, len(ids))
	return driver.Batch(&ids[0], uint32(len(ids)), &values[0])
}

// made sizes opts by vals.
func made(vals []uintptr) (driver.LinkState, error) {
	if len(vals) == 0 {
		return 0, nil
	}
	opts := make([]driver.JITOption, len(vals))
	opts[0] = driver.JITInfoLogBufferSizeBytes
	return driver.LinkCreate(uint32(len(vals)), &opts[0], &vals[0])
}

// indexedFirst reads opts[0] first, which would already have panicked.
func indexedFirst(opts []driver.JITOption, vals []uintptr) (driver.LinkState, error) {
	opts[0], vals[0] = driver.JITInfoLogBufferSizeBytes, 0
	return driver.LinkCreate(uint32(len(opts)), &opts[0], &vals[0])
}

// grownSlices are built by appending elements, so they are never empty.
func grownSlices(extra []uintptr, vals []uintptr) (driver.LinkState, error) {
	opts := []driver.JITOption{driver.JITInfoLogBuffer}
	opts = append(opts, driver.JITInfoLogBufferSizeBytes)
	head := append(extra, 0)
	return driver.LinkCreate(uint32(len(opts)), &opts[0], &head[0])
}

func grownOther(extra []driver.JITOption, vals []uintptr) (driver.LinkState, error) {
	opts := append(extra, driver.JITInfoLogBuffer)
	return driver.LinkCreate(uint32(len(opts)), &opts[0], &vals[0]) // want `&vals\[0\] panics if vals is empty; test len\(vals\) first`
}

// closure passes slices that the function around it has tested.
func closure(opts []driver.JITOption, vals []uintptr) (l driver.LinkState, err error) {
	if len(opts) == 0 || len(opts) != len(vals) {
		return 0, errors.New("bad options")
	}
	load := func() {
		l, err = driver.LinkCreate(uint32(len(opts)), &opts[0], &vals[0])
	}
	load()
	return l, err
}

// counted makes opts with a length it has tested.
func counted(n uint32, vals []uintptr) (driver.LinkState, error) {
	if n == 0 || len(vals) == 0 {
		return 0, nil
	}
	opts := make([]driver.JITOption, n)
	return driver.LinkCreate(n, &opts[0], &vals[0])
}

func options() ([]driver.JITOption, []uintptr) { return nil, nil }

// paired tests one of two slices that come back together.
func paired() (driver.LinkState, error) {
	opts, vals := options()
	if len(opts) > 0 {
		return driver.LinkCreate(uint32(len(opts)), &opts[0], &vals[0])
	}
	return 0, nil
}

func widen(ids []int32) []int32 {
	out := make([]int32, len(ids))
	copy(out, ids)
	return out
}

// helperMade makes ids by a helper that keeps the length of what it is
// given, which was tested.
func helperMade(in []int32) error {
	if len(in) == 0 {
		return errors.New("no ids")
	}
	ids := widen(in)
	values := make([]float64, len(in))
	return driver.Batch(&ids[0], uint32(len(ids)), &values[0])
}

func helperUntested(in []int32) error {
	ids := widen(in)
	values := make([]float64, len(in))
	return driver.Batch(&ids[0], uint32(len(ids)), &values[0]) // want `&ids\[0\] panics if ids is empty; test len\(ids\) first` `&values\[0\] panics if values is empty; test len\(values\) first`
}

// product makes values with the product of two tested lengths.
func product(ids []int32, rows []uintptr) error {
	if len(ids) == 0 || len(rows) == 0 {
		return errors.New("empty")
	}
	values := make([]float64, len(ids)*len(rows))
	return driver.Batch(&ids[0], uint32(len(ids)), &values[0])
}

func productUntested(ids []int32, rows []uintptr) error {
	if len(ids) == 0 {
		return errors.New("empty")
	}
	values := make([]float64, len(ids)*len(rows))
	return driver.Batch(&ids[0], uint32(len(ids)), &values[0]) // want `&values\[0\] panics if values is empty; test len\(values\) first`
}

func check(ids []int32, want int) error {
	if len(ids) != want {
		return errors.New("wrong count")
	}
	return nil
}

// validated tests the length of ids in a helper before using it.
func validated(ids []int32, want int) error {
	if err := check(ids, want); err != nil {
		return err
	}
	values := make([]float64, len(ids))
	return driver.Batch(&ids[0], uint32(len(ids)), &values[0])
}
