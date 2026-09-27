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

// countAfter passes len(deps) after &deps[0]; it counts deps, not params.
func countAfter(g driver.Graph, deps []driver.GraphNode, params []driver.KernelNodeParams) error {
	if len(deps) == 0 || len(params) == 0 {
		return nil
	}
	_, err := g.AddKernelNode(&deps[0], uint(len(deps)), &params[0])
	return err
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

func widen(deps []driver.GraphNode) []driver.GraphNode {
	out := make([]driver.GraphNode, len(deps))
	copy(out, deps)
	return out
}

// helperMade makes deps by a helper that keeps the length of what it is
// given, which was tested.
func helperMade(g driver.Graph, in []driver.GraphNode) error {
	if len(in) == 0 {
		return errors.New("no dependencies")
	}
	deps := widen(in)
	params := make([]driver.KernelNodeParams, len(in))
	_, err := g.AddKernelNode(&deps[0], uint(len(deps)), &params[0])
	return err
}

func helperUntested(g driver.Graph, in []driver.GraphNode) error {
	deps := widen(in)
	params := make([]driver.KernelNodeParams, len(in))
	_, err := g.AddKernelNode(&deps[0], uint(len(deps)), &params[0]) // want `&deps\[0\] panics if deps is empty; test len\(deps\) first` `&params\[0\] panics if params is empty; test len\(params\) first`
	return err
}

// product makes params with the product of two tested lengths.
func product(g driver.Graph, deps []driver.GraphNode, rows []uintptr) error {
	if len(deps) == 0 || len(rows) == 0 {
		return errors.New("empty")
	}
	params := make([]driver.KernelNodeParams, len(deps)*len(rows))
	_, err := g.AddKernelNode(&deps[0], uint(len(deps)), &params[0])
	return err
}

func productUntested(g driver.Graph, deps []driver.GraphNode, rows []uintptr) error {
	if len(deps) == 0 {
		return errors.New("empty")
	}
	params := make([]driver.KernelNodeParams, len(deps)*len(rows))
	_, err := g.AddKernelNode(&deps[0], uint(len(deps)), &params[0]) // want `&params\[0\] panics if params is empty; test len\(params\) first`
	return err
}

func check(deps []driver.GraphNode, want int) error {
	if len(deps) != want {
		return errors.New("wrong count")
	}
	return nil
}

// validated tests the length of deps in a helper before using it.
func validated(g driver.Graph, deps []driver.GraphNode, want int) error {
	if err := check(deps, want); err != nil {
		return err
	}
	params := make([]driver.KernelNodeParams, len(deps))
	_, err := g.AddKernelNode(&deps[0], uint(len(deps)), &params[0])
	return err
}
