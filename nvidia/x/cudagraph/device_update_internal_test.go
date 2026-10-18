package cudagraph

import (
	"errors"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

func TestSetKernelNodeLaunchKeepsOwnedStorage(t *testing.T) {
	e, n := updateFixture(t)
	original := n.graph.kernelStorage()[n.raw]
	old := e.kernelNodes[n.raw]
	table := old.params.KernelParams
	var sent []driver.KernelNodeParams
	set := func(p *driver.KernelNodeParams) error { sent = append(sent, *p); return nil }

	if err := e.updateKernel(n, 0, driver.Dim3{}, driver.Dims(64, 2, 1), nil, updateRun, updateCount, updateInfo, set); err != nil {
		t.Fatal(err)
	}
	got := e.kernelNodes[n.raw]
	if got.args != old.args || got.params.KernelParams != table || got.params.KernelParams != uintptr(unsafe.Pointer(&old.args.params[0])) || got.params.Extra != 0 {
		t.Fatalf("block update replaced owned storage: %+v, was %+v", got, old)
	}
	if got.params.BlockDim != driver.Dims(64, 2, 1) || got.params.GridDim != old.params.GridDim || got.params.Func != old.params.Func || got.params.SharedMemBytes != old.params.SharedMemBytes {
		t.Fatalf("block update params = %+v", got.params)
	}
	if len(sent) != 1 || sent[0] != got.params {
		t.Fatalf("native params = %+v, want %+v", sent, got.params)
	}
	if *(*int32)(got.args.params[0]) != 11 || *(*float64)(got.args.params[1]) != 22 {
		t.Fatal("block update changed argument values")
	}

	// Ownership survives: grid-only and parameter updates still accept the node.
	if err := e.updateKernelNode(n, 0, driver.Dims(9, 1, 1), nil, updateRun, updateCount, updateInfo, set); err != nil {
		t.Fatalf("grid update after block update: %v", err)
	}
	got = e.kernelNodes[n.raw]
	if got.params.GridDim != driver.Dims(9, 1, 1) || got.params.BlockDim != driver.Dims(64, 2, 1) || got.args != old.args {
		t.Fatalf("grid update after block update = %+v", got)
	}
	value := float64(99)
	update := KernelNodeUpdate{Field: KernelNodeFieldParam, Offset: 8, Size: 8, Value: unsafe.Pointer(&value)}
	if err := e.applyKernelParam(n, update, updateRun, updateCount, updateInfo, set); err != nil {
		t.Fatalf("parameter update after block update: %v", err)
	}
	if got := e.kernelNodes[n.raw]; got.params.BlockDim != driver.Dims(64, 2, 1) || *(*float64)(got.args.params[1]) != 99 {
		t.Fatalf("parameter update after block update = %+v", got)
	}

	// Identity: the same node handle, one executable entry, original graph untouched.
	if n.raw != 2 || len(e.kernelNodes) != 1 {
		t.Fatalf("node identity changed: raw %v, %d nodes", n.raw, len(e.kernelNodes))
	}
	if _, ok := e.kernelNodes[n.raw]; !ok {
		t.Fatal("node handle no longer keys executable state")
	}
	if n.graph.kernelStorage()[n.raw] != original {
		t.Fatal("exec launch update changed original graph node")
	}
	if e.kernelOrder != nil {
		t.Fatal("launch update kept stale kernel order")
	}
}

func TestSetKernelNodeLaunchGridAndBlock(t *testing.T) {
	e, n := updateFixture(t)
	if err := e.updateKernel(n, 0, driver.Dims(4, 2, 1), driver.Dims(256, 1, 1), nil, updateRun, updateCount, updateInfo, updateSet); err != nil {
		t.Fatal(err)
	}
	got := e.kernelNodes[n.raw].params
	if got.GridDim != driver.Dims(4, 2, 1) || got.BlockDim != driver.Dims(256, 1, 1) {
		t.Fatalf("launch = grid %v block %v", got.GridDim, got.BlockDim)
	}
	if err := e.updateKernel(n, 0, driver.Dims(5, 1, 1), driver.Dim3{}, nil, updateRun, updateCount, updateInfo, updateSet); err != nil {
		t.Fatal(err)
	}
	got = e.kernelNodes[n.raw].params
	if got.GridDim != driver.Dims(5, 1, 1) || got.BlockDim != driver.Dims(256, 1, 1) {
		t.Fatalf("zero block changed block: grid %v block %v", got.GridDim, got.BlockDim)
	}
}

func TestSetKernelNodeLaunchRefusals(t *testing.T) {
	tests := []struct {
		name        string
		grid, block driver.Dim3
		change      func(*Exec, *Node) *Node
		want        string // substring; empty means *LaunchGeometryError
	}{
		{name: "zero_block_extent", block: driver.Dim3{X: 32, Y: 0, Z: 1}},
		{name: "zero_grid_extent", grid: driver.Dim3{X: 0, Y: 1, Z: 1}},
		{name: "block_x", block: driver.Dims(1025, 1, 1)},
		{name: "block_y", block: driver.Dims(1, 1025, 1)},
		{name: "block_z", block: driver.Dims(1, 1, 65)},
		{name: "block_threads", block: driver.Dims(32, 32, 2)},
		{name: "block_threads_overflow", block: driver.Dims(1<<16, 1<<16, 1<<16)},
		{name: "grid_x", grid: driver.Dims(1<<31, 1, 1)},
		{name: "grid_y", grid: driver.Dims(1, 65536, 1)},
		{name: "grid_z", grid: driver.Dims(1, 1, 65536)},
		{name: "unretained", block: driver.Dims(64, 1, 1), want: "not retained", change: func(e *Exec, n *Node) *Node {
			return &Node{raw: 99, graph: n.graph}
		}},
		{name: "raw_storage", block: driver.Dims(64, 1, 1), want: "not owned", change: func(e *Exec, n *Node) *Node {
			s := e.kernelNodes[n.raw]
			s.args = nil
			e.kernelNodes[n.raw] = s
			return n
		}},
		{name: "raw_extra", block: driver.Dims(64, 1, 1), want: "not owned", change: func(e *Exec, n *Node) *Node {
			s := e.kernelNodes[n.raw]
			s.args, s.params.KernelParams, s.params.Extra = nil, 0, 1
			e.kernelNodes[n.raw] = s
			return n
		}},
		{name: "foreign_node", block: driver.Dims(64, 1, 1), want: "originally instantiated", change: func(e *Exec, n *Node) *Node {
			e.kernelOrigins = map[driver.GraphNode]struct{}{}
			return n
		}},
		{name: "closed_node", block: driver.Dims(64, 1, 1), want: "closed", change: func(e *Exec, n *Node) *Node {
			return &Node{}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, n := updateFixture(t)
			if tt.change != nil {
				n = tt.change(e, n)
			}
			before := make(map[driver.GraphNode]kernelNodeState)
			for k, v := range e.kernelNodes {
				before[k] = v
			}
			calls := 0
			err := e.updateKernel(n, 0, tt.grid, tt.block, nil, updateRun, updateCount, updateInfo, func(*driver.KernelNodeParams) error { calls++; return nil })
			if err == nil {
				t.Fatal("refusal accepted")
			}
			if tt.want == "" {
				var ge *LaunchGeometryError
				if !errors.As(err, &ge) {
					t.Fatalf("error = %v, want *LaunchGeometryError", err)
				}
			} else if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want %q", err, tt.want)
			}
			if calls != 0 || len(e.kernelNodes) != len(before) || e.kernelOrder == nil {
				t.Fatalf("refusal reached setter (%d calls) or changed state", calls)
			}
			for k, v := range before {
				if e.kernelNodes[k] != v {
					t.Fatalf("refusal changed node %v: %+v, was %+v", k, e.kernelNodes[k], v)
				}
			}
		})
	}
}

func TestSetKernelNodeLaunchNativeRejection(t *testing.T) {
	e, n := updateFixture(t)
	old := e.kernelNodes[n.raw]
	rejected := errors.New("too many resources requested for launch")
	err := e.updateKernel(n, 0, driver.Dim3{}, driver.Dims(1024, 1, 1), nil, updateRun, updateCount, updateInfo, func(*driver.KernelNodeParams) error { return rejected })
	if !errors.Is(err, rejected) || e.kernelNodes[n.raw] != old || e.kernelOrder == nil {
		t.Fatal("native rejection changed executable state", err)
	}
}

func TestSetKernelNodeLaunchArgsOneCommit(t *testing.T) {
	e, n := updateFixture(t)
	old := e.kernelNodes[n.raw]
	var sent []driver.KernelNodeParams
	set := func(p *driver.KernelNodeParams) error { sent = append(sent, *p); return nil }
	args := []any{int32(33), float64(44)}
	if err := e.updateKernel(n, 0, driver.Dims(8, 1, 1), driver.Dims(128, 1, 1), args, updateRun, updateCount, updateInfo, set); err != nil {
		t.Fatal(err)
	}
	got := e.kernelNodes[n.raw]
	if len(sent) != 1 || sent[0] != got.params {
		t.Fatalf("native calls = %+v, want one carrying %+v", sent, got.params)
	}
	if got.params.GridDim != driver.Dims(8, 1, 1) || got.params.BlockDim != driver.Dims(128, 1, 1) || got.params.Func != old.params.Func || got.params.SharedMemBytes != old.params.SharedMemBytes {
		t.Fatalf("launch params = %+v", got.params)
	}
	if got.args == old.args || got.params.KernelParams != uintptr(unsafe.Pointer(&got.args.params[0])) {
		t.Fatal("arguments not replaced by owned storage")
	}
	if *(*int32)(got.args.params[0]) != 33 || *(*float64)(got.args.params[1]) != 44 {
		t.Fatal("new argument values not committed")
	}
	if got.args.desc.Grid != driver.Dims(8, 1, 1) || got.args.desc.Block != driver.Dims(128, 1, 1) {
		t.Fatalf("owned description = grid %v block %v", got.args.desc.Grid, got.args.desc.Block)
	}
}

// TestSetKernelNodeLaunchArgsRejection checks that a rejected combined update
// leaves the previous grid, block and arguments in place.
func TestSetKernelNodeLaunchArgsRejection(t *testing.T) {
	rejected := errors.New("too many resources requested for launch")
	tests := []struct {
		name        string
		grid, block driver.Dim3
		set         func(*driver.KernelNodeParams) error
		count       func(driver.Function) (uint, error)
		wantCalls   int
	}{
		{name: "native", grid: driver.Dims(8, 1, 1), block: driver.Dims(1024, 1, 1), set: func(*driver.KernelNodeParams) error { return rejected }, count: updateCount, wantCalls: 1},
		{name: "geometry", grid: driver.Dims(8, 1, 1), block: driver.Dims(2048, 1, 1), count: updateCount},
		{name: "argument_count", grid: driver.Dims(8, 1, 1), block: driver.Dims(128, 1, 1), count: func(driver.Function) (uint, error) { return 3, nil }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, n := updateFixture(t)
			old := e.kernelNodes[n.raw]
			calls := 0
			set := func(p *driver.KernelNodeParams) error {
				calls++
				if tt.set != nil {
					return tt.set(p)
				}
				return nil
			}
			err := e.updateKernel(n, 0, tt.grid, tt.block, []any{int32(33), float64(44)}, updateRun, tt.count, updateInfo, set)
			if err == nil {
				t.Fatal("rejected update succeeded")
			}
			if tt.set != nil && !errors.Is(err, rejected) {
				t.Fatalf("error = %v, want %v", err, rejected)
			}
			if calls != tt.wantCalls {
				t.Fatalf("native calls = %d, want %d", calls, tt.wantCalls)
			}
			got := e.kernelNodes[n.raw]
			if got != old || e.kernelOrder == nil {
				t.Fatalf("rejection changed state: %+v, was %+v", got, old)
			}
			if got.params.GridDim != driver.Dims(1, 1, 1) || got.params.BlockDim != driver.Dims(32, 1, 1) {
				t.Fatalf("launch = grid %v block %v, want prior 1x1x1 32x1x1", got.params.GridDim, got.params.BlockDim)
			}
			if *(*int32)(got.args.params[0]) != 11 || *(*float64)(got.args.params[1]) != 22 {
				t.Fatal("rejection changed prior argument values")
			}
		})
	}
}

func TestSetKernelNodeGridGeometry(t *testing.T) {
	for _, grid := range []driver.Dim3{{X: 0, Y: 1, Z: 1}, driver.Dims(1<<31, 1, 1), driver.Dims(1, 65536, 1), driver.Dims(1, 1, 65536)} {
		e, n := updateFixture(t)
		old := e.kernelNodes[n.raw]
		var ge *LaunchGeometryError
		if err := e.SetKernelNodeGrid(n, grid); !errors.As(err, &ge) {
			t.Errorf("SetKernelNodeGrid(%v) = %v, want *LaunchGeometryError", grid, err)
		}
		if e.kernelNodes[n.raw] != old {
			t.Errorf("SetKernelNodeGrid(%v) changed state", grid)
		}
	}
}

func TestCheckLaunchGeometry(t *testing.T) {
	tests := []struct {
		grid, block driver.Dim3
		ok          bool
	}{
		{driver.Dims(1, 1, 1), driver.Dims(1, 1, 1), true},
		{driver.Dims(1<<31-1, 65535, 65535), driver.Dims(1024, 1, 1), true},
		{driver.Dims(1, 1, 1), driver.Dims(1, 1024, 1), true},
		{driver.Dims(1, 1, 1), driver.Dims(16, 1, 64), true},
		{driver.Dims(1, 1, 1), driver.Dims(32, 32, 1), true},
		{driver.Dims(1, 1, 1), driver.Dims(32, 33, 1), false},
		{driver.Dim3{}, driver.Dims(1, 1, 1), false},
		{driver.Dims(1, 1, 1), driver.Dim3{}, false},
	}
	for _, tt := range tests {
		err := CheckLaunchGeometry(tt.grid, tt.block)
		if (err == nil) != tt.ok {
			t.Errorf("CheckLaunchGeometry(%v, %v) = %v, want ok %v", tt.grid, tt.block, err, tt.ok)
		}
	}
}

// driverCommit is one cuGraphExecKernelNodeSetParams call as the driver saw it.
type driverCommit struct {
	exec   driver.GraphExec
	node   driver.GraphNode
	params driver.KernelNodeParams
	a      int32
	b      float64
}

// stubDriver replaces the driver entry points of the Exec update methods for
// the duration of t. Each native set call is recorded, reading the argument
// values through the pointer table as the driver would, and then answered
// with refuse.
func stubDriver(t *testing.T, refuse error) *[]driverCommit {
	t.Helper()
	set, count, info := execKernelNodeSetParams, kernelParamCount, kernelParamInfo
	t.Cleanup(func() { execKernelNodeSetParams, kernelParamCount, kernelParamInfo = set, count, info })
	var commits []driverCommit
	execKernelNodeSetParams = func(g driver.GraphExec, n driver.KernelNode, p *driver.KernelNodeParams) error {
		// KernelParams holds the address of the pointer table; reread the field as a pointer.
		table := unsafe.Slice((*unsafe.Pointer)(*(*unsafe.Pointer)(unsafe.Pointer(&p.KernelParams))), 2)
		commits = append(commits, driverCommit{g, n.GraphNode, *p, *(*int32)(table[0]), *(*float64)(table[1])})
		return refuse
	}
	kernelParamCount = updateCount
	kernelParamInfo = func(fn driver.Function, i uint) (uint, uint, error) { return updateInfo(fn, i) }
	return &commits
}

// TestPublicSetKernelNodeLaunchArgs drives the public SetKernelNodeLaunch with
// arguments down to the cuGraphExecKernelNodeSetParams call.
func TestPublicSetKernelNodeLaunchArgs(t *testing.T) {
	e, n := updateFixture(t)
	old := e.kernelNodes[n.raw]
	commits := stubDriver(t, nil)

	if err := e.SetKernelNodeLaunch(n, driver.Dims(8, 1, 1), driver.Dims(128, 1, 1), int32(33), float64(44)); err != nil {
		t.Fatal(err)
	}
	got, ok := e.kernelNodes[n.raw]
	if !ok || len(e.kernelNodes) != 1 || n.raw != 2 {
		t.Fatalf("node identity changed: raw %v, %d nodes", n.raw, len(e.kernelNodes))
	}
	if len(*commits) != 1 {
		t.Fatalf("driver commits = %d, want 1", len(*commits))
	}
	c := (*commits)[0]
	if c.exec != e.raw || c.node != n.raw {
		t.Fatalf("driver call on exec %v node %v, want %v %v", c.exec, c.node, e.raw, n.raw)
	}
	if c.params.GridDim != driver.Dims(8, 1, 1) || c.params.BlockDim != driver.Dims(128, 1, 1) || c.a != 33 || c.b != 44 {
		t.Fatalf("driver commit = grid %v block %v args %v %v, want 8x1x1 128x1x1 33 44", c.params.GridDim, c.params.BlockDim, c.a, c.b)
	}
	if c.params != got.params || c.params.Func != old.params.Func || c.params.SharedMemBytes != old.params.SharedMemBytes {
		t.Fatalf("driver params %+v, retained %+v", c.params, got.params)
	}
	// The arguments live in storage the Exec owns, and the driver read them there.
	if got.args == nil || got.params.KernelParams != uintptr(unsafe.Pointer(&got.args.params[0])) || got.params.Extra != 0 {
		t.Fatal("committed arguments are not in Exec-owned storage")
	}
	if *(*int32)(got.args.params[0]) != 33 || *(*float64)(got.args.params[1]) != 44 {
		t.Fatal("owned storage does not hold the new arguments")
	}
	if *(*int32)(old.args.params[0]) != 11 || *(*float64)(old.args.params[1]) != 22 {
		t.Fatal("update wrote into the superseded argument storage")
	}

	// A later launch change without arguments keeps that storage and its values.
	owner := got.args
	if err := e.SetKernelNodeLaunch(n, driver.Dims(4, 1, 1), driver.Dims(64, 1, 1)); err != nil {
		t.Fatal(err)
	}
	got = e.kernelNodes[n.raw]
	if got.args != owner || len(*commits) != 2 {
		t.Fatalf("launch update without args replaced owned storage or commit count %d", len(*commits))
	}
	if c := (*commits)[1]; c.params.GridDim != driver.Dims(4, 1, 1) || c.params.BlockDim != driver.Dims(64, 1, 1) || c.a != 33 || c.b != 44 {
		t.Fatalf("second driver commit = %+v", c)
	}
}

// TestPublicSetKernelNodeLaunchArgsRefused checks that a driver refusal of the
// public combined update leaves the prior arguments, grid and block.
func TestPublicSetKernelNodeLaunchArgsRefused(t *testing.T) {
	e, n := updateFixture(t)
	old := e.kernelNodes[n.raw]
	refused := errors.New("too many resources requested for launch")
	commits := stubDriver(t, refused)

	err := e.SetKernelNodeLaunch(n, driver.Dims(8, 1, 1), driver.Dims(1024, 1, 1), int32(33), float64(44))
	if !errors.Is(err, refused) {
		t.Fatalf("error = %v, want %v", err, refused)
	}
	if len(*commits) != 1 {
		t.Fatalf("driver commits = %d, want 1", len(*commits))
	}
	if c := (*commits)[0]; c.params.BlockDim != driver.Dims(1024, 1, 1) || c.a != 33 || c.b != 44 {
		t.Fatalf("refused commit = %+v, want block 1024 args 33 44", c)
	}
	got := e.kernelNodes[n.raw]
	if got != old || got.args != old.args || e.kernelOrder == nil {
		t.Fatalf("refusal changed state: %+v, was %+v", got, old)
	}
	if got.params.GridDim != driver.Dims(1, 1, 1) || got.params.BlockDim != driver.Dims(32, 1, 1) {
		t.Fatalf("launch = grid %v block %v, want prior 1x1x1 32x1x1", got.params.GridDim, got.params.BlockDim)
	}
	if *(*int32)(got.args.params[0]) != 11 || *(*float64)(got.args.params[1]) != 22 {
		t.Fatal("refusal changed prior argument values")
	}
}
