package cudagraph

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func updateRun(f func() error) error           { return f() }
func updateSet(*driver.KernelNodeParams) error { return nil }

func updateFixture(t *testing.T) (*Exec, *Node) {
	t.Helper()
	args := []any{int32(11), float64(22)}
	ptrs, err := cudalaunch.PackArgs(args...)
	if err != nil {
		t.Fatal(err)
	}
	owner := &launchArgs{params: ptrs, sizes: argumentSizes(args, len(ptrs)), desc: cudalaunch.Description{Function: 7}}
	params := driver.KernelNodeParams{Func: 7, GridDim: driver.Dims(1, 1, 1), BlockDim: driver.Dims(32, 1, 1), SharedMemBytes: 16, KernelParams: uintptr(unsafe.Pointer(&ptrs[0]))}
	g := &Graph{raw: 1, retained: []*launchArgs{owner}}
	g.recordKernelNode(2, params, owner)
	n := &Node{raw: 2, graph: g}
	return &Exec{raw: 3, retained: g.launchStorage(), kernelNodes: g.kernelStorage(), kernelOrder: []string{"original"}}, n
}

func TestKernelUpdateCurrentSnapshot(t *testing.T) {
	e, n := updateFixture(t)
	original := n.graph.kernelStorage()[n.raw]
	if err := e.updateKernelNode(n, 9, driver.Dims(4, 1, 1), []any{int32(33), float64(44)}, updateRun, updateCount, updateInfo, updateSet); err != nil {
		t.Fatal(err)
	}
	owner := e.kernelNodes[n.raw].args
	if err := e.updateKernelNode(n, 0, driver.Dims(5, 1, 1), nil, updateRun, updateCount, updateInfo, updateSet); err != nil {
		t.Fatal(err)
	}
	got := e.kernelNodes[n.raw]
	if got.params.Func != 9 || got.params.GridDim.X != 5 || got.params.BlockDim.X != 32 || got.params.SharedMemBytes != 16 || got.args != owner || *(*int32)(got.args.params[0]) != 33 || *(*float64)(got.args.params[1]) != 44 {
		t.Fatalf("lost updated executable state: %+v", got)
	}
	if n.graph.kernelStorage()[n.raw] != original {
		t.Fatal("exec update changed original graph node")
	}
	for i := 0; i < 1000; i++ {
		if err := e.updateKernelNode(n, 0, driver.Dim3{}, []any{int32(i), float64(i)}, updateRun, updateCount, updateInfo, updateSet); err != nil {
			t.Fatal(err)
		}
	}
	if len(e.kernelNodes) != 1 || len(e.retained) != 1 {
		t.Fatalf("unbounded retention: %d current, %d original", len(e.kernelNodes), len(e.retained))
	}
}

func TestKernelUpdateFailureAndCommittedRestore(t *testing.T) {
	e, n := updateFixture(t)
	old := e.kernelNodes[n.raw]
	rejected := errors.New("driver rejection")
	err := e.updateKernelNode(n, 9, driver.Dim3{}, []any{int32(99), float64(100)}, updateRun, updateCount, updateInfo, func(*driver.KernelNodeParams) error { return rejected })
	if !errors.Is(err, rejected) || e.kernelNodes[n.raw] != old || len(e.kernelOrder) != 1 {
		t.Fatal("driver rejection changed ownership or order", err)
	}
	restore := errors.New("context restoration")
	run := func(f func() error) error { return errors.Join(f(), restore) }
	err = e.updateKernelNode(n, 9, driver.Dim3{}, []any{int32(99), float64(100)}, run, updateCount, updateInfo, updateSet)
	got := e.kernelNodes[n.raw]
	if !errors.Is(err, restore) || got.args == old.args || got.params.Func != 9 || *(*int32)(got.args.params[0]) != 99 || e.kernelOrder != nil {
		t.Fatal("restoration failure lost committed ownership", err)
	}
}

func updateInfo(_ driver.Function, i uint) (uint, uint, error) {
	if i == 0 {
		return 0, 4, nil
	}
	return 8, 8, nil
}
func updateCount(driver.Function) (uint, error) { return 2, nil }

func TestKernelUpdateParameterPointees(t *testing.T) {
	e, n := updateFixture(t)
	old := e.kernelNodes[n.raw]
	value := float64(99)
	update := KernelNodeUpdate{Field: KernelNodeFieldParam, Offset: 8, Size: 8, Value: unsafe.Pointer(&value)}
	reject := errors.New("rejected")
	if err := e.applyKernelParam(n, update, updateRun, updateCount, updateInfo, func(*driver.KernelNodeParams) error { return reject }); !errors.Is(err, reject) {
		t.Fatal(err)
	}
	if e.kernelNodes[n.raw] != old || *(*float64)(old.args.params[1]) != 22 {
		t.Fatal("failed update mutated old argument bytes")
	}
	if err := e.applyKernelParam(n, update, updateRun, updateCount, updateInfo, updateSet); err != nil {
		t.Fatal(err)
	}
	got := e.kernelNodes[n.raw]
	if *(*int32)(got.args.params[0]) != 11 || *(*float64)(got.args.params[1]) != 99 || *(*float64)(old.args.params[1]) != 22 {
		t.Fatal("packed offset corrupted pointer array or old storage")
	}
	for _, p := range got.args.params {
		if uintptr(p)%16 != 0 {
			t.Fatal("clone not 16-byte aligned")
		}
	}
	value = 100
	restore := errors.New("restore")
	if err := e.applyKernelParam(n, update, func(f func() error) error { return errors.Join(f(), restore) }, updateCount, updateInfo, updateSet); !errors.Is(err, restore) {
		t.Fatal(err)
	}
	if *(*float64)(e.kernelNodes[n.raw].args.params[1]) != 100 {
		t.Fatal("restoration failure lost parameter update")
	}
}

func TestKernelUpdateParameterRefusals(t *testing.T) {
	value := uint64(1)
	tests := []struct {
		name         string
		offset, size uintptr
		change       func(*Exec, *Node)
		info         func(driver.Function, uint) (uint, uint, error)
	}{
		{name: "padding", offset: 4, size: 1},
		{name: "cross_parameter", offset: 3, size: 3},
		{name: "overflow", offset: ^uintptr(0), size: 8},
		{name: "oversized", offset: 0, size: 65537},
		{name: "empty", offset: 0, size: 0},
		{name: "unknown_raw", offset: 0, size: 4, change: func(e *Exec, n *Node) { s := e.kernelNodes[n.raw]; s.args = nil; e.kernelNodes[n.raw] = s }},
		{name: "wrong_pointer_array", offset: 0, size: 4, change: func(e *Exec, n *Node) { s := e.kernelNodes[n.raw]; s.params.KernelParams = 1; e.kernelNodes[n.raw] = s }},
		{name: "size_mismatch", offset: 0, size: 4, info: func(driver.Function, uint) (uint, uint, error) { return 0, 8, nil }},
		{name: "overlap", offset: 0, size: 4, info: func(_ driver.Function, i uint) (uint, uint, error) {
			if i == 0 {
				return 0, 4, nil
			}
			return 2, 8, nil
		}},
		{name: "native_oversize", offset: 0, size: 4, info: func(driver.Function, uint) (uint, uint, error) { return 0, 65537, nil }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, n := updateFixture(t)
			if tt.change != nil {
				tt.change(e, n)
			}
			old := e.kernelNodes[n.raw]
			info := tt.info
			if info == nil {
				info = updateInfo
			}
			calls := 0
			err := e.applyKernelParam(n, KernelNodeUpdate{Offset: tt.offset, Size: tt.size, Value: unsafe.Pointer(&value)}, updateRun, updateCount, info, func(*driver.KernelNodeParams) error { calls++; return nil })
			if err == nil || calls != 0 || e.kernelNodes[n.raw] != old {
				t.Fatal("invalid layout reached setter or changed snapshot", err, calls)
			}
		})
	}
}

func TestKernelUpdateArgumentBounds(t *testing.T) {
	e, n := updateFixture(t)
	for _, args := range [][]any{make([]any, 129), {struct{ Bytes [65537]byte }{}}} {
		calls := 0
		if err := e.updateKernelNode(n, 0, driver.Dim3{}, args, updateRun, updateCount, updateInfo, func(*driver.KernelNodeParams) error { calls++; return nil }); err == nil || calls != 0 {
			t.Fatal("unbounded argument storage accepted", err)
		}
	}
}

func TestKernelUpdateGraphSnapshotAndDestroy(t *testing.T) {
	e, n := updateFixture(t)
	old := e.kernelNodes[n.raw]
	n.graph.recordKernelNode(n.raw, driver.KernelNodeParams{Func: 99}, nil)
	if e.kernelNodes[n.raw] != old {
		t.Fatal("graph update mutated instantiated snapshot")
	}
	failed := errors.New("destroy rejected")
	if err := e.destroy(updateRun, func() error { return failed }); !errors.Is(err, failed) || e.kernelNodes[n.raw] != old {
		t.Fatal("failed destroy lost storage", err)
	}
	restore := errors.New("restore")
	if err := e.destroy(func(f func() error) error { return errors.Join(f(), restore) }, func() error { return nil }); !errors.Is(err, restore) || e.raw != 0 || e.kernelNodes != nil || e.retained != nil {
		t.Fatal("committed destroy retained storage", err)
	}
}

func TestKernelUpdateWholeGraphCommit(t *testing.T) {
	e, n := updateFixture(t)
	old := e.kernelNodes[n.raw]
	next := storageOwner(99)
	next.recordKernelNode(4, driver.KernelNodeParams{Func: 99}, nil)
	reject := errors.New("reject")
	if err := e.updateGraph(next, updateRun, func(*driver.GraphExecUpdateResultInfo) error { return reject }); !errors.Is(err, reject) || e.kernelNodes[n.raw] != old || len(e.kernelOrder) != 1 {
		t.Fatal("failed update dropped original storage", err)
	}
	restore := errors.New("restore")
	if err := e.updateGraph(next, func(f func() error) error { return errors.Join(f(), restore) }, func(*driver.GraphExecUpdateResultInfo) error { return nil }); !errors.Is(err, restore) || len(e.retained) != 1 || e.retained[0].desc.Function != 99 || e.kernelNodes != nil || e.checkKernelNode(n) != nil || e.checkKernelNode(&Node{raw: 4}) == nil || e.kernelOrder != nil {
		t.Fatal("committed whole graph update lost storage", err)
	}
}

func TestKernelUpdatePartialAndCount(t *testing.T) {
	e, n := updateFixture(t)
	value := uint8(12)
	update := KernelNodeUpdate{Offset: 0, Size: 1, Value: unsafe.Pointer(&value)}
	if err := e.applyKernelParam(n, update, updateRun, func(driver.Function) (uint, error) { return 1, nil }, updateInfo, updateSet); err == nil {
		t.Fatal("argument count mismatch accepted")
	}
	if err := e.applyKernelParam(n, update, updateRun, updateCount, updateInfo, updateSet); err != nil {
		t.Fatal(err)
	}
	if *(*int32)(e.kernelNodes[n.raw].args.params[0]) != 12 {
		t.Fatal("contained partial update failed")
	}
	update.Value = nil
	if err := e.applyKernelParam(n, update, updateRun, updateCount, updateInfo, updateSet); err == nil {
		t.Fatal("nil value accepted")
	}
}

func TestKernelUpdateCommittedGraphRetention(t *testing.T) {
	e, n := updateFixture(t)
	owner := e.kernelNodes[n.raw].args
	n.graph.recordKernelNode(4, e.kernelNodes[n.raw].params, nil)
	if n.graph.kernelStorage()[4].args != owner {
		t.Fatal("success metadata failed to capture typed owner")
	}
	n.graph.release(owner)
	if len(n.graph.retained) != 1 {
		t.Fatal("committed add lost owner on restoration failure")
	}
}

func TestKernelUpdateSignaturePreflight(t *testing.T) {
	queryError := errors.New("metadata unavailable")
	tests := []struct {
		name  string
		fn    driver.Function
		args  []any
		count func(driver.Function) (uint, error)
		info  func(driver.Function, uint) (uint, uint, error)
	}{
		{name: "unchanged_four_parameter_function", args: []any{uint8(1)}, count: func(driver.Function) (uint, error) { return 4, nil }, info: updateInfo},
		{name: "undersized_parameter", args: []any{uint8(1), float64(2)}, count: updateCount, info: updateInfo},
		{name: "hotswap_count", fn: 9, count: func(driver.Function) (uint, error) { return 3, nil }, info: updateInfo},
		{name: "hotswap_size", fn: 9, count: updateCount, info: func(_ driver.Function, i uint) (uint, uint, error) {
			if i == 0 {
				return 0, 8, nil
			}
			return 8, 8, nil
		}},
		{name: "count_query_error", args: []any{int32(1), float64(2)}, count: func(driver.Function) (uint, error) { return 0, queryError }, info: updateInfo},
		{name: "parameter_query_error", args: []any{int32(1), float64(2)}, count: updateCount, info: func(driver.Function, uint) (uint, uint, error) { return 0, 0, queryError }},
		{name: "overlapping_metadata", args: []any{int32(1), float64(2)}, count: updateCount, info: func(_ driver.Function, i uint) (uint, uint, error) {
			if i == 0 {
				return 0, 4, nil
			}
			return 2, 8, nil
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, n := updateFixture(t)
			old := e.kernelNodes[n.raw]
			calls := 0
			err := e.updateKernelNode(n, tt.fn, driver.Dim3{}, tt.args, updateRun, tt.count, tt.info, func(*driver.KernelNodeParams) error { calls++; return nil })
			if err == nil || calls != 0 || e.kernelNodes[n.raw] != old || len(e.kernelOrder) != 1 {
				t.Fatal("signature mismatch reached setter or changed state", err, calls)
			}
			if tt.name == "count_query_error" || tt.name == "parameter_query_error" {
				if !errors.Is(err, queryError) {
					t.Fatal("query error identity lost", err)
				}
			}
		})
	}
}

func TestKernelUpdateHotSwapSignatureAndZero(t *testing.T) {
	e, n := updateFixture(t)
	owner := e.kernelNodes[n.raw].args
	queried := driver.Function(0)
	count := func(fn driver.Function) (uint, error) { queried = fn; return 2, nil }
	if err := e.updateKernelNode(n, 9, driver.Dim3{}, nil, updateRun, count, updateInfo, updateSet); err != nil {
		t.Fatal(err)
	}
	if queried != 9 || e.kernelNodes[n.raw].args != owner {
		t.Fatal("new function ABI not checked or retained args lost")
	}
	zero := &Exec{raw: 3, kernelNodes: map[driver.GraphNode]kernelNodeState{2: {params: driver.KernelNodeParams{Func: 7, GridDim: driver.Dims(1, 1, 1)}}}}
	calls := 0
	if err := zero.updateKernelNode(n, 9, driver.Dim3{}, nil, updateRun, func(driver.Function) (uint, error) { return 0, nil }, func(driver.Function, uint) (uint, uint, error) {
		t.Fatal("zero signature queried parameter")
		return 0, 0, nil
	}, func(*driver.KernelNodeParams) error { calls++; return nil }); err != nil || calls != 1 {
		t.Fatal("genuine zero-parameter hot swap refused", err, calls)
	}
}

func TestKernelUpdateOriginalIdentityAfterWholeUpdate(t *testing.T) {
	e, n := updateFixture(t)
	next := storageOwner(99)
	next.recordKernelNode(4, driver.KernelNodeParams{Func: 99}, nil)
	if err := e.updateGraph(next, updateRun, func(*driver.GraphExecUpdateResultInfo) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if _, err := e.kernelState(n); err == nil {
		t.Fatal("whole update guessed original node correspondence")
	}
	replacement := &Node{raw: 4, graph: next}
	if err := e.SetKernelNodeParams(replacement, &driver.KernelNodeParams{Func: 99}); err == nil {
		t.Fatal("replacement graph node reached native setter")
	}
	if err := e.checkKernelNode(n); err != nil {
		t.Fatal("original instantiated identity forgotten", err)
	}
	full := driver.KernelNodeParams{Func: 99, GridDim: driver.Dims(1, 1, 1)}
	if err := e.setKernelNodeParams(n, full, nil, updateRun, updateSet); err != nil {
		t.Fatal(err)
	}
	got, err := e.kernelState(n)
	if err != nil || got.params.Func != 99 {
		t.Fatal("complete original-node setter could not recover helper metadata", err)
	}
	if _, ok := e.kernelOrigins[4]; ok {
		t.Fatal("original identity set adopted replacement handle")
	}
}
