package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
)

var filterProbeError = errors.New("injected Filter operation failure")

type filterProbe struct {
	storage                                       map[driver.DevicePtr][]float32
	allocations, frees, launches, compiles, reads int
	closes, syncs, releases                       int
	pending                                       bool
	lengths                                       []int
	sources                                       []FusedKernel
	failAlloc, failCompile, failLaunch, failRead  int
	failSync, failFree                            bool
	damageCount, damagePrefix, damageStatus       bool
	onLaunch                                      func()
}

func filterHostOnly(t *testing.T) {
	t.Helper()
	check := func() {
		if driver.Loaded() || nvrtc.Loaded() {
			t.Error("Filter host control loaded a vendor library")
		}
	}
	check()
	t.Cleanup(check)
}

func (p *filterProbe) ops() executionOps {
	return executionOps{
		alloc: func(n int) (*executionBuffer, error) {
			if n <= 0 {
				return nil, fmt.Errorf("allocation used unresolved or empty extent %d", n)
			}
			p.allocations++
			p.lengths = append(p.lengths, n)
			ptr := driver.DevicePtr(1000000 + p.allocations*8192)
			data := make([]float32, n)
			for i := range data {
				data[i] = math.Float32frombits(0x7fc01234)
			}
			p.storage[ptr] = data
			b := &executionBuffer{ptr: ptr, check: func() error {
				if p.pending {
					return fmt.Errorf("check before completion")
				}
				for _, v := range data {
					if math.Float32bits(v) == 0x7fc01234 {
						return ErrOutputUnwritten
					}
				}
				return nil
			}, read: func(out []float32) error {
				p.reads++
				if p.pending || len(out) != n {
					return fmt.Errorf("read before completion or wrong extent")
				}
				if p.reads == p.failRead {
					return filterProbeError
				}
				copy(out, data)
				return nil
			}, free: func() error {
				if p.pending {
					return fmt.Errorf("free before completion")
				}
				if p.failFree {
					return filterProbeError
				}
				p.frees++
				delete(p.storage, ptr)
				return nil
			}}
			if p.allocations == p.failAlloc {
				return b, filterProbeError
			}
			return b, nil
		},
		compile: func(f FusedKernel) (*executionKernel, error) {
			p.compiles++
			p.sources = append(p.sources, f)
			k := &executionKernel{close: func() error { p.closes++; return nil }, launch: func(ptrs []driver.DevicePtr, n int) error {
				p.pending = true
				p.launches++
				if p.onLaunch != nil {
					p.onLaunch()
				}
				if p.launches == p.failLaunch {
					return filterProbeError
				}
				if len(ptrs) != len(f.Inputs)+len(f.Outputs) {
					return fmt.Errorf("Filter pointer ABI")
				}
				var inputs [][]float32
				for _, ptr := range ptrs[:len(f.Inputs)] {
					inputs = append(inputs, p.storage[ptr])
				}
				outputs, err := filterSourceModel(f, inputs, n)
				if err != nil {
					return err
				}
				if len(outputs) == 4 {
					if p.damageCount {
						outputs[3][1]++
					}
					if p.damageStatus {
						outputs[3][0] = 2
					}
					if p.damagePrefix {
						outputs[2][n-1]++
					}
				}
				for i, ptr := range ptrs[len(f.Inputs):] {
					if len(p.storage[ptr]) != len(outputs[i]) {
						return fmt.Errorf("Filter output allocation/domain mismatch")
					}
					copy(p.storage[ptr], outputs[i])
				}
				return nil
			}}
			if p.compiles == p.failCompile {
				return k, filterProbeError
			}
			return k, nil
		},
		sync: func(ctx context.Context) error {
			p.syncs++
			if ctx != nil && ctx.Err() != nil {
				return ctx.Err()
			}
			if p.failSync {
				return filterProbeError
			}
			p.pending = false
			return nil
		},
	}
}

func prepareFilter(t *testing.T, predicate []float32, consumer string, eager bool) (*MaterializedPlan, *executionPlan, *filterProbe, map[string]executionInput, []float32) {
	t.Helper()
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{len(predicate)}, cudair.Float32)
	b := NewLazyArray(ctx, []int{len(predicate)}, cudair.Float32)
	p := NewLazyArray(ctx, []int{len(predicate)}, cudair.Float32)
	f := a.Add(b).Mul(a).Filter(p.Map("neg").Map("neg"))
	root := f
	data := map[*LazyNode][]float32{a.Node: make([]float32, len(predicate)), b.Node: make([]float32, len(predicate)), p.Node: slices.Clone(predicate)}
	var want []float32
	for i := range predicate {
		data[a.Node][i], data[b.Node][i] = float32(i+1)/4, float32(2*i+3)/8
		if predicate[i] == 1 {
			// Independent stable append: no prefix scan or production graph helper.
			want = append(want, (data[a.Node][i]+data[b.Node][i])*data[a.Node][i])
		}
	}
	var materialized *MaterializedPlan
	var err error
	switch consumer {
	case "pointwise":
		s := NewLazyArray(ctx, []int{1}, cudair.Float32)
		data[s.Node] = []float32{7}
		root = f.Map("neg").Add(s)
		for i := range want {
			want[i] = -want[i] + 7
		}
	case "scan":
		materialized, err = f.Scan("sum")
		want = ordinaryScan(want)
	case "sum":
		materialized, err = f.Sum()
		var total float64
		for _, v := range want {
			total += float64(v)
		}
		want = []float32{float32(total)}
	}
	if materialized == nil && err == nil {
		materialized, err = root.Materialize()
	}
	if err != nil {
		t.Fatal(err)
	}
	plan, err := snapshotExecution(materialized, eager)
	if err != nil {
		t.Fatal(err)
	}
	probe := &filterProbe{storage: make(map[driver.DevicePtr][]float32)}
	dev := new(cuda.Device)
	bindings := make(map[string]executionInput)
	for node, values := range data {
		ptr := driver.DevicePtr(node.ID * 8192)
		probe.storage[ptr] = slices.Clone(values)
		bindings[node.Name()] = executionInput{description: executionBindingDescription{shape: slices.Clone(node.Shape), dtype: node.DType, length: len(values), ptr: uint64(ptr), bytes: uint64(len(values)) * 4, device: dev}, borrow: func(*cuda.Device) (driver.DevicePtr, func(), error) {
			return ptr, func() { probe.releases++ }, nil
		}}
	}
	return materialized, plan, probe, bindings, want
}

func executeFilterProbe(ctx context.Context, plan *executionPlan, probe *filterProbe, bindings map[string]executionInput) (*Execution, error) {
	var dev *cuda.Device
	for _, b := range bindings {
		dev = b.description.device
		break
	}
	return executePlanInputs(ctx, plan, dev, bindings, probe.ops())
}

func TestFilterCountValuesAndConsumers(t *testing.T) {
	filterHostOnly(t)
	for _, n := range []int{0, 1, 31, 32, 33, 129, 1003} {
		for _, mode := range []string{"none", "all", "alternating"} {
			for _, consumer := range []string{"filter", "pointwise", "scan", "sum"} {
				t.Run(fmt.Sprintf("%d/%s/%s", n, mode, consumer), func(t *testing.T) {
					predicate := make([]float32, n)
					for i := range predicate {
						if mode == "all" || mode == "alternating" && i%2 == 0 {
							predicate[i] = 1
						}
					}
					public, plan, probe, bindings, want := prepareFilter(t, predicate, consumer, false)
					if consumer != "sum" && (public.AllocatedBytes != -1 || public.RootNode.Shape[0] != -1) {
						t.Fatal("Filter planning guessed an output extent")
					}
					e, err := executeFilterProbe(context.Background(), plan, probe, bindings)
					if e != nil {
						t.Cleanup(func() {
							if err := e.Close(); err != nil {
								t.Error(err)
							}
						})
					}
					if err != nil {
						t.Fatal(err)
					}
					got, err := e.Readback(context.Background())
					if err != nil || !slices.Equal(got, want) || !slices.Equal(e.Shape(), []int{len(want)}) {
						t.Fatal(got, want, e.Shape(), err)
					}
					if n > 0 {
						if !slices.Equal(probe.lengths[:4], []int{n, n, n, 2}) || probe.syncs < 1 || len(probe.sources[0].Inputs) != 3 {
							t.Fatal("upstream/predicate fusion or count scratch", probe.lengths, probe.syncs)
						}
						selected := 0
						for _, v := range predicate {
							if v == 1 {
								selected++
							}
						}
						if selected > 0 && probe.lengths[4] != selected {
							t.Fatal("boundary allocation did not use actual count", probe.lengths)
						}
						if selected == 0 && consumer != "sum" && len(probe.lengths) != 4 {
							t.Fatal("zero selection allocated an output", probe.lengths)
						}
					}
					if err := e.Close(); err != nil || probe.frees != probe.allocations || probe.releases != len(bindings) && n != 0 {
						t.Fatal("Filter cleanup custody", err, probe)
					}
				})
			}
		}
	}
}

func TestFilterInvalidPredicatesAndCountEvidence(t *testing.T) {
	filterHostOnly(t)
	for _, value := range []float32{-1, 0.5, 2, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		_, plan, probe, bindings, _ := prepareFilter(t, []float32{1, value, 0}, "filter", false)
		e, err := executeFilterProbe(context.Background(), plan, probe, bindings)
		if e == nil || err == nil || probe.allocations != 4 {
			t.Fatal("invalid predicate allocated dynamic output", value, e, err, probe.allocations)
		}
		if _, err := e.Readback(nil); err == nil || e.Shape() != nil {
			t.Fatal("invalid count published a result")
		}
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
	}
	for _, damage := range []string{"count", "prefix", "status"} {
		_, plan, probe, bindings, _ := prepareFilter(t, []float32{1, 0, 1}, "filter", false)
		probe.damageCount, probe.damagePrefix, probe.damageStatus = damage == "count", damage == "prefix", damage == "status"
		e, err := executeFilterProbe(nil, plan, probe, bindings)
		if e == nil || err == nil || probe.allocations != 4 {
			t.Fatal(damage, e, err, probe.allocations)
		}
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestFilterFailureCustody(t *testing.T) {
	filterHostOnly(t)
	for _, fault := range []string{"scratch-allocation", "boundary-allocation", "downstream-allocation", "count-compile", "scatter-compile", "downstream-compile", "count-launch", "scatter-launch", "downstream-launch", "count-read", "count-sync", "cancel", "cancel-scatter", "cancel-downstream"} {
		t.Run(fault, func(t *testing.T) {
			_, plan, probe, bindings, _ := prepareFilter(t, []float32{1, 0, 1}, "pointwise", false)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch fault {
			case "scratch-allocation":
				probe.failAlloc = 2
			case "boundary-allocation":
				probe.failAlloc = 5
			case "downstream-allocation":
				probe.failAlloc = 6
			case "count-compile":
				probe.failCompile = 1
			case "scatter-compile":
				probe.failCompile = 2
			case "downstream-compile":
				probe.failCompile = 3
			case "count-launch":
				probe.failLaunch = 1
			case "scatter-launch":
				probe.failLaunch = 2
			case "downstream-launch":
				probe.failLaunch = 3
			case "count-read":
				probe.failRead = 1
			case "count-sync":
				probe.failSync = true
			case "cancel":
				probe.onLaunch = cancel
			case "cancel-scatter", "cancel-downstream":
				at := 2
				if fault == "cancel-downstream" {
					at = 3
				}
				probe.onLaunch = func() {
					if probe.launches == at {
						cancel()
					}
				}
			}
			e, err := executeFilterProbe(ctx, plan, probe, bindings)
			if e == nil || err == nil {
				t.Fatal("lost returned owner/error", e, err)
			}
			if probe.pending {
				probe.failSync = true
				before := probe.frees
				if err := e.Close(); err == nil || probe.frees != before || probe.releases != 0 {
					t.Fatal("failed drain released ownership")
				}
			}
			probe.failSync = false
			if err := e.Close(); err != nil || probe.frees != probe.allocations || probe.releases != len(bindings) {
				t.Fatal(err, probe)
			}
			if _, err := e.Readback(nil); err == nil {
				t.Fatal("failed execution admitted output")
			}
		})
	}
}

func TestFilterEagerAndCycleBoundary(t *testing.T) {
	filterHostOnly(t)
	_, plan, probe, bindings, want := prepareFilter(t, []float32{1, 0, 1, 1}, "pointwise", true)
	e, err := executeFilterProbe(nil, plan, probe, bindings)
	if e != nil {
		t.Cleanup(func() {
			if err := e.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Readback(nil)
	if err != nil || !slices.Equal(got, want) {
		t.Fatal(got, want, err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{2}, cudair.Float32)
	p := NewLazyArray(ctx, []int{5}, cudair.Float32)
	f := a.Cycle([]int{5}).Filter(p)
	public, err := f.Materialize()
	if err != nil {
		t.Fatal(err)
	}
	plan, err = snapshotExecution(public, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.steps) != 2 || plan.steps[0].filter || !plan.steps[1].filter || plan.steps[1].inputs[0].Op != "cycle" {
		t.Fatal("Filter bypassed the Cycle owner", plan.steps)
	}
	probe = &filterProbe{storage: map[driver.DevicePtr][]float32{8192: {2, 9}, 16384: {1, 0, 1, 0, 1}}}
	dev := new(cuda.Device)
	bindings = map[string]executionInput{}
	for i, n := range []*LazyNode{a.Node, p.Node} {
		ptr := driver.DevicePtr((i + 1) * 8192)
		bindings[n.Name()] = executionInput{description: executionBindingDescription{shape: n.Shape, dtype: cudair.Float32, length: n.Shape[0], ptr: uint64(ptr), bytes: uint64(n.Shape[0]) * 4, device: dev}, borrow: func(*cuda.Device) (driver.DevicePtr, func(), error) { return ptr, func() { probe.releases++ }, nil }}
	}
	e, err = executeFilterProbe(nil, plan, probe, bindings)
	if e != nil {
		defer e.Close()
	}
	if err != nil {
		t.Fatal(err)
	}
	got, err = e.Readback(nil)
	if err != nil || !slices.Equal(got, []float32{2, 2, 2}) || e.Stats().Allocations != 6 {
		t.Fatal(got, err, e.Stats())
	}
}

func TestFilterAdmissionAndDistinctOperations(t *testing.T) {
	filterHostOnly(t)
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{3}, cudair.Float32)
	p := NewLazyArray(ctx, []int{3}, cudair.Float32)
	f := a.Filter(p)
	if !unresolvedShape(f.Shape) || !slices.Equal(a.Where(p, a).Shape, []int{3}) || !slices.Equal(a.Keep(p).Shape, []int{3}) {
		t.Fatal("Filter/Where/Keep aliased")
	}
	if _, err := f.Add(a).Materialize(); err == nil {
		t.Fatal("ambiguous fixed-vector consumer admitted")
	}
	if _, err := f.Reshape([]int{3}).Materialize(); err == nil {
		t.Fatal("unresolved reshape admitted")
	}
	for _, shape := range [][]int{{2, 3}, {-1}, {maxFilterElements + 1}, {math.MaxInt}} {
		b := NewLazyArray(ctx, shape, cudair.Float32)
		if _, err := b.Filter(b).Materialize(); err == nil {
			t.Fatal("invalid Filter shape", shape)
		}
	}
	for _, op := range []*LazyArray{a.Where(p, a), a.Keep(p)} {
		plan, err := op.Materialize()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := snapshotExecution(plan, false); err == nil {
			t.Fatal("Where/Keep silently became Filter")
		}
	}
	plan, err := f.Add(a.Filter(p)).Materialize()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := snapshotExecution(plan, false); err == nil {
		t.Fatal("multiple unresolved Filters admitted")
	}
	plan, err = f.Materialize()
	if err != nil {
		t.Fatal(err)
	}
	if owner, err := plan.Execute(nil, nil, nil); err == nil || owner != nil {
		t.Fatal("nonempty public execution fabricated device work")
	}
	if _, err := filterCount([]float32{1}, []float32{1}, []float32{0, float32(math.Inf(1))}); err == nil {
		t.Fatal("nonfinite count admitted")
	}
	if _, err := filterCount([]float32{1}, []float32{1}, []float32{0, 2}); err == nil {
		t.Fatal("count overflow admitted")
	}
	if !strings.Contains(plan.Regions[len(plan.Regions)-1].Reason, "Filter count") {
		t.Fatal("deferred source reason missing")
	}
}

func TestFilterCycleSharedPredicateBoundary(t *testing.T) {
	filterHostOnly(t)
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{2}, cudair.Float32)
	cycle := a.Cycle([]int{5})
	public, err := cycle.Filter(cycle).Materialize()
	if err != nil {
		t.Fatal(err)
	}
	plan, err := snapshotExecution(public, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.steps) != 2 || plan.steps[0].filter || !plan.steps[1].filter || len(plan.steps[1].inputs) != 1 || plan.steps[1].inputs[0].Op != "cycle" {
		t.Fatal("Filter bypassed adjacent Cycle ownership", plan.steps)
	}
	probe := &filterProbe{storage: map[driver.DevicePtr][]float32{8192: {0, 1}}}
	dev := new(cuda.Device)
	bindings := map[string]executionInput{a.Node.Name(): {
		description: executionBindingDescription{shape: []int{2}, dtype: cudair.Float32, length: 2, ptr: 8192, bytes: 8, device: dev},
		borrow: func(*cuda.Device) (driver.DevicePtr, func(), error) {
			return 8192, func() { probe.releases++ }, nil
		},
	}}
	e, err := executeFilterProbe(nil, plan, probe, bindings)
	if e != nil {
		t.Cleanup(func() {
			if err := e.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Readback(nil)
	if err != nil || !slices.Equal(got, []float32{1, 1}) || e.Stats().Allocations != 6 {
		t.Fatal(got, err, e.Stats())
	}
}
