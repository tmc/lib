package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func executionChain(t *testing.T, n int) (*MaterializedPlan, []*LazyNode) {
	t.Helper()
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{n}, cudair.Float32)
	b := NewLazyArray(ctx, []int{n}, cudair.Float32)
	c := NewLazyArray(ctx, []int{n}, cudair.Float32)
	root := a.Map("relu").Add(b.Map("relu")).Mul(c)
	p, err := CompileDeferredGraph(ctx, root.Node)
	if err != nil {
		t.Fatal(err)
	}
	nodes, _, err := executionNodes(p)
	if err != nil {
		t.Fatal(err)
	}
	return p, nodes
}

func TestLazyExecution_RejectsBindingsBeforeAllocation(t *testing.T) {
	_, nodes := executionChain(t, 1003)
	dev := &cuda.Device{}
	valid := map[string]executionBindingDescription{}
	i := 0
	for _, n := range nodes {
		if n.Op == "input" {
			i++
			valid[n.Name()] = executionBindingDescription{shape: []int{1003}, dtype: cudair.Float32, length: 1003, ptr: uint64(i) * 8192, bytes: 4012, device: dev}
		}
	}
	names := make([]string, 0, len(valid))
	for n := range valid {
		names = append(names, n)
	}
	slices.Sort(names)
	cases := []struct {
		name   string
		mutate func(map[string]executionBindingDescription)
	}{
		{"missing", func(b map[string]executionBindingDescription) { delete(b, names[0]) }},
		{"short", func(b map[string]executionBindingDescription) { v := b[names[0]]; v.length--; b[names[0]] = v }},
		{"cross-device", func(b map[string]executionBindingDescription) {
			v := b[names[0]]
			v.device = &cuda.Device{}
			b[names[0]] = v
		}},
		{"dtype", func(b map[string]executionBindingDescription) {
			v := b[names[0]]
			v.dtype = cudair.Float16
			b[names[0]] = v
		}},
		{"shape", func(b map[string]executionBindingDescription) {
			v := b[names[0]]
			v.shape = []int{1, 1003}
			b[names[0]] = v
		}},
		{"alias", func(b map[string]executionBindingDescription) {
			v := b[names[0]]
			v.ptr = b[names[1]].ptr + 4
			b[names[0]] = v
		}},
		{"freed", func(b map[string]executionBindingDescription) { v := b[names[0]]; v.ptr = 0; b[names[0]] = v }},
		{"overflow", func(b map[string]executionBindingDescription) {
			v := b[names[0]]
			v.ptr = math.MaxUint64 - 100
			b[names[0]] = v
		}},
		{"extent", func(b map[string]executionBindingDescription) { v := b[names[0]]; v.bytes--; b[names[0]] = v }},
		{"extra", func(b map[string]executionBindingDescription) { b["unexpected"] = b[names[0]] }},
	}
	if err := validateExecutionBindings(nodes, 1003, dev, valid); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := map[string]executionBindingDescription{}
			for k, v := range valid {
				b[k] = v
			}
			tc.mutate(b)
			if err := validateExecutionBindings(nodes, 1003, dev, b); err == nil {
				t.Fatal("invalid binding admitted before native path")
			}
		})
	}
}

func TestLazyExecution_UnsupportedBeforeDeviceCalls(t *testing.T) {
	for _, op := range []string{"transpose", "gather", "reshape", "compact"} {
		t.Run(op, func(t *testing.T) {
			p, _ := executionChain(t, 1003)
			p.RootNode.Op = op
			if e, err := p.Execute(context.Background(), nil, nil); err == nil || e != nil {
				t.Fatal("unsupported graph admitted")
			}
		})
	}
	p, _ := executionChain(t, 1003)
	p.RootNode.Inputs[0] = p.RootNode
	if _, _, err := executionNodes(p); err == nil {
		t.Fatal("cycle accepted")
	}
}

func TestLazyExecution_ZeroOwnerRefusesAndRegionMembershipExact(t *testing.T) {
	var e Execution
	if err := e.Wait(context.Background()); err == nil {
		t.Fatal("zero owner accepted")
	}
	if _, err := e.Readback(context.Background()); err == nil {
		t.Fatal("zero owner readback accepted")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"empty", "missing-root", "duplicate", "missing-input", "duplicate-input", "nonleaf-input", "unrelated-node"} {
		t.Run(mode, func(t *testing.T) {
			p, _ := executionChain(t, 1003)
			r := p.Regions[0]
			switch mode {
			case "empty":
				r.Nodes = nil
			case "missing-root":
				r.Nodes = r.Nodes[:len(r.Nodes)-1]
			case "duplicate":
				r.Nodes = append(r.Nodes, r.Nodes[0])
			case "missing-input":
				r.Inputs = r.Inputs[1:]
			case "duplicate-input":
				r.Inputs = append(r.Inputs, r.Inputs[0])
			case "nonleaf-input":
				r.Inputs[0] = p.RootNode
			case "unrelated-node":
				replacement := *r.Nodes[0]
				r.Nodes[0] = &replacement
			}
			if _, _, err := executionNodes(p); err == nil {
				t.Fatal("inconsistent region admitted")
			}
		})
	}
}

func TestLazyExecution_EmptyDomainNoDeviceCalls(t *testing.T) {
	plan, nodes := executionChain(t, 0)
	dev := &cuda.Device{}
	b := map[string]executionBindingDescription{}
	for _, n := range nodes {
		if n.Op == "input" {
			b[n.Name()] = executionBindingDescription{shape: []int{0}, dtype: cudair.Float32}
		}
	}
	if err := validateExecutionBindings(nodes, 0, dev, b); err != nil {
		t.Fatal(err)
	}

	bindings := map[string]ExecutionBinding{}
	for _, node := range nodes {
		if node.Op == "input" {
			bindings[node.Name()] = ExecutionBinding{Shape: []int{0}, DType: cudair.Float32}
		}
	}
	e, err := plan.Execute(context.Background(), nil, bindings)
	if err != nil {
		t.Fatal(err)
	}
	out, err := e.Readback(context.Background())
	if err != nil || len(out) != 0 {
		t.Fatal(out, err)
	}
	if e.Stats() != (ExecutionStats{}) || len(e.Receipts()) != 0 {
		t.Fatal("invented device work")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
}

type executionProbe struct {
	allocs, frees, launches, syncs, closed, released           int
	compileErr, allocErr, launchErr, syncErr, copyErr, freeErr error
	cancel                                                     bool
	sources                                                    []FusedKernel
}

func (p *executionProbe) ops() executionOps {
	return executionOps{
		alloc: func(n int) (*executionBuffer, error) {
			if p.allocErr != nil {
				return nil, p.allocErr
			}
			p.allocs++
			return &executionBuffer{ptr: driver.DevicePtr(100000 + p.allocs*8192), read: func(out []float32) error {
				if p.syncs == 0 {
					return errors.New("read without completion")
				}
				return p.copyErr
			}, free: func() error {
				if p.freeErr != nil {
					return p.freeErr
				}
				p.frees++
				return nil
			}}, nil
		},
		compile: func(f FusedKernel) (*executionKernel, error) {
			if p.compileErr != nil {
				return nil, p.compileErr
			}
			p.sources = append(p.sources, f)
			return &executionKernel{launch: func(ptrs []driver.DevicePtr, n int) error {
				if n != 1003 || len(ptrs) != len(f.Inputs)+1 {
					return errors.New("wrong ABI")
				}
				if p.launchErr != nil {
					return p.launchErr
				}
				p.launches++
				return nil
			}, close: func() error { p.closed++; return nil }}, nil
		},
		sync: func(ctx context.Context) error {
			p.syncs++
			if p.cancel && ctx != nil && ctx.Done() != nil {
				return context.Canceled
			}
			return p.syncErr
		},
	}
}
func executionPointers(nodes []*LazyNode) map[*LazyNode]driver.DevicePtr {
	out := map[*LazyNode]driver.DevicePtr{}
	for _, n := range nodes {
		if n.Op == "input" {
			out[n] = driver.DevicePtr(n.ID * 8192)
		}
	}
	return out
}

func TestLazyExecution_RealPathCountersAndCompletion(t *testing.T) {
	_, nodes := executionChain(t, 1003)
	for _, eager := range []bool{false, true} {
		t.Run(fmt.Sprint(eager), func(t *testing.T) {
			probe := &executionProbe{}
			e, err := runExecution(context.Background(), nodes, 1003, executionPointers(nodes), eager, probe.ops(), nil)
			if err != nil {
				t.Fatal(err)
			}
			want := 1
			if eager {
				want = 4
			}
			if probe.allocs != want || probe.launches != want || probe.syncs != 1 || e.Stats() != (ExecutionStats{Allocations: want, Launches: want}) {
				t.Fatalf("actual calls=%+v receipt=%+v", probe, e.Stats())
			}
			if _, err := e.Readback(context.Background()); err != nil {
				t.Fatal(err)
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if probe.frees != want || probe.closed != want || e.Stats().Frees != want {
				t.Fatal("cleanup count differs")
			}
			if !eager {
				for n := range executionPointers(nodes) {
					if !strings.Contains(probe.sources[0].Source, n.Name()+"[idx]") {
						t.Fatalf("actual operand %s omitted", n.Name())
					}
				}
			}
		})
	}
}

func TestLazyExecution_ErrorCleanupAndCancellation(t *testing.T) {
	_, nodes := executionChain(t, 1003)
	sentinel := errors.New("injected original failure")
	for _, stage := range []string{"compile", "allocate", "launch", "synchronize", "copy", "free"} {
		t.Run(stage, func(t *testing.T) {
			p := &executionProbe{}
			switch stage {
			case "compile":
				p.compileErr = sentinel
			case "allocate":
				p.allocErr = sentinel
			case "launch":
				p.launchErr = sentinel
			case "synchronize":
				p.syncErr = sentinel
			case "copy":
				p.copyErr = sentinel
			case "free":
				p.freeErr = sentinel
			}
			e, err := runExecution(context.Background(), nodes, 1003, executionPointers(nodes), false, p.ops(), []func(){func() { p.released++ }})
			if stage == "copy" {
				_, err = e.Readback(context.Background())
			}
			if stage == "free" {
				err = e.Close()
			}
			if !errors.Is(err, sentinel) {
				t.Fatal("original error lost", err)
			}
			if stage == "launch" || stage == "synchronize" {
				if p.frees != 0 || p.released != 0 {
					t.Fatal("uncompleted owners released")
				}
			}

			if stage == "launch" {
				if _, readErr := e.Readback(context.Background()); !errors.Is(readErr, sentinel) {
					t.Fatal("failed launch exposed output", readErr)
				}
			}
			p.syncErr = nil
			p.freeErr = nil
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if p.released != 1 || p.frees != p.allocs {
				t.Fatal("owner not cleaned once")
			}
		})
	}
	p := &executionProbe{cancel: true}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e, err := runExecution(ctx, nodes, 1003, executionPointers(nodes), false, p.ops(), []func(){func() { p.released++ }})
	if !errors.Is(err, context.Canceled) || p.launches != 1 || p.frees != 0 || p.released != 0 {
		t.Fatal("cancellation treated as completion", err)
	}
	p.syncErr = sentinel
	if err := e.Close(); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	if p.frees != 0 {
		t.Fatal("failed drain freed work")
	}
	p.syncErr = nil
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if p.frees != 1 || p.released != 1 {
		t.Fatal("recovery omitted cleanup")
	}
}

func TestLazyExecution_PartialAcquisitionCleanup(t *testing.T) {
	_, nodes := executionChain(t, 1003)
	sentinel := errors.New("partial acquisition")
	for _, stage := range []string{"module-returned-with-error", "buffer-returned-with-error", "second-eager-compile"} {
		t.Run(stage, func(t *testing.T) {
			p := &executionProbe{}
			ops := p.ops()
			if stage == "module-returned-with-error" {
				compile := ops.compile
				ops.compile = func(f FusedKernel) (*executionKernel, error) { k, _ := compile(f); return k, sentinel }
			}
			if stage == "buffer-returned-with-error" {
				alloc := ops.alloc
				ops.alloc = func(n int) (*executionBuffer, error) { b, _ := alloc(n); return b, sentinel }
			}
			if stage == "second-eager-compile" {
				compile := ops.compile
				calls := 0
				ops.compile = func(f FusedKernel) (*executionKernel, error) {
					calls++
					if calls == 2 {
						return nil, sentinel
					}
					return compile(f)
				}
			}
			e, err := runExecution(context.Background(), nodes, 1003, executionPointers(nodes), stage == "second-eager-compile", ops, []func(){func() { p.released++ }})
			if !errors.Is(err, sentinel) {
				t.Fatal(err)
			}
			if stage == "second-eager-compile" && (p.frees != 0 || p.released != 0) {
				t.Fatal("previous launch owners freed before drain")
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if p.frees != p.allocs || p.closed != len(p.sources) || p.released != 1 {
				t.Fatal("partial owner leaked", p)
			}
			if e.Stats().Allocations != p.allocs || e.Stats().Frees != p.frees {
				t.Fatal("actual successful allocation/free lost", e.Stats())
			}
		})
	}
}

type executionPendingCleanup struct {
	cause, cleanupErr error
	calls             int
}

func (e *executionPendingCleanup) Error() string { return e.cause.Error() }
func (e *executionPendingCleanup) Unwrap() error { return e.cause }
func (e *executionPendingCleanup) Close() error  { e.calls++; return e.cleanupErr }

func TestLazyExecution_CompilerErrorCleanupOwner(t *testing.T) {
	_, nodes := executionChain(t, 1003)
	for _, kind := range []string{"NVRTC-program", "failed-module-load"} {
		t.Run(kind, func(t *testing.T) {
			original := errors.New(kind + " original failure")
			cleanupErr := errors.New("retained native cleanup failure")
			pending := &executionPendingCleanup{cause: original, cleanupErr: cleanupErr}
			p := &executionProbe{}
			ops := p.ops()
			ops.compile = func(FusedKernel) (*executionKernel, error) {
				return nil, fmt.Errorf("wrapped compiler error: %w", pending)
			}
			e, err := runExecution(context.Background(), nodes, 1003, executionPointers(nodes), false, ops, []func(){func() { p.released++ }})
			if !errors.Is(err, original) || !errors.Is(err, cleanupErr) {
				t.Fatal("original or cleanup error lost", err)
			}
			if e.closed || pending.calls != 1 || p.released != 0 {
				t.Fatal("pending native owner discarded")
			}
			if err := e.Close(); !errors.Is(err, cleanupErr) {
				t.Fatal("failed retry lost owner", err)
			}
			pending.cleanupErr = nil
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if pending.calls != 3 || !e.closed || p.released != 1 {
				t.Fatal("pending owner was not retried")
			}
			if err := e.Close(); err != nil || pending.calls != 3 {
				t.Fatal("successful owner released twice")
			}
		})
	}
}

func TestLazyExecution_FreeSucceededBeforeContextRestoreError(t *testing.T) {
	_, nodes := executionChain(t, 1003)
	p := &executionProbe{}
	ops := p.ops()
	alloc := ops.alloc
	sentinel := errors.New("restore previous context")
	freed := false
	calls := 0
	ops.alloc = func(n int) (*executionBuffer, error) {
		b, err := alloc(n)
		b.free = func() error { calls++; freed = true; return sentinel }
		b.freed = func() bool { return freed }
		return b, err
	}
	e, err := runExecution(context.Background(), nodes, 1003, executionPointers(nodes), false, ops, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	if e.Stats().Frees != 1 {
		t.Fatal("successful native free lost because restoration failed")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if calls != 1 || e.Stats().Frees != 1 {
		t.Fatal("native free repeated or fabricated")
	}
}

func executionValues(n int) (a, b, c, want []float32) {
	a = make([]float32, n)
	b = make([]float32, n)
	c = make([]float32, n)
	want = make([]float32, n)
	for i := range a {
		a[i] = float32((i*17)%97-48) / 8
		b[i] = float32((i*29)%89-44) / 16
		c[i] = float32(i%13+1) / 4
		want[i] = (max(a[i], 0) + max(b[i], 0)) * c[i]
	}
	return
}
func TestLazyExecution_WrongSecondOperandDetected(t *testing.T) {
	a, _, c, want := executionValues(1003)
	wrong := make([]float32, len(a))
	for i := range wrong {
		wrong[i] = (max(a[i], 0) + max(a[i], 0)) * c[i]
	}
	policy := cudatest.CUDAOutputPolicy{Target: "host control", DType: "fp32", MathMode: "strict --fmad=false", SourceDigest: strings.Repeat("1", 64), InputDigest: strings.Repeat("2", 64), Elements: 1003, NaN: "reject", Infinity: "reject", Zero: "equal"}
	if err := cudatest.CompareCUDAOutputFP32(want, want, policy); err != nil {
		t.Fatal(err)
	}
	var mismatch *cudatest.CUDAOutputMismatch
	if err := cudatest.CompareCUDAOutputFP32(wrong, want, policy); !errors.As(err, &mismatch) {
		t.Fatal("wrong second operand accepted", err)
	}
}

func TestLazyExecution_OutputFillAndTailControls(t *testing.T) {
	poison := math.Float32frombits(0x7fc01234)
	good := make([]float32, 1005)
	good[0] = poison
	good[1004] = poison
	if err := checkExecutionOutput(good, 1003); err != nil {
		t.Fatal(err)
	}
	for _, index := range []int{0, 1, 1003, 1004} {
		bad := slices.Clone(good)
		if index == 0 || index == 1004 {
			bad[index] = 0
		} else {
			bad[index] = poison
		}
		if err := checkExecutionOutput(bad, 1003); err == nil {
			t.Fatalf("output control %d passed", index)
		}
	}
}

func TestLazyExecution_ActualOperandDataflow(t *testing.T) {
	p, nodes := executionChain(t, 1003)
	source, err := executionSource(nodes, p.RootNode, "dataflow_control", false)
	if err != nil {
		t.Fatal(err)
	}
	// Follow the emitted scalar definitions, not mere presence of all input names.
	// A kernel can load b but still calculate relu(a)+relu(a).
	roots := map[string]string{}
	for _, m := range regexp.MustCompile(`float (v_[A-Za-z0-9_]+) = ([A-Za-z0-9_]+)\[idx\];`).FindAllStringSubmatch(source.Source, -1) {
		roots[m[1]] = m[2]
	}
	relus := regexp.MustCompile(`float (v_[A-Za-z0-9_]+) = \((v_[A-Za-z0-9_]+) > 0\.0f\) \? (v_[A-Za-z0-9_]+) : 0\.0f;`).FindAllStringSubmatch(source.Source, -1)
	for _, m := range relus {
		if m[2] != m[3] || roots[m[2]] == "" {
			t.Fatal("relu does not retain actual operand")
		}
		roots[m[1]] = roots[m[2]]
	}
	adds := regexp.MustCompile(`float (v_[A-Za-z0-9_]+) = (v_[A-Za-z0-9_]+) \+ (v_[A-Za-z0-9_]+);`).FindAllStringSubmatch(source.Source, -1)
	if len(adds) != 1 || roots[adds[0][2]] != "node_1_input" || roots[adds[0][3]] != "node_2_input" {
		t.Fatal("add does not consume both distinct relus", source.Source)
	}
	muls := regexp.MustCompile(`float (v_[A-Za-z0-9_]+) = (v_[A-Za-z0-9_]+) \* (v_[A-Za-z0-9_]+);`).FindAllStringSubmatch(source.Source, -1)
	if len(muls) != 1 || muls[0][2] != adds[0][1] || roots[muls[0][3]] != "node_3_input" {
		t.Fatal("multiply does not consume sum and c", source.Source)
	}
	store := regexp.MustCompile(`(?m)^\s*out_` + regexp.QuoteMeta(source.Outputs[0]) + `\[idx\] = ` + regexp.QuoteMeta(muls[0][1]) + `;\s*$`)
	if !strings.Contains(source.Source, "idx < (unsigned int)n;") || !store.MatchString(source.Source) {
		t.Fatal("missing bounded output store", source.Source)
	}
}
