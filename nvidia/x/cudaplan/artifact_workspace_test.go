package cudaplan

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func artifactWorkspace(t testing.TB, bytes uint64, alignment uint32, position int) *cudakernel.Artifact {
	t.Helper()
	params := []cudakernel.Param{{Name: "input", Type: "ptr", IsPointer: true}, {Name: "output", Type: "ptr", IsPointer: true}}
	var w cudakernel.WorkspaceRequirements
	if bytes != 0 {
		w = cudakernel.WorkspaceRequirements{Bytes: bytes, Alignment: alignment, Parameter: "scratch"}
		params = append(params, cudakernel.Param{})
		copy(params[position+1:], params[position:])
		params[position] = cudakernel.Param{Name: "scratch", Type: "ptr", IsPointer: true}
	}
	schema, err := cudakernel.NewSchema(params...)
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("host ABI fixture only"), "kernel", schema, "sm_80", cudakernel.WithResources(cudakernel.ResourceRequirements{Workspace: w}))
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func workspaceBranch(t testing.TB) (*cudair.Snapshot, *KernelRegistry) {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{8}, cudair.Float32)
	a := g.AddUnary("neg", x)
	b := g.AddUnary("neg", x)
	if err := g.SetOutputs(a, b); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	r := NewKernelRegistry()
	for i, n := range []*cudair.Node{a, b} {
		art := artifactWorkspace(t, uint64(64+32*i), uint32(4096>>uint(i*3)), i)
		if err := r.Register(n.Name, art); err != nil {
			t.Fatal(err)
		}
	}
	return snap, r
}

func TestArtifactWorkspaceCompileBudgetAndSideTable(t *testing.T) {
	snap, registry := workspaceBranch(t)
	prefix := 256 // Existing frozen arena reuses one 256-byte block for these unary outputs.
	required := prefix + (64 + 4096 - 1) + (96 + 512 - 1)
	required = (required + 255) &^ 255
	acquired := 0
	if _, err := Compile(snap, WithKernelRegistry(registry), WithPeakWorkspaceBudget(required-1), WithInjectedCompileFailure(false, func() { acquired++ })); !errors.Is(err, ErrWorkspaceBudgetExceeded) || acquired != 0 {
		t.Fatalf("scratch peak not enforced before acquisition: %v %d", err, acquired)
	}
	p, err := Compile(snap, WithKernelRegistry(registry), WithPeakWorkspaceBudget(required))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	steps := p.Steps()
	if len(steps) != 2 {
		t.Fatalf("steps: %d", len(steps))
	}
	if steps[0].ScratchOffset != prefix || steps[0].ScratchCapacity != 4159 || steps[1].ScratchOffset != prefix+4159 || steps[1].ScratchCapacity != 607 {
		t.Fatalf("actual Compile reservations: %+v", steps)
	}
	if steps[0].WorkspaceBytes != 32 || steps[1].WorkspaceBytes != 32 {
		t.Fatal("scratch replaced output extents")
	}
	low, err := LowerRegion(steps, Region{0, 2})
	if err != nil {
		t.Fatal(err)
	}
	tasks := low.Graph().Tasks()
	if tasks[0].Rank != tasks[1].Rank {
		t.Fatal("independent branch serialized to share scratch")
	}
	for i, id := range low.TaskIDs() {
		b, ok := low.Binding(id)
		if !ok || b.ScratchOffset != steps[i].ScratchOffset || b.ScratchCapacity != steps[i].ScratchCapacity || b.Artifact.Resources().Workspace != steps[i].Artifact.Resources().Workspace {
			t.Fatal("scratch side-table binding lost")
		}
		b.ScratchCapacity = 0
		if again, _ := low.Binding(id); again.ScratchCapacity == 0 {
			t.Fatal("side table aliases caller")
		}
	}
	if got, err := vendorWorkspaceBudget(required, required+1024); err != nil || got != 1024 {
		t.Fatalf("vendor allowance escaped scratch charge: %d %v", got, err)
	}
	// The registry wrapper can be replaced after Compile; the plan's workspace
	// declaration and execution handle must still be the originally captured one.
	first := steps[0].Artifact
	replacement := artifactWorkspace(t, 1, 1, 0)
	*first = *replacement
	if p.Steps()[0].Artifact.Resources().Workspace.Bytes != 64 {
		t.Fatal("Steps wrapper changed captured workspace")
	}
}

func TestArtifactWorkspaceArgumentSlots(t *testing.T) {
	for position := 0; position < 3; position++ {
		art := artifactWorkspace(t, 4096, 4096, position)
		step := StepPlan{NodeName: "y", InputNames: []string{"x"}, OutputName: "y", Artifact: art, ScratchOffset: 256, ScratchCapacity: 8191}
		ptr, err := artifactScratchPointer(0x1100, step.ScratchCapacity, art.Resources().Workspace)
		if err != nil || ptr != 0x2000 {
			t.Fatalf("alignment > allocation alignment: %x %v", ptr, err)
		}
		args, err := artifactKernelArguments(step, map[string]driver.DevicePtr{"x": 0x1000, "y": 0x4000}, ptr)
		if err != nil {
			t.Fatal(err)
		}
		packed, err := art.Pack(args...)
		if err != nil {
			t.Fatal(err)
		}
		tensor := 0
		for j, p := range art.Schema().Params {
			want := uint64(ptr)
			if j != position {
				want = []uint64{0x1000, 0x4000}[tensor]
				tensor++
			}
			if got := binary.LittleEndian.Uint64(packed[p.Offset:]); got != want {
				t.Fatalf("real packed parameter %d: %x want %x", j, got, want)
			}
		}
		if _, err := artifactKernelArguments(step, map[string]driver.DevicePtr{"x": 0x1000, "y": 0x4000}, ptr+1); err == nil {
			t.Fatal("misaligned scratch accepted")
		}
	}
}

func TestArtifactWorkspaceRefusal(t *testing.T) {
	w := cudakernel.WorkspaceRequirements{Bytes: 4096, Alignment: 4096, Parameter: "scratch"}
	for _, tc := range []struct {
		name     string
		base     driver.DevicePtr
		capacity int
	}{
		{"nil", 0, 8191}, {"negative", 0x1000, -1}, {"short", 0x1100, 4096}, {"pointer wrap", driver.DevicePtr(math.MaxUint64 - 255), 8191},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := artifactScratchPointer(tc.base, tc.capacity, w); err == nil {
				t.Fatal("invalid scratch range accepted")
			}
		})
	}
	for _, tc := range []struct {
		name                    string
		base                    driver.DevicePtr
		total, offset, capacity int
	}{
		{"base plus offset wrap", driver.DevicePtr(math.MaxUint64 - 31), 16384, 256, 8191},
		{"past object", 0x1000, 4096, 256, 8191},
		{"negative offset", 0x1000, 16384, -1, 8191},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := artifactScratchPointerAt(tc.base, tc.total, tc.offset, tc.capacity, w); err == nil {
				t.Fatal("invalid workspace pointer formation accepted")
			}
		})
	}
	if ptr, err := artifactScratchPointerAt(0x1000, 16384, 256, 8191, w); err != nil || ptr != 0x2000 {
		t.Fatalf("valid owned reservation: %x %v", ptr, err)
	}

	snap, registry := workspaceBranch(t)
	art := registry.Get(snap.Outputs()[0].Name)
	for _, tc := range []struct {
		name           string
		bytes          uint64
		alignment      uint32
		prefix, budget int
	}{
		{"uint to int", math.MaxUint64, 1, 0, math.MaxInt}, {"alignment overflow", uint64(math.MaxInt), 4096, 0, math.MaxInt}, {"cumulative", 1, 1, math.MaxInt, math.MaxInt}, {"rounding", 1, 1, math.MaxInt - 1, math.MaxInt},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := artifactWorkspace(t, tc.bytes, tc.alignment, 0)
			if _, _, err := planArtifactWorkspace(snap.Order(), map[int]*cudakernel.Artifact{snap.Outputs()[0].ID: a}, false, tc.prefix, tc.budget); !errors.Is(err, ErrWorkspaceBudgetExceeded) {
				t.Fatalf("overflow accepted: %v", err)
			}
		})
	}
	p, err := Compile(snap, WithKernelRegistry(registry))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	steps := p.Steps()
	if _, _, err := p.kernelArguments(steps[0], map[string]driver.DevicePtr{"x": 0x1000, steps[0].OutputName: 0x2000}); !errors.Is(err, ErrMissingBinding) {
		t.Fatalf("missing workspace owner: %v", err)
	}
	edits := []struct {
		name string
		edit func([]StepPlan)
	}{
		{"overlap", func(s []StepPlan) { s[1].ScratchOffset = s[0].ScratchOffset }},
		{"truncated", func(s []StepPlan) { s[0].ScratchCapacity = 64 }},
		{"offset overflow", func(s []StepPlan) { s[0].ScratchOffset = math.MaxInt }},
		{"finite worker", func(s []StepPlan) { s[0].Backend = BackendMega }},
		{"output arena overlap", func(s []StepPlan) {
			s[1].IsOutput = false
			s[1].WorkspaceOffset = s[0].ScratchOffset
			s[1].WorkspaceBytes = 32
		}},
	}
	for _, tc := range edits {
		t.Run(tc.name, func(t *testing.T) {
			s := p.Steps()
			tc.edit(s)
			if _, err := LowerRegion(s, Region{0, len(s)}); err == nil {
				t.Fatal("invalid lowered scratch accepted")
			}
		})
	}
	bad := art.Spec()
	bad.Schema, err = cudakernel.NewSchema(cudakernel.Param{Name: "scratch", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifactFromSpec(bad)
	if err != nil {
		t.Fatal(err)
	}
	r := NewKernelRegistry()
	if err := r.Register("neg", a); err != nil {
		t.Fatal(err)
	}
	if _, err := Compile(snap, WithKernelRegistry(r)); err == nil {
		t.Fatal("positive scratch without real tensor ABI admitted")
	}
}

func TestArtifactWorkspaceGraphArgumentsAndLifetime(t *testing.T) {
	snap, registry := workspaceBranch(t)
	p, err := Compile(snap, WithKernelRegistry(registry))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { p.graphObj = nil; p.graphExec = nil; p.Close() }()
	pointers := map[string]driver.DevicePtr{"x": 0x1000, p.steps[0].OutputName: 0x2000, p.steps[1].OutputName: 0x3000}
	graph := &cudagraph.Graph{}
	exec := &cudagraph.Exec{}
	previous := (*cudagraph.Node)(nil)
	var captured [][]any
	ops := graphBuildOps{create: func() (*cudagraph.Graph, error) { return graph, nil }, instantiate: func(g *cudagraph.Graph) (*cudagraph.Exec, error) { return exec, nil }, add: func(g *cudagraph.Graph, s StepPlan, ptrs map[string]driver.DevicePtr, prev *cudagraph.Node) (*cudagraph.Node, error) {
		if prev != previous {
			t.Fatal("lost workspace sequencing edge")
		}
		base := driver.DevicePtr(0x10000 + s.ScratchOffset)
		scratch, err := artifactScratchPointer(base, s.ScratchCapacity, s.Artifact.Resources().Workspace)
		if err != nil {
			return nil, err
		}
		args, err := artifactKernelArguments(s, ptrs, scratch)
		captured = append(captured, args)
		previous = &cudagraph.Node{}
		return previous, err
	}}
	if err := p.prepareGraphWith(pointers, ops); err != nil {
		t.Fatal(err)
	}
	first := append([][]any(nil), captured...)
	if err := p.prepareGraphWith(pointers, graphBuildOps{}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, captured) || len(captured) != 2 {
		t.Fatal("graph replay rebuilt/changed retained scratch arguments")
	}
	pointers["x"]++
	if err := p.prepareGraphWith(pointers, graphBuildOps{}); !errors.Is(err, ErrStaleBinding) {
		t.Fatal("stale tensor binding escaped")
	}
	// Host ownership control uses the same Completion release path Execute
	// transfers its whole-workspace borrow to. It does not simulate GPU math.
	held, attempts := true, 0
	c := newHostCompletion(func(ctx context.Context) error {
		if ctx != nil && ctx.Err() != nil {
			return ctx.Err()
		}
		attempts++
		if attempts == 1 {
			return errors.New("drain failed")
		}
		return nil
	}, nil, func() { held = false })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if !errors.Is(c.Wait(ctx), context.Canceled) || !held {
		t.Fatal("cancellation freed scratch lease")
	}
	if err := c.Close(); err == nil || !held {
		t.Fatal("failed drain freed scratch lease")
	}
	if err := c.Close(); err != nil || held {
		t.Fatalf("successful drain retained scratch lease: %v", err)
	}
	// Remove synthetic native owners; this test did not create CUDA resources.
	p.graphObj = nil
	p.graphExec = nil
}

func TestArtifactWorkspaceBudgetBeforeDeviceAcquisition(t *testing.T) {
	snap, r := workspaceBranch(t)
	if _, err := Compile(snap, WithDevice(&cuda.Device{}), WithKernelRegistry(r), WithPeakWorkspaceBudget(256)); !errors.Is(err, ErrWorkspaceBudgetExceeded) {
		t.Fatalf("scratch budget should fail before target/device API: %v", err)
	}
}

func TestArtifactWorkspaceVendorDemandIsSeparate(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2, 2}, cudair.Float32)
	y := g.AddMatMul(x, x)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	r := NewKernelRegistry()
	// The current policy selects cuBLASLt, so this unused ordinary artifact
	// must not consume either kernel scratch or the vendor workspace budget.
	if err := r.Register("matmul", artifactWorkspace(t, math.MaxUint64, 4096, 0)); err != nil {
		t.Fatal(err)
	}
	p, err := Compile(snap, WithKernelRegistry(r), WithPeakWorkspaceBudget(256))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if p.Steps()[0].Backend != BackendCuBLASLt || p.Steps()[0].ScratchCapacity != 0 {
		t.Fatal("vendor artifact charged ordinary scratch")
	}
	steps := p.Steps()
	if err := validateRegionScratch(steps); err != nil {
		t.Fatalf("unused vendor artifact demand: %v", err)
	}
	if _, err := LowerRegion(steps, Region{Start: 0, End: len(steps)}); !errors.Is(err, ErrNotLowerable) || !strings.Contains(err.Error(), "vendor library") {
		t.Fatalf("vendor worker refusal = %v", err)
	}
	steps[0].ScratchCapacity = 1
	if err := validateRegionScratch(steps); !errors.Is(err, ErrNotLowerable) || !strings.Contains(err.Error(), "vendor step has artifact scratch reservation") {
		t.Fatalf("vendor reservation = %v", err)
	}
}

func TestArtifactWorkspaceZeroArgumentABIStillChecked(t *testing.T) {
	art := artifactWorkspace(t, 64, 16, 1)
	spec := art.Spec()
	spec.Resources.Workspace = cudakernel.WorkspaceRequirements{}
	zero, err := cudakernel.NewArtifactFromSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	step := StepPlan{InputNames: []string{"x"}, OutputName: "y", Artifact: zero}
	if _, err := artifactKernelArguments(step, map[string]driver.DevicePtr{"x": 0x1000, "y": 0x2000}, 0); err == nil {
		t.Fatal("zero declaration silently dropped extra kernel parameter")
	}
}
