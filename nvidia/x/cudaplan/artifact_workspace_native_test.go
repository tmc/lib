package cudaplan_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

// This one-CTA fixture actually uses global scratch: write all inputs, barrier,
// read in reverse lane order and store -x at the matching reversed output index.
// Guards occupy the first and last declared float32 scratch slots. All 256
// threads reach the barrier; inactive lanes never access data. The omission
// mutant returns zeros after the same scratch writes and must fail the oracle.
func artifactWorkspaceNativeFixture(t testing.TB, target string, mutant bool) (*cudair.Snapshot, *cudaplan.KernelRegistry) {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{64}, cudair.Float32)
	a := g.AddUnary("neg", x)
	b := g.AddUnary("neg", x)
	if err := g.SetOutputs(a, b); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "input", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "scratch", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "output", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	registry := cudaplan.NewKernelRegistry()
	for i, node := range []*cudair.Node{a, b} {
		demand := uint64(264 + 256*i)
		read := "ld.global.f32 %f1, [%rd5];"
		if mutant {
			read = "mov.f32 %f1, 0f00000000;"
		}
		source := fmt.Sprintf(`.version 9.0
.target %s
.address_size 64
.visible .entry scratch_neg(.param .u64 input, .param .u64 scratch, .param .u64 output) {
 .reg .b32 %%r<8>;
 .reg .b64 %%rd<10>;
 .reg .f32 %%f<4>;
 .reg .pred %%p<4>;
 ld.param.u64 %%rd0, [input];
 ld.param.u64 %%rd1, [scratch];
 ld.param.u64 %%rd2, [output];
 mov.u32 %%r0, %%tid.x;
 setp.eq.u32 %%p1, %%r0, 0;
 @!%%p1 bra INIT_DONE;
 mov.u32 %%r4, 0x4f012345;
 st.global.u32 [%%rd1], %%r4;
 add.u64 %%rd8, %%rd1, %d;
 st.global.u32 [%%rd8], %%r4;
INIT_DONE:
 setp.lt.u32 %%p0, %%r0, 64;
 @!%%p0 bra WRITES_DONE;
 mul.wide.u32 %%rd3, %%r0, 4;
 add.u64 %%rd4, %%rd0, %%rd3;
 ld.global.f32 %%f0, [%%rd4];
 add.u64 %%rd5, %%rd1, %%rd3;
 add.u64 %%rd5, %%rd5, 4;
 st.global.f32 [%%rd5], %%f0;
WRITES_DONE:
 bar.sync 0;
 @!%%p0 bra DONE;
 sub.u32 %%r1, 63, %%r0;
 mul.wide.u32 %%rd3, %%r1, 4;
 add.u64 %%rd5, %%rd1, %%rd3;
 add.u64 %%rd5, %%rd5, 4;
 %s
 ld.global.u32 %%r4, [%%rd1];
 add.u64 %%rd8, %%rd1, %d;
 ld.global.u32 %%r5, [%%rd8];
 setp.ne.u32 %%p1, %%r4, 0x4f012345;
 setp.ne.u32 %%p2, %%r5, 0x4f012345;
 or.pred %%p1, %%p1, %%p2;
 @%%p1 mov.b32 %%f1, 0x7fc00000;
 neg.f32 %%f2, %%f1;
 add.u64 %%rd6, %%rd2, %%rd3;
 st.global.f32 [%%rd6], %%f2;
DONE:
 ret;
}
`, target, demand-4, read, demand-4)
		art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(source), "scratch_neg", schema, target, cudakernel.WithResources(cudakernel.ResourceRequirements{Workspace: cudakernel.WorkspaceRequirements{Bytes: demand, Alignment: uint32(4096 << uint(i)), Parameter: "scratch"}}))
		if err != nil {
			t.Fatal(err)
		}
		if err := registry.Register(node.Name, art); err != nil {
			t.Fatal(err)
		}
	}
	return snap, registry
}

func TestArtifactWorkspaceNativeFixtureSource(t *testing.T) {
	snap, registry := artifactWorkspaceNativeFixture(t, "sm_80", false)
	p, err := cudaplan.Compile(snap, cudaplan.WithKernelRegistry(registry))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	steps := p.Steps()
	if len(steps) != 2 || steps[0].ScratchOffset+steps[0].ScratchCapacity > steps[1].ScratchOffset {
		t.Fatal("native fixture scratch is not disjoint")
	}
	for _, step := range steps {
		text := string(step.Artifact.Bytes())
		for _, token := range []string{"st.global.f32 [%rd5]", "bar.sync 0", "ld.global.f32 %f1, [%rd5]", "ld.global.u32 %r5", "st.global.f32 [%rd6]"} {
			if !strings.Contains(text, token) {
				t.Fatalf("fixture lacks actual scratch effect %s", token)
			}
		}
	}
	// This only checks reference inputs and native fixture custody, not PTX
	// compiler acceptance or kernel execution.
	data := make([]float32, 64)
	for i := range data {
		data[i] = float32(i-31) / 4
	}
	want, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"x": data})
	if err != nil {
		t.Fatal(err)
	}
	for _, output := range snap.Outputs() {
		for i, v := range want[output.Name] {
			if v != -data[i] {
				t.Fatal("reference fixture shape/semantics")
			}
		}
	}
}

// TestNativeArtifactWorkspaceReference is source-only until separately
// authorized target execution enables its explicit gate. Ordinary execution
// must pass before graph replay. Host fixture/encoding checks are not this gate.
func TestNativeArtifactWorkspaceReference(t *testing.T) {
	if os.Getenv("CUDAPLAN_WORKSPACE_NATIVE") != "1" {
		t.Skip("native workspace qualification requires CUDAPLAN_WORKSPACE_NATIVE=1")
	}
	target := os.Getenv("CUDAPLAN_NATIVE_TARGET")
	if target != "sm_100a" && target != "sm_121a" {
		t.Fatal("set CUDAPLAN_NATIVE_TARGET explicitly to sm_100a or sm_121a")
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDAPLAN_NATIVE_DEVICE"))
	if err != nil || ordinal < 0 {
		t.Fatal("set CUDAPLAN_NATIVE_DEVICE to an explicit ordinal")
	}
	owners := &workflowNativeOwners{}
	t.Cleanup(func() {
		if err := closeNativePlanOwners(owners); err != nil {
			t.Error(err)
		}
	})
	var dev *cuda.Device
	err = setupNativePlanOwners(owners,
		func() error { _, err := cudalock.Inherited(os.Getenv); return err },
		func() (func() error, error) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			lease, err := cudalock.Acquire(ctx, cudalock.WithLabel(t.Name()))
			if lease != nil {
				return lease.Release, err
			}
			return nil, err
		},
		func() (func() error, error) {
			var err error
			dev, err = cuda.NewDevice(ordinal)
			if dev != nil {
				return dev.Close, err
			}
			return nil, err
		})
	// setupNativePlanOwners has already retained any partial error owner.
	if err != nil {
		t.Fatal(err)
	}
	check := func(err error) {
		retainNativePlanError(owners, err)
		if err != nil {
			t.Fatal(err)
		}
	}
	major, minor, err := dev.ComputeCapability()
	check(err)
	if fmt.Sprintf("sm_%d%da", major, minor) != target {
		t.Fatalf("native workspace target: SM%d%d %v", major, minor, err)
	}
	stream, err := dev.NewStream()
	if stream != nil {
		owners.ownStream(stream)
	}
	check(err)
	alloc := func() cudamem.Slice[float32] {
		b, err := cudamem.Alloc[float32](dev, 64)
		if b.Ptr() != 0 {
			owners.add(workflowBuffer, "workspace buffer", b.Free)
		}
		check(err)
		return b
	}
	input, outA, outB := alloc(), alloc(), alloc()
	values := make([]float32, 64)
	for i := range values {
		values[i] = float32(i-31) / 4
	}
	check(input.CopyFromHost(values))
	for _, mutant := range []bool{false, true} {
		for _, replay := range []bool{false, true} {
			if mutant && replay {
				continue
			}
			snap, registry := artifactWorkspaceNativeFixture(t, target, mutant)
			reference, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"x": values})
			if err != nil {
				t.Fatal(err)
			}
			p, err := cudaplan.Compile(snap, cudaplan.WithDevice(dev), cudaplan.WithKernelRegistry(registry), cudaplan.WithGraphReplay(replay))
			if p != nil {
				owners.add(workflowPlan, "workspace plan", p.Close)
			}
			check(err)
			binding := cudaplan.NewBindings()
			check(cudaplan.SetInputSlice(binding, "x", input, dev))
			for i, dest := range []cudamem.Slice[float32]{outA, outB} {
				check(cudaplan.SetOutputSlice(binding, snap.Outputs()[i].Name, dest, dev))
			}
			for round := 0; round < 2; round++ {
				poison := make([]float32, 64)
				for i := range poison {
					poison[i] = float32(math.NaN())
				}
				for _, out := range []cudamem.Slice[float32]{outA, outB} {
					check(out.CopyFromHost(poison))
				}
				completion, err := p.Execute(stream, binding)
				owners.retainCompletion(completion)
				check(err)
				canceled, cancel := context.WithCancel(context.Background())
				cancel()
				if err := completion.Wait(canceled); !errors.Is(err, context.Canceled) {
					t.Fatalf("canceled workspace wait: %v", err)
				}
				if err := outA.Free(); !errors.Is(err, cudamem.ErrInUse) {
					t.Fatal("canceled wait released output owner")
				}
				if err := p.Close(); !errors.Is(err, cudaplan.ErrInvocationInFlight) {
					t.Fatalf("canceled wait released plan/workspace owner: %v", err)
				}
				wait, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
				err = completion.Wait(wait)
				cancel()
				check(err)
				differs := false
				for i, out := range []cudamem.Slice[float32]{outA, outB} {
					got := make([]float32, 64)
					check(out.CopyToHost(got))
					for j, v := range got {
						if math.Float32bits(v) != math.Float32bits(reference[snap.Outputs()[i].Name][j]) {
							differs = true
						}
					}
				}
				if differs != mutant {
					t.Fatalf("workspace reference mismatch=%v mutant=%v replay=%v", differs, mutant, replay)
				}
			}
		}
	}
}

// Check the actual native entry point without opening a device. The shared
// setup helper's injected failure tests cover admission, partial acquisition,
// and retained cleanup; this test prevents this client from bypassing it.
func TestArtifactWorkspaceNativeOwnerRouting(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "artifact_workspace_native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var body *ast.BlockStmt
	for _, decl := range f.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "TestNativeArtifactWorkspaceReference" {
			body = fn.Body
		}
	}
	if body == nil {
		t.Fatal("native fixture missing")
	}
	var buf bytes.Buffer
	if err := format.Node(&buf, token.NewFileSet(), body); err != nil {
		t.Fatal(err)
	}
	text := strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return -1
		}
		return r
	}, buf.String())
	sequence := func(parts ...string) {
		t.Helper()
		pos := 0
		for _, part := range parts {
			i := strings.Index(text[pos:], part)
			if i < 0 {
				t.Fatalf("missing ordered ownership operation %s", part)
			}
			pos += i + len(part)
		}
	}
	sequence(`t.Cleanup(`, `closeNativePlanOwners(owners)`, `setupNativePlanOwners(owners,`, `cudalock.Inherited(os.Getenv)`, `cudalock.Acquire(`, `returnlease.Release,err`, `cuda.NewDevice(ordinal)`, `returndev.Close,err`, `iferr!=nil{t.Fatal(err)}`, `check:=func(errerror){retainNativePlanError(owners,err)`, `stream,err:=dev.NewStream()`, `owners.ownStream(stream)`, `check(err)`, `cudamem.Alloc[float32]`, `ifb.Ptr()!=0{owners.add(workflowBuffer,`, `check(err)`, `cudaplan.Compile(`, `ifp!=nil{owners.add(workflowPlan,`, `check(err)`, `p.Execute(`, `owners.retainCompletion(completion)`, `check(err)`)
	if strings.Contains(text, `GPU_LOCK_HELD`) || strings.Contains(text, `lease.Release()`) {
		t.Fatal("native fixture bypasses inherited admission or ownership chain")
	}
}

func TestArtifactWorkspaceNativeSetupRetainsChain(t *testing.T) {
	failed := errors.New("workspace acquisition failure")
	var calls []string
	owners := &workflowNativeOwners{}
	err := setupNativePlanOwners(owners,
		func() error { calls = append(calls, "admit"); return nil },
		func() (func() error, error) { return func() error { calls = append(calls, "lease"); return nil }, nil },
		func() (func() error, error) {
			return func() error { calls = append(calls, "device"); return nil }, &nativePlanClientError{failed, func() error { calls = append(calls, "error owner"); return failed }}
		})
	if !errors.Is(err, failed) {
		t.Fatalf("setup = %v", err)
	}
	// A failed partial owner close must keep device and child lease retained.
	if err := closeNativePlanOwners(owners); !errors.Is(err, failed) {
		t.Fatalf("close = %v", err)
	}
	if strings.Join(calls, ",") != "admit,error owner" || len(owners.resources[workflowDevice]) != 1 || len(owners.resources[workflowLease]) != 1 {
		t.Fatalf("lost owner chain: %v", calls)
	}
}
