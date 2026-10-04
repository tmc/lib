package cudaplan

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestBackendSelectionIsTestableWithoutDevice(t *testing.T) {
	tests := []struct {
		name    string
		request BackendRequest
		want    TargetBackend
		fail    bool
	}{
		{"ordinary", BackendRequest{Kind: cudair.KindUnary}, BackendLaunch, false},
		{"vendor", BackendRequest{Kind: cudair.KindMatMul, VendorAvailable: true}, BackendCuBLASLt, false},
		{"missing vendor", BackendRequest{Kind: cudair.KindMatMul}, 0, true},
		{"graph", BackendRequest{Kind: cudair.KindUnary, GraphRequested: true, GraphAvailable: true}, BackendGraph, false},
		{"missing graph", BackendRequest{Kind: cudair.KindUnary, GraphRequested: true}, 0, true},
		{"missing code", BackendRequest{Kind: cudair.KindReduce, RequireArtifact: true}, 0, true},
		{"unknown kind", BackendRequest{Kind: cudair.NodeKind(99)}, 0, true},
		{"unadmitted finite worker", BackendRequest{Kind: cudair.KindUnary, FiniteMegaRequested: true}, 0, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := SelectBackend(test.request)
			if (err != nil) != test.fail || got.Backend != test.want {
				t.Fatalf("decision=%+v error=%v", got, err)
			}
			if !test.fail && got.Reason == "" {
				t.Fatal("admission has no reason")
			}
		})
	}
}

func TestNilRegistryRefusesRegistration(t *testing.T) {
	var registry *KernelRegistry
	if registry.Get("x") != nil || registry.Register("x", nil) == nil {
		t.Fatal("nil registry did not refuse")
	}
}

func TestTypedBindingRejectsEqualSizeWrongDType(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{4}, cudair.Float32)
	y := g.AddUnary("neg", x)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	b := NewBindings()
	b.inputs["x"] = BufferBinding{typed: true, dtype: sliceDataType[int32](), Bytes: 16}
	if err := b.validateTypes(snap); err == nil {
		t.Fatal("equal-size int32 input admitted as float32")
	}
	binding := b.inputs["x"]
	binding.dtype = sliceDataType[float32]()
	b.inputs["x"] = binding
	if err := b.validateTypes(snap); err != nil {
		t.Fatal(err)
	}
}

func TestVendorWorkspaceSharesPeakBudget(t *testing.T) {
	for _, test := range []struct {
		scratch, peak, want int
		fail                bool
	}{
		{1, 256, 0, false}, {1, 512, 256, false}, {0, 32 * 1024 * 1024, 16 * 1024 * 1024, false}, {257, 511, 0, true}, {-1, 100, 0, true},
	} {
		got, err := vendorWorkspaceBudget(test.scratch, test.peak)
		if got != test.want || (err != nil) != test.fail {
			t.Fatalf("scratch=%d peak=%d got=%d err=%v", test.scratch, test.peak, got, err)
		}
	}
}

func TestBindingBorrowsPerInvocationAndCancellation(t *testing.T) {
	users := 0
	b := NewBindings()
	b.inputs["x"] = BufferBinding{borrow: func() (func(), error) {
		users++
		return func() { users-- }, nil
	}}
	for range 2 {
		releases, err := b.borrow()
		if err != nil || users != 1 {
			t.Fatalf("borrow users=%d err=%v", users, err)
		}
		c := newHostCompletion(func(ctx context.Context) error {
			if ctx != nil {
				return ctx.Err()
			}
			return nil
		}, nil, releases...)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if !errors.Is(c.Wait(ctx), context.Canceled) || users != 1 {
			t.Fatal("canceled wait released owners")
		}
		if err := c.Close(); err != nil || users != 0 {
			t.Fatalf("drain users=%d err=%v", users, err)
		}
		if err := c.Close(); err != nil || users != 0 {
			t.Fatal("release repeated")
		}
	}
}

func TestBindingBorrowFailureUnwinds(t *testing.T) {
	users := 0
	b := NewBindings()
	b.inputs["x"] = BufferBinding{borrow: func() (func(), error) {
		users++
		return func() { users-- }, nil
	}}
	b.outputs["y"] = BufferBinding{borrow: func() (func(), error) { return nil, errors.New("freed allocation") }}
	if _, err := b.borrow(); err == nil || users != 0 {
		t.Fatalf("users=%d error=%v", users, err)
	}
}

func TestStreamDrainFailureRetainsOwnerUntilRetry(t *testing.T) {
	attempts, releases := 0, 0
	c := newHostCompletion(func(context.Context) error {
		attempts++
		if attempts == 1 {
			return errors.New("native stream drain failed")
		}
		return nil
	}, nil, func() { releases++ })
	if c.Done() {
		t.Fatal("unobserved completion advertised done")
	}
	if err := c.Close(); err == nil || c.Done() || releases != 0 {
		t.Fatalf("first drain err=%v done=%v releases=%d", err, c.Done(), releases)
	}
	if err := c.Close(); err != nil || !c.Done() || releases != 1 {
		t.Fatalf("retry drain err=%v done=%v releases=%d", err, c.Done(), releases)
	}
	if err := c.Close(); err != nil || releases != 1 || attempts != 2 {
		t.Fatal("drain or release repeated")
	}
}

func TestExecutedBackendReportedPerStep(t *testing.T) {
	c := &Completion{steps: []ExecutedStep{{0, "x", BackendLaunch}, {1, "y", BackendCuBLASLt}}}
	got := c.ExecutedSteps()
	if len(got) != 2 || got[0].Backend != BackendLaunch || got[1].Backend != BackendCuBLASLt {
		t.Fatal(got)
	}
	got[0].Backend = BackendMega
	if c.ExecutedSteps()[0].Backend != BackendLaunch {
		t.Fatal("receipt aliases completion")
	}
	p := &ExecutablePlan{steps: []StepPlan{{NodeName: "unsupported", Backend: BackendMega}}}
	called := false
	if err := p.enqueueWith(nil, nil, func(int, StepPlan) { called = true }); !errors.Is(err, ErrUnsupportedBackend) || called {
		t.Fatalf("err=%v recorded=%v", err, called)
	}
}
