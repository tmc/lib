package cudahotswap

import (
	"context"
	"errors"
	"io"
	"os/exec"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahotswap/stage"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// These are actual C libraries, in explicitly CPU-only mode. No synthetic
// CUDA function, PTX compilation or hardware coverage supports these tests.
const hostStageOriginal = `int stage(struct stage_ctx *ctx) { return ctx && ctx->version == 1 ? 0 : 91; }`
const hostStageWrong = `int stage(struct stage_ctx *ctx) { return 17; }`

func TestPendingStageZeroValueRefuses(t *testing.T) {
	var owner PendingStage
	if err := owner.Wait(t.Context()); err == nil {
		t.Fatal("zero-value owner was admitted")
	}
	if err := (*PendingStage)(nil).Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestStageReplayGeometryIsOwnedByCheckedStage(t *testing.T) {
	spec := fixtureSpec()
	spec.stageOwnedGeometry = true
	spec.Config = cudalaunch.Config{}
	policy := replayPolicy(DefaultReplayPolicy())
	if err := validateReplay(spec, policy); err != nil {
		t.Fatalf("stage-owned geometry: %v", err)
	}
	kernel := spec
	kernel.stageOwnedGeometry = false
	if err := validateReplay(kernel, policy); err == nil {
		t.Fatal("kernel accepted absent geometry")
	}
	for _, which := range []string{"grid", "block", "shared"} {
		t.Run(which, func(t *testing.T) {
			changed := spec
			switch which {
			case "grid":
				changed.Config.Grid.X = 1
			case "block":
				changed.Config.Block.X = 1
			case "shared":
				changed.Config.SharedMem = 1
			}
			if err := validateReplay(changed, policy); err == nil {
				t.Fatal("stage accepted kernel geometry override")
			}
		})
	}
}

func TestStageDeviceOwnerRefusedBeforeCompiler(t *testing.T) {
	s, builder, _ := newHostStageService(t)
	_, err := s.RegisterStage(t.Context(), "native", builder, stage.Source{Code: "#error must not compile"}, StageSpec{
		Buffers:   []StageBuffer{{Name: "output", Bytes: 16}},
		Functions: []StageFunction{{Name: "entry", Kernel: "writer"}},
	}, "native")
	if err == nil || !strings.Contains(err.Error(), "actual NVRTCLoader") {
		t.Fatalf("device owner admission: %v", err)
	}
}

func FuzzStageNamedSchema(f *testing.F) {
	f.Add("output", "entry", 16, false)
	f.Add("output", "output", 16, false)
	f.Add("", "", 0, true)
	f.Fuzz(func(t *testing.T, bufferName, functionName string, bytes int, hostOnly bool) {
		spec := StageSpec{HostOnly: hostOnly, Buffers: []StageBuffer{{Name: bufferName, Bytes: bytes}}, Functions: []StageFunction{{Name: functionName, Kernel: "writer"}}}
		checked, err := validateStageSpec(spec)
		if err != nil {
			return
		}
		if hostOnly || bytes <= 0 || bytes > replayMaxBytes || bufferName == "" || functionName == "" || bufferName == functionName || len(bufferName) > 128 || len(functionName) > 128 || strings.ContainsRune(bufferName, 0) || strings.ContainsRune(functionName, 0) {
			t.Fatal("invalid named schema admitted")
		}
		spec.Buffers[0].Name = "changed"
		spec.Functions[0].Name = "changed"
		if checked.Buffers[0].Name != bufferName || checked.Functions[0].Name != functionName {
			t.Fatal("schema aliases caller storage")
		}
	})
}

func newHostStageService(t *testing.T) (*Service, *stage.Builder, State) {
	t.Helper()
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("host shared libraries require Linux or Darwin")
	}
	if _, err := exec.LookPath("cc"); err != nil {
		t.Skip("cc unavailable")
	}
	loader := &serviceLoader{compile: func(context.Context, Source) (*Compilation, error) {
		t.Fatal("host stage invoked CUDA compiler")
		return nil, nil
	}}
	s, err := NewService(loader)
	if err != nil {
		t.Fatal(err)
	}
	s.acquire = func(context.Context, string) (io.Closer, error) {
		t.Fatal("host-only stage requested GPU lease")
		return nil, nil
	}
	b, err := stage.NewBuilder(t.TempDir(), "cc")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(context.Background()); err != nil {
			t.Error(err)
		}
	})
	state, err := s.RegisterStage(t.Context(), "stage", b, stage.Source{Code: hostStageOriginal}, StageSpec{HostOnly: true}, "original")
	if err != nil {
		t.Fatal(err)
	}
	return s, b, state
}

func hostStageCheck(s *Service) Checker {
	return func(ctx context.Context, selected State) error {
		pending, err := s.RunStage(ctx, selected.Name, StageLaunch{})
		if pending != nil {
			err = errors.Join(err, pending.Wait(ctx))
		}
		return err
	}
}

func TestStageActualHostWrongRevertAndChecks(t *testing.T) {
	s, _, original := newHostStageService(t)
	if _, err := s.SetCheck("stage", CheckConfig{Checker: hostStageCheck(s)}, "checker"); err != nil {
		t.Fatal(err)
	}
	wrong, err := s.PushStage(t.Context(), "stage", stage.Source{Code: hostStageWrong}, StageSpec{HostOnly: true}, "wrong")
	if err == nil || !strings.Contains(err.Error(), "17") || wrong == nil || !wrong.Published || !wrong.RolledBack || wrong.Selected.Version != original.Version || wrong.Selected.Generation <= wrong.State.Generation {
		t.Fatalf("wrong stage result=%+v err=%v", wrong, err)
	}
	good, err := s.PushStage(t.Context(), "stage", stage.Source{Code: hostStageOriginal + "\n/* valid variant */"}, StageSpec{HostOnly: true}, "good")
	if err != nil || !good.Published || good.RolledBack || good.Selected.Version == original.Version {
		t.Fatalf("good=%+v err=%v", good, err)
	}
	reverted, err := s.RevertStage(t.Context(), "stage", "original restored")
	if err != nil || reverted.Version != original.Version || reverted.Generation <= good.Selected.Generation {
		t.Fatalf("revert=%+v err=%v", reverted, err)
	}
	if err := hostStageCheck(s)(t.Context(), reverted); err != nil {
		t.Fatal(err)
	}
}

func TestStageNamedABIRefusedBeforeCompile(t *testing.T) {
	s, _, original := newHostStageService(t)
	changed := StageSpec{Buffers: []StageBuffer{{Name: "output", Bytes: 16}}, Functions: []StageFunction{{Name: "entry", Kernel: "must-be-real"}}}
	result, err := s.PushStage(t.Context(), "stage", stage.Source{Code: "#error must not compile"}, changed, "bad ABI")
	if err == nil || !strings.Contains(err.Error(), "ABI differs") || result != nil {
		t.Fatalf("ABI result=%+v err=%v", result, err)
	}
	state, err := s.Status("stage")
	if err != nil || state != original {
		t.Fatalf("refused ABI changed state: %+v %v", state, err)
	}
	if _, err := s.RecordStageFixture(t.Context(), "stage", FixtureConfig{}); err == nil || !strings.Contains(err.Error(), "actual native") {
		t.Fatalf("host replay fabricated=%v", err)
	}
}

func TestStageUnconfiguredCandidateIsRefused(t *testing.T) {
	s, _, original := newHostStageService(t)
	r, err := s.PushStage(t.Context(), "stage", stage.Source{Code: hostStageWrong}, StageSpec{HostOnly: true}, "unchecked")
	if err == nil || r.Published || r.Selected.Version != original.Version {
		t.Fatalf("unchecked result=%+v err=%v", r, err)
	}
}

func TestStageConcurrentHookProtectsRollbackTarget(t *testing.T) {
	s, _, original := newHostStageService(t)
	started, release := make(chan struct{}), make(chan struct{})
	checker := func(ctx context.Context, state State) error {
		if err := hostStageCheck(s)(ctx, state); err != nil {
			return err
		}
		close(started)
		<-release
		return errors.New("application control")
	}
	if _, err := s.SetCheck("stage", CheckConfig{Checker: checker}, "hook"); err != nil {
		t.Fatal(err)
	}
	type answer struct {
		r   *PushResult
		err error
	}
	done := make(chan answer, 1)
	go func() {
		r, err := s.PushStage(t.Context(), "stage", stage.Source{Code: hostStageOriginal + "\n/* pending hook */"}, StageSpec{HostOnly: true}, "candidate")
		done <- answer{r, err}
	}()
	<-started
	for _, operation := range []func() error{
		func() error {
			_, err := s.PushStage(t.Context(), "stage", stage.Source{Code: hostStageOriginal}, StageSpec{HostOnly: true}, "interloper")
			return err
		},
		func() error { _, err := s.RevertStage(t.Context(), "stage", "interloper"); return err },
		func() error { return s.Close(t.Context()) },
	} {
		if err := operation(); err == nil || !strings.Contains(err.Error(), "check is active") {
			t.Errorf("hook admitted mutation/close: %v", err)
		}
	}
	close(release)
	a := <-done
	if a.err == nil || a.r == nil || !a.r.RolledBack || a.r.Selected.Version != original.Version {
		t.Fatalf("hook rollback=%+v err=%v", a.r, a.err)
	}
}

func TestStagePendingOwnersDrainOnClose(t *testing.T) {
	s, _, _ := newHostStageService(t)
	p, err := s.RunStage(t.Context(), "stage", StageLaunch{})
	if err != nil || p == nil {
		t.Fatalf("run=%v %v", p, err)
	}
	if err := s.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if !p.complete || p.call.pin != nil {
		t.Fatal("close did not drain pending host generation")
	}
	if err := p.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := s.stages.entries["stage"].current.library.Pin(); err == nil {
		t.Fatal("closed service library remained callable")
	}
}

func TestStageSpecCopiesAndBounds(t *testing.T) {
	spec := StageSpec{Buffers: []StageBuffer{{"output", 16}}, Functions: []StageFunction{{"entry", "kernel"}}}
	got, err := validateStageSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	spec.Buffers[0].Name = "mutated"
	spec.Functions[0].Kernel = "mutated"
	if got.Buffers[0].Name != "output" || got.Functions[0].Kernel != "kernel" {
		t.Fatal("caller mutated admitted ABI")
	}
	for _, bad := range []StageSpec{{}, {HostOnly: true, Buffers: got.Buffers}, {Buffers: []StageBuffer{{"same", 1}}, Functions: []StageFunction{{"same", "kernel"}}}, {Buffers: []StageBuffer{{"out", 0}}, Functions: got.Functions}} {
		if _, err := validateStageSpec(bad); err == nil {
			t.Fatalf("invalid schema admitted %+v", bad)
		}
	}
}

func TestStageReportEventsPreserveRequestIdentity(t *testing.T) {
	s, _, _ := newHostStageService(t)
	if _, err := s.SetCheck("stage", CheckConfig{Checker: hostStageCheck(s)}, "check config"); err != nil {
		t.Fatal(err)
	}
	sub, err := s.Subscribe(t.Context(), 0)
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Close()
	result, err := s.PushStage(t.Context(), "stage", stage.Source{Code: hostStageOriginal + "\n/* event identity */"}, StageSpec{HostOnly: true}, "exact stage label")
	if err != nil {
		t.Fatal(err)
	}
	select {
	case event := <-sub.Events():
		if event.Name != result.State.Name || event.Label != result.State.Label || event.Version != result.State.Version || event.Generation != result.State.Generation || event.ConfigRevision != result.State.ConfigRevision || event.Report == nil || event.Report.Generation != result.State.Generation || event.Report.ConfigRevision != result.State.ConfigRevision || event.Report.Compile.SourceSHA256 != result.State.SourceDigest {
			t.Fatalf("stage request identity drift: %+v; result %+v", event, result)
		}
		result.Report.Compile.SourceSHA256 = "caller mutation"
		if event.Report.Compile.SourceSHA256 == "caller mutation" {
			t.Fatal("stage event retained caller-owned report")
		}
	default:
		t.Fatal("accepted actual host stage did not publish its report")
	}
	r := testReportEvents(t)
	for _, kind := range []string{"", "host", "anything", "kernel stage"} {
		state := eventState()
		state.Kind = kind
		if _, err := r.append(state, &Report{}); err == nil {
			t.Fatalf("arbitrary report kind %q admitted", kind)
		}
	}
}

func TestStageCanceledCheckerRestoresRetainedSelection(t *testing.T) {
	s, _, original := newHostStageService(t)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	if _, err := s.SetCheck("stage", CheckConfig{Checker: func(context.Context, State) error { cancel(); return context.Canceled }}, "canceling checker"); err != nil {
		t.Fatal(err)
	}
	result, err := s.PushStage(ctx, "stage", stage.Source{Code: hostStageWrong}, StageSpec{HostOnly: true}, "unchecked")
	if !errors.Is(err, context.Canceled) || result == nil || !result.Published || !result.RolledBack || result.Selected.Version != original.Version || result.Selected.Generation <= result.State.Generation {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	states := s.List()
	if len(states) != 1 || states[0].Version != original.Version {
		t.Fatalf("selected=%+v", states)
	}
	selected := states[0]
	// A fresh request must invoke the restored actual C body, never the unchecked
	// candidate whose return value is 17.
	if err := hostStageCheck(s)(t.Context(), selected); err != nil {
		t.Fatalf("unchecked candidate remained launchable: %v", err)
	}
}

func TestStageCheckerPublicationReservesRollbackGeneration(t *testing.T) {
	s, _, original := newHostStageService(t)
	if _, err := s.SetCheck("stage", CheckConfig{Checker: hostStageCheck(s)}, "checker"); err != nil {
		t.Fatal(err)
	}
	s.mutations.Lock()
	s.stages.next = ^uint64(0) - 1
	s.mutations.Unlock()
	result, err := s.PushStage(t.Context(), "stage", stage.Source{Code: hostStageWrong}, StageSpec{HostOnly: true}, "exhausted")
	if err == nil || result == nil || result.Published || !strings.Contains(err.Error(), "generation counter exhausted") {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	states := s.List()
	if len(states) != 1 || states[0].Version != original.Version {
		t.Fatalf("selected=%+v", states)
	}
	if err := hostStageCheck(s)(t.Context(), states[0]); err != nil {
		t.Fatal(err)
	}
}

func TestStageOtherRegistrationCannotConsumeRollbackReservation(t *testing.T) {
	for _, tc := range []struct {
		name           string
		remaining      uint64
		wantRegistered bool
	}{
		{"enough for interleaved registration", 3, true},
		{"rollback reservation excludes registration", 2, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, builder, original := newHostStageService(t)
			var other State
			var registrationErr error
			if _, err := s.SetCheck("stage", CheckConfig{Checker: func(ctx context.Context, _ State) error {
				other, registrationErr = s.RegisterStage(ctx, "other", builder, stage.Source{Code: hostStageOriginal}, StageSpec{HostOnly: true}, "other")
				return errors.New("refused by checker")
			}}, "reentrant checker"); err != nil {
				t.Fatal(err)
			}
			s.mutations.Lock()
			s.stages.next = ^uint64(0) - tc.remaining
			s.mutations.Unlock()
			result, err := s.PushStage(t.Context(), "stage", stage.Source{Code: hostStageWrong}, StageSpec{HostOnly: true}, "unchecked")
			if err == nil || result == nil || !result.Published || !result.RolledBack || result.Selected.Version != original.Version || result.Selected.Generation != ^uint64(0) {
				t.Fatalf("result=%+v err=%v", result, err)
			}
			if (registrationErr == nil) != tc.wantRegistered {
				t.Fatalf("other=%+v registration error=%v", other, registrationErr)
			}
			if tc.wantRegistered && other.Generation != result.State.Generation+1 {
				t.Fatalf("other generation=%d candidate=%d", other.Generation, result.State.Generation)
			}
			if !tc.wantRegistered && !strings.Contains(registrationErr.Error(), "generation counter exhausted") {
				t.Fatal(registrationErr)
			}
			if err := hostStageCheck(s)(t.Context(), result.Selected); err != nil {
				t.Fatalf("unchecked candidate still selected: %v", err)
			}
			s.mutations.RLock()
			reserved := len(s.stages.rollback)
			s.mutations.RUnlock()
			if reserved != 0 {
				t.Fatalf("leaked reservations: %d", reserved)
			}
		})
	}
}

func TestStageSuccessfulCheckerReleasesRollbackReservation(t *testing.T) {
	s, builder, _ := newHostStageService(t)
	if _, err := s.SetCheck("stage", CheckConfig{Checker: func(context.Context, State) error { return nil }}, "accept"); err != nil {
		t.Fatal(err)
	}
	s.mutations.Lock()
	s.stages.next = ^uint64(0) - 2
	s.mutations.Unlock()
	result, err := s.PushStage(t.Context(), "stage", stage.Source{Code: hostStageOriginal}, StageSpec{HostOnly: true}, "checked")
	if err != nil || result == nil || !result.Published || result.RolledBack {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	other, err := s.RegisterStage(t.Context(), "other", builder, stage.Source{Code: hostStageOriginal}, StageSpec{HostOnly: true}, "other")
	if err != nil || other.Generation != ^uint64(0) {
		t.Fatalf("other=%+v err=%v", other, err)
	}
}
