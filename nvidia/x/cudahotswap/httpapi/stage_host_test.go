package httpapi

import (
	"context"
	"errors"
	"os/exec"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudahotswap/stage"
)

type hostStageLoader struct{}

func (hostStageLoader) Compile(context.Context, cudahotswap.Source) (*cudahotswap.Compilation, error) {
	return nil, errors.New("host-stage test must not compile CUDA")
}
func (hostStageLoader) Load(context.Context, *cudahotswap.Compilation) (cudahotswap.Kernel, error) {
	return nil, errors.New("host-stage test must not load CUDA")
}

func TestStageActualHostEndpointLifecycle(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("host stage library is supported on linux and darwin")
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Fatal(err)
	}
	builder, err := stage.NewBuilder(t.TempDir(), cc)
	if err != nil {
		t.Fatal(err)
	}
	service, err := cudahotswap.NewService(hostStageLoader{})
	if err != nil {
		t.Fatal(err)
	}
	checker := func(ctx context.Context, state cudahotswap.State) error {
		p, err := service.RunStage(ctx, state.Name, cudahotswap.StageLaunch{})
		if p != nil {
			err = errors.Join(err, p.Wait(ctx))
		}
		return err
	}
	h, err := New(service, Resources{Stages: map[string]StageBinding{"glue": {Builder: builder, Schema: cudahotswap.StageSpec{HostOnly: true}}}, Checkers: map[string]cudahotswap.Checker{"actual-C": checker}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := h.Close(context.Background()); err != nil {
			t.Error(err)
		}
	})
	original := "int stage(struct stage_ctx *ctx) { return ctx && ctx->version == 1 ? 0 : 91; }"
	r, status := stageRequest(t, h, "register_stage", Request{Name: "glue", Label: "original", StageCode: original})
	if status != 200 || r.Failure != nil || r.State == nil || r.State.Kind != "stage" || r.State.Label != "original" {
		t.Fatal(r, status)
	}
	initial := *r.State
	if _, status := stageRequest(t, h, "set_check", Request{Name: "glue", Label: "policy", Checker: "actual-C"}); status != 200 {
		t.Fatal(status)
	}
	wrong, status := stageRequest(t, h, "push_stage", Request{Name: "glue", Label: "wrong", StageCode: "int stage(struct stage_ctx *ctx) { return 17; }"})
	if status != 200 || wrong.Failure == nil || wrong.Push == nil || !wrong.Push.Published || !wrong.Push.RolledBack || wrong.Push.Report.Outcome != cudahotswap.OutcomeWrongAnswer {
		t.Fatal("missing actual C refusal/rollback", wrong, status)
	}
	r, status = stageRequest(t, h, "run_stage", Request{Name: "glue", Label: "restored"})
	if status != 200 || r.Failure != nil || r.Handle == "" || r.State != nil {
		t.Fatal("restored original cannot run", r, status)
	}
	completed, status := stageRequest(t, h, "wait", Request{Handle: r.Handle, Label: "complete"})
	if status != 200 || completed.Failure != nil || completed.Handle != "" || completed.State != nil {
		t.Fatal("stage owner not drained", completed, status)
	}
	good, status := stageRequest(t, h, "push_stage", Request{Name: "glue", Label: "good", StageCode: original + "\n/* repaired source */\n"})
	if status != 200 || good.Failure != nil || good.Push == nil || !good.Push.Published || good.Push.RolledBack {
		t.Fatal(good, status)
	}
	r, status = stageRequest(t, h, "revert_stage", Request{Name: "glue", Label: "revert"})
	if status != 200 || r.Failure != nil || r.State == nil || r.State.Version != initial.Version || r.State.SourceDigest != initial.SourceDigest || r.State.Generation <= initial.Generation || r.State.Label != "revert" {
		t.Fatal(r, status)
	}
}
