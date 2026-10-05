//go:build linux && native

package cudahotswap

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahotswap/stage"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// This prepared consumer requires independent review before native dispatch.
func TestNativeStageServiceReplay(t *testing.T) {
	dir := publicNativeRequire(t)
	f := publicNativeFrameNew(t, filepath.Join(dir, "stage-frame"), 256)
	recording := f.service.loader.(*publicRecordingLoader)
	ctx := context.Background()
	kernel := Source{Filename: "stage-write.cu", Entry: "write_output", Code: `extern "C" __global__ void write_output(float *out, int n) {
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < n) out[i] = (float)i;
}`}
	if _, err := f.service.Register(ctx, "writer", kernel, "actual writer"); err != nil {
		t.Fatal(err)
	}
	// Preserve the actual NVRTC source/PTX/CUBIN/dependency receipts, then
	// restore the concrete device owner required by stage admission.
	f.service.loader = recording.base
	cudaPath := os.Getenv("CUDA_PATH")
	if !filepath.IsAbs(cudaPath) {
		t.Fatal("absolute CUDA_PATH is required")
	}
	builder, err := stage.NewBuilder(filepath.Join(dir, "stage-builds"), "cc")
	if err != nil {
		t.Fatal(err)
	}
	flags := []string{"-I" + filepath.Join(cudaPath, "include"), "-L" + filepath.Join(cudaPath, "lib64", "stubs"), "-lcuda"}
	source := func(block int, wrong bool) stage.Source {
		n := 256
		if wrong {
			n = 255
		}
		return stage.Source{Flags: flags, Code: fmt.Sprintf(`#include <cuda.h>
#include <string.h>
int stage(struct stage_ctx *ctx) {
 if (!ctx || ctx->buffer_count != 1 || ctx->function_count != 1 ||
     strcmp(ctx->buffers[0].name, "output") || strcmp(ctx->functions[0].name, "entry") ||
     ctx->buffers[0].bytes != 1024) return -1;
 CUdeviceptr out = ctx->buffers[0].pointer;
 int n = %d;
 void *args[] = { &out, &n };
 return (int)cuLaunchKernel((CUfunction)ctx->functions[0].handle,
     %d, 1, 1, %d, 1, 1, 0, (CUstream)ctx->stream, args, 0);
}`, n, 256/block, block)}
	}
	schema := StageSpec{Buffers: []StageBuffer{{Name: "output", Bytes: 1024}}, Functions: []StageFunction{{Name: "entry", Kernel: "writer"}}}
	if _, err := f.service.RegisterStage(ctx, "stage", builder, source(64, false), schema, "original"); err != nil {
		t.Fatal(err)
	}
	fixture, err := f.service.RecordStageFixture(ctx, "stage", FixtureConfig{
		Args: []any{nil}, Buffers: []FixtureBuffer{{Arg: 0, Bytes: 1024, Buffer: f.buffers[0]}},
		Outputs: []FixtureOutput{{Arg: 0, Elem: "f32"}}, Policy: DefaultReplayPolicy(),
		Launch: cudalaunch.Config{Stream: f.stream},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.service.SetCheck("stage", CheckConfig{Fixture: fixture}, "same replay"); err != nil {
		t.Fatal(err)
	}
	changed := StageSpec{Buffers: []StageBuffer{{Name: "renamed", Bytes: 1024}}, Functions: schema.Functions}
	if result, err := f.service.PushStage(ctx, "stage", stage.Source{Code: "#error must not compile"}, changed, "changed named ABI"); err == nil || result != nil {
		t.Fatalf("changed named ABI: %+v, %v", result, err)
	}
	packed := source(64, false)
	packed.Flags = append(append([]string(nil), packed.Flags...), "-fpack-struct=1")
	if result, err := f.service.PushStage(ctx, "stage", packed, schema, "compiler ABI mismatch"); err == nil || result == nil || result.Published {
		t.Fatalf("compiler ABI mismatch: %+v, %v", result, err)
	}
	for _, wrong := range []bool{true, false} {
		result, err := f.service.PushStage(ctx, "stage", source(128, wrong), schema, fmt.Sprintf("wrong=%t", wrong))
		if writeErr := publicWriteJSON(filepath.Join(dir, fmt.Sprintf("stage-wrong-%t.json", wrong)), struct {
			Result *PushResult
			Error  string
		}{result, fmt.Sprint(err)}); writeErr != nil {
			t.Fatal(writeErr)
		}
		if result == nil || result.Published == wrong || (err != nil) != wrong || result.Report.Replay == nil || !result.Report.Replay.Native {
			t.Fatalf("replay: %+v, %v", result, err)
		}
	}
	if _, err := f.service.SetCheck("stage", CheckConfig{Fixture: fixture, Checker: func(context.Context, State) error {
		return errors.New("application refuses selected stage")
	}}, "application negative"); err != nil {
		t.Fatal(err)
	}
	rollback, err := f.service.PushStage(ctx, "stage", source(64, false), schema, "hook refusal")
	if err == nil || rollback == nil || !rollback.Published || !rollback.RolledBack || rollback.Selected.Version == rollback.State.Version {
		t.Fatalf("application rollback: %+v, %v", rollback, err)
	}
	if _, err := f.service.SetCheck("stage", CheckConfig{Fixture: fixture}, "replay restored"); err != nil {
		t.Fatal(err)
	}
	// Keep the selected library/function owners pinned across a real variant
	// publication. Completion, not publication, releases the submitted owners.
	held, err := f.service.RunStage(ctx, "stage", StageLaunch{Buffers: []cudamem.Slice[byte]{f.buffers[0]}, Stream: f.stream})
	if err != nil {
		t.Fatal(err)
	}
	result, err := f.service.PushStage(ctx, "stage", source(64, false), schema, "swap while submitted")
	if heldErr := held.Wait(ctx); heldErr != nil {
		t.Fatal(heldErr)
	}
	if err != nil || result == nil || !result.Published {
		t.Fatalf("swap while submitted: %+v, %v", result, err)
	}
	if _, err := f.service.RevertStage(ctx, "stage", "original"); err != nil {
		t.Fatal(err)
	}
	pending, err := f.service.RunStage(ctx, "stage", StageLaunch{Buffers: []cudamem.Slice[byte]{f.buffers[0]}, Stream: f.stream})
	if err != nil {
		t.Fatal(err)
	}
	if err := pending.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	view, err := cudamem.Reinterpret[float32](f.buffers[0])
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 256)
	if err := view.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	for i, value := range got {
		if value != float32(i) {
			t.Fatalf("output[%d]=%g", i, value)
		}
	}
}
