//go:build linux && native

package cudahotswap

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap/stage"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type stageNativeObserver struct {
	replayBackend
	frame    *publicNativeFrame
	proof    *nativereceipt.Session
	receipt  *stageNativeReceipt
	expected map[string][6]uint32
	native   *cudaReplayBackend
	arm      string
}

func (o *stageNativeObserver) observation(call *stageCallable, data []byte, domain string, stream *cuda.Stream) (stageNativeObservation, error) {
	v := call.version
	if v == nil || v.artifact == nil || call.dev != o.frame.dev || len(v.spec.Functions) != 1 {
		return stageNativeObservation{}, errors.New("stage observer: actual stage owner required")
	}
	if stream == nil || call.stream != stream {
		return stageNativeObservation{}, errors.New("stage observer: actual stream owner differs")
	}
	r, err := stageNativeArtifactIdentity(v)
	if err != nil {
		return r, err
	}
	source := r.SourceSHA256
	want, ok := o.expected[source]
	if !ok || len(data) != 24 {
		return stageNativeObservation{}, errors.New("stage observer: source or telemetry extent differs")
	}
	image, err := replayPacketFile(v.artifact.Path(), 64<<20)
	if err != nil {
		return stageNativeObservation{}, err
	}
	r.Index, r.Arm = len(o.receipt.Observations), o.arm
	r.ArtifactSHA256 = fmt.Sprintf("%x", sha256.Sum256(image))
	r.ExpectedGeometry, r.Completed, r.StreamDomain = want, true, domain
	for i := range r.Geometry {
		r.Geometry[i] = binary.LittleEndian.Uint32(data[4*i:])
	}
	if r.Geometry != want {
		return r, fmt.Errorf("stage observer: actual geometry %v, want %v", r.Geometry, want)
	}
	o.frame.service.table.mu.RLock()
	g := o.frame.service.table.entries[v.spec.Functions[0].Kernel].current.Load()
	if g != nil {
		r.Function, r.KernelVersion = uint64(g.function), g.version
	}
	o.frame.service.table.mu.RUnlock()
	if r.Function == 0 || r.KernelVersion == 0 {
		return r, errors.New("stage observer: missing bound kernel owner")
	}
	err = call.dev.WithContext(func() error {
		h, err := driver.CtxGetCurrent()
		if err != nil {
			return err
		}
		r.Context = uint64(h)
		if err := call.stream.WithHandle(func(h driver.Stream) error { r.Stream = uint64(h); return nil }); err != nil {
			return err
		}
		return stream.WithHandle(func(h driver.Stream) error { r.ExpectedStream = uint64(h); return nil })
	})
	if err != nil {
		return r, err
	}
	if err := o.proof.Capture(fmt.Sprintf("stage-complete-%03d", r.Index), source, r.ArtifactSHA256); err != nil {
		return r, err
	}
	o.receipt.Observations = append(o.receipt.Observations, r)
	return r, nil
}

func (o *stageNativeObserver) Run(ctx context.Context, arm replayArm, spec replaySpec, timed bool) (time.Duration, error) {
	call, ok := arm.Kernel.(*stageCallable)
	if !ok {
		return 0, errors.New("stage observer: concrete stage callable required")
	}
	d, err := o.replayBackend.Run(ctx, arm, spec, timed)
	if err != nil {
		return d, err
	}
	data, err := o.replayBackend.Read(1)
	if err != nil {
		return d, err
	}
	_, err = o.observation(call, data, "replay", o.native.stream)
	return d, err
}

// This prepared consumer requires independent review before native dispatch.
func TestNativeStageServiceReplay(t *testing.T) {
	dir := publicNativeRequire(t)
	f := publicNativeFrameNew(t, filepath.Join(dir, "stage-frame"), 256)
	recording := f.service.loader.(*publicRecordingLoader)
	ctx := context.Background()
	kernel := Source{Filename: "stage-write.cu", Entry: "write_output", Code: `extern "C" __global__ void write_output(float *out, unsigned *geometry, int n) {
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i == 0) {
  geometry[0] = gridDim.x; geometry[1] = gridDim.y; geometry[2] = gridDim.z;
  geometry[3] = blockDim.x; geometry[4] = blockDim.y; geometry[5] = blockDim.z;
 }
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
	compiler := os.Getenv("HOTSWAP_STAGE_CC")
	if !filepath.IsAbs(compiler) {
		t.Fatal("absolute reviewed HOTSWAP_STAGE_CC is required")
	}
	builder, err := stage.NewBuilder(filepath.Join(dir, "stage-builds"), compiler)
	if err != nil {
		t.Fatal(err)
	}
	flags := []string{"-I" + filepath.Join(cudaPath, "include"), "-L" + filepath.Join(cudaPath, "lib64", "stubs"), "-lcuda"}
	proof, err := nativereceipt.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = proof.Close() })
	process, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	receipt := stageNativeReceipt{Version: 1, Process: process, Scope: "stage-library-only"}
	expected := make(map[string][6]uint32)
	var observer *stageNativeObserver
	baseReplay := f.service.newReplay
	f.service.newReplay = func(name string, spec replaySpec, policy replayPolicy, bindings []replayDeviceBinding) (replayBackend, error) {
		b, err := baseReplay(name, spec, policy, bindings)
		if err != nil {
			return b, err
		}
		native, ok := b.(*cudaReplayBackend)
		if !ok {
			return b, errors.New("stage observer: actual CUDA replay backend required")
		}
		observer = &stageNativeObserver{replayBackend: b, native: native, frame: f, proof: proof, receipt: &receipt, expected: expected, arm: "original"}
		return observer, nil
	}
	telemetry, err := cudamem.Alloc[byte](f.dev, 24)
	if telemetry.Ptr() != 0 {
		f.buffers = append(f.buffers, telemetry)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := telemetry.CopyFromHost(make([]byte, 24)); err != nil {
		t.Fatal(err)
	}
	source := func(block int, wrong bool) stage.Source {
		n := 256
		if wrong {
			n = 255
		}
		s := stage.Source{Flags: flags, Code: fmt.Sprintf(`#include <cuda.h>
#include <string.h>
int stage(struct stage_ctx *ctx) {
 if (!ctx || ctx->buffer_count != 2 || ctx->function_count != 1 ||
     strcmp(ctx->buffers[0].name, "output") || strcmp(ctx->functions[0].name, "entry") ||
     strcmp(ctx->buffers[1].name, "geometry") || ctx->buffers[1].bytes != 24 ||
     ctx->buffers[0].bytes != 1024) return -1;
 CUdeviceptr out = ctx->buffers[0].pointer;
 CUdeviceptr geometry = ctx->buffers[1].pointer;
 int n = %d;
 void *args[] = { &out, &geometry, &n };
 return (int)cuLaunchKernel((CUfunction)ctx->functions[0].handle,
     %d, 1, 1, %d, 1, 1, 0, (CUstream)ctx->stream, args, 0);
}`, n, 256/block, block)}
		expected[fmt.Sprintf("%x", sha256.Sum256([]byte(s.Code)))] = [6]uint32{uint32(256 / block), 1, 1, uint32(block), 1, 1}
		return s
	}
	schema := StageSpec{Buffers: []StageBuffer{{Name: "output", Bytes: 1024}, {Name: "geometry", Bytes: 24}}, Functions: []StageFunction{{Name: "entry", Kernel: "writer"}}}
	if _, err := f.service.RegisterStage(ctx, "stage", builder, source(64, false), schema, "original"); err != nil {
		t.Fatal(err)
	}
	fixture, err := f.service.RecordStageFixture(ctx, "stage", FixtureConfig{
		Args: []any{nil, nil}, Buffers: []FixtureBuffer{{Arg: 0, Bytes: 1024, Buffer: f.buffers[0]}, {Arg: 1, Bytes: 24, Buffer: telemetry}},
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
	receipt.Controls = append(receipt.Controls, "named_abi_refusal")
	packed := source(64, false)
	packed.Flags = append(append([]string(nil), packed.Flags...), "-fpack-struct=1")
	if result, err := f.service.PushStage(ctx, "stage", packed, schema, "compiler ABI mismatch"); err == nil || result == nil || result.Published {
		t.Fatalf("compiler ABI mismatch: %+v, %v", result, err)
	}
	receipt.Controls = append(receipt.Controls, "compiler_abi_refusal")
	for _, wrong := range []bool{true, false} {
		observer.arm = "wrong_tail_refusal"
		if !wrong {
			observer.arm = "geometry_change_pass"
		}
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
		receipt.Controls = append(receipt.Controls, observer.arm)
	}
	observer.arm = "checker_rollback"
	if _, err := f.service.SetCheck("stage", CheckConfig{Fixture: fixture, Checker: func(context.Context, State) error {
		return errors.New("application refuses selected stage")
	}}, "application negative"); err != nil {
		t.Fatal(err)
	}
	rollback, err := f.service.PushStage(ctx, "stage", source(64, false), schema, "hook refusal")
	if err == nil || rollback == nil || !rollback.Published || !rollback.RolledBack || rollback.Selected.Version == rollback.State.Version {
		t.Fatalf("application rollback: %+v, %v", rollback, err)
	}
	receipt.Controls = append(receipt.Controls, "checker_rollback")
	if _, err := f.service.SetCheck("stage", CheckConfig{Fixture: fixture}, "replay restored"); err != nil {
		t.Fatal(err)
	}
	// Keep the selected library/function owners pinned across a real variant
	// publication. Completion, not publication, releases the submitted owners.
	held, err := f.service.RunStage(ctx, "stage", StageLaunch{Buffers: []cudamem.Slice[byte]{f.buffers[0], telemetry}, Stream: f.stream})
	if err != nil {
		t.Fatal(err)
	}
	heldLibrary := held.call.version.library
	heldImage, err := replayPacketFile(held.call.version.artifact.Path(), 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err := proof.Capture("held-before-completion", held.call.version.state.SourceDigest, fmt.Sprintf("%x", sha256.Sum256(heldImage))); err != nil {
		t.Fatal(err)
	}
	if held.complete || held.call.pin == nil || len(held.call.functions) != 1 || !errors.Is(heldLibrary.Close(), stage.ErrPinned) {
		t.Fatal("submitted library/function ownership not retained")
	}
	g := held.call.functions[0].gen
	receipt.Ownership = stageNativeOwnership{Version: held.call.version.state.Version, KernelVersion: g.version, Function: uint64(g.function), BeforePending: !held.complete, BeforeLibraryPinned: true, BeforeFunctionsPinned: g.pins.Load() > 0}
	observer.arm = "swap_while_pinned"
	result, err := f.service.PushStage(ctx, "stage", source(64, false), schema, "swap while submitted")
	if heldErr := held.Wait(ctx); heldErr != nil {
		t.Fatal(heldErr)
	}
	heldData := make([]byte, 24)
	if err := telemetry.CopyToHost(heldData); err != nil {
		t.Fatal(err)
	}
	observer.arm = "held_submission"
	if _, err := observer.observation(held.call, heldData, "direct", f.stream); err != nil {
		t.Fatal(err)
	}
	receipt.Ownership.AfterComplete = held.complete
	receipt.Ownership.AfterFunctionsReleased = held.call.functions == nil
	receipt.Ownership.AfterLeaseReleased = held.cleanup.lease == nil
	if err := heldLibrary.Close(); err != nil {
		t.Fatal(err)
	}
	receipt.Ownership.AfterLibraryReleased = held.call.pin == nil
	if err != nil || result == nil || !result.Published {
		t.Fatalf("swap while submitted: %+v, %v", result, err)
	}
	receipt.Controls = append(receipt.Controls, "swap_while_pinned")
	if _, err := f.service.RevertStage(ctx, "stage", "original"); err != nil {
		t.Fatal(err)
	}
	pending, err := f.service.RunStage(ctx, "stage", StageLaunch{Buffers: []cudamem.Slice[byte]{f.buffers[0], telemetry}, Stream: f.stream})
	if err != nil {
		t.Fatal(err)
	}
	if err := pending.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	observer.arm = "revert_output"
	data := make([]byte, 24)
	if err := telemetry.CopyToHost(data); err != nil {
		t.Fatal(err)
	}
	if _, err := observer.observation(pending.call, data, "direct", f.stream); err != nil {
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
	receipt.Controls = append(receipt.Controls, "revert_output")
	if err := proof.Write(filepath.Join(dir, "stage-native-libraries.json")); err != nil {
		t.Fatal(err)
	}
	if err := proof.Close(); err != nil {
		t.Fatal(err)
	}
	mappings, err := nativereceipt.Read(filepath.Join(dir, "stage-native-libraries.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := validateStageNativeMappings(receipt, mappings); err != nil {
		t.Fatal(err)
	}
	if err := fixture.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if err := f.close(); err != nil {
		t.Fatal(err)
	}
	receipt.Cleanup = true
	if err := validateStageNativeReceipt(receipt); err != nil {
		t.Fatal(err)
	}
	if err := publicWriteJSON(filepath.Join(dir, "stage-native-receipt.json"), receipt); err != nil {
		t.Fatal(err)
	}
}
