//go:build native

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

const resiliencePTX = `.version 8.0
.target sm_90
.address_size 64
.visible .entry resilience_store(.param .u64 output) {
.reg .u64 address;
ld.param.u64 address, [output];
st.global.u32 [address], 0x42280000;
ret;
}
`

func init() { resilienceNativeWork = resilienceRealDeviceWork }

func resilienceRealDeviceWork() error {
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		return err
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		return err
	}
	releaseDevice := false
	defer func() {
		if releaseDevice {
			dev.Close()
		}
	}()
	var value float32
	var workErr error
	id := cudaprof.Do(context.Background(), pprof.Labels("resilience", "actual-device"), func(context.Context) {
		workErr = dev.WithContext(func() error {
			module, err := driver.ModuleLoadData([]byte(resiliencePTX + "\x00"))
			if err != nil {
				return err
			}
			completed := false
			defer func() {
				if completed {
					module.Close()
				}
			}()
			fn, err := module.Function("resilience_store")
			if err != nil {
				return err
			}
			ptr, err := driver.MemAllocBytes(4)
			if err != nil {
				return err
			}
			defer func() {
				if completed {
					ptr.Free()
				}
			}()
			if err := fn.LaunchKernel(driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: 1, Y: 1, Z: 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&ptr)}); err != nil {
				return err
			}
			if err := driver.CtxSynchronize(); err != nil {
				return err
			}
			completed = true
			return ptr.CopyToHostPtr(unsafe.Pointer(&value), 4)
		})
	})
	if workErr != nil {
		return workErr
	}
	releaseDevice = true
	if id == 0 {
		return fmt.Errorf("actual CUPTI correlation push failed")
	}
	h := sha256.Sum256([]byte(resiliencePTX))
	input := sha256.Sum256(nil)
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return err
	}
	p := cudatest.CUDAOutputPolicy{Target: fmt.Sprintf("sm_%d%d", major, minor), DType: "fp32", MathMode: "constant-fp32", SourceDigest: hex.EncodeToString(h[:]), InputDigest: hex.EncodeToString(input[:]), Elements: 1, NaN: "reject", Infinity: "reject", Zero: "equal"}
	raw, err := json.Marshal(struct {
		Process      uint64
		ExternalID   uint64
		ActualBits   uint32
		ExpectedBits uint32
		Policy       cudatest.CUDAOutputPolicy
		Source       string
	}{uint64(os.Getpid()), id, math.Float32bits(value), math.Float32bits(42), p, resiliencePTX})
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(os.Getenv("CUDAPROF_TRANSPORT"), fmt.Sprintf("resilience-device-%d.json", os.Getpid())), raw, 0600); err != nil {
		return err
	}
	if err := cudatest.CompareCUDAOutputFP32([]float32{value}, []float32{42}, p); err != nil {
		return err
	}
	if err := cupti.ActivityFlushAll(0); err != nil {
		return err
	}
	// Go exits do not run ELF destructors. Invoke the existing agent export
	// after genuine device completion/flush; this generates no host activities.
	handle, err := purego.Dlopen(os.Getenv("CUDAPROF_RESILIENCE_AGENT"), purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return err
	}
	defer purego.Dlclose(handle)
	symbol, err := purego.Dlsym(handle, "cudaprof_finalize")
	if err != nil {
		return err
	}
	var finalize func() int
	purego.RegisterFunc(&finalize, symbol)
	if rc := finalize(); rc != 0 {
		return fmt.Errorf("target agent finalize returned %d", rc)
	}
	fmt.Printf("device_output=%g external_id=%d process=%d\n", value, id, os.Getpid())
	return nil
}

func resilienceNativeCapture(t *testing.T, mode, dir string) captureResult {
	t.Helper()
	if os.Getenv("CUDAPROF_RESILIENCE_NATIVE") != "1" {
		t.Skip("actual CUPTI capture qualification pending")
	}
	if runtime.GOOS != "linux" {
		t.Fatal("native capture requires Linux")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	agent := os.Getenv("CUDAPROF_RESILIENCE_AGENT")
	if agent == "" {
		t.Fatal("prebuilt native agent required")
	}
	var out, stderr bytes.Buffer
	env := setCaptureEnv(os.Environ(), "CUDAPROF_RESILIENCE_TARGET", mode)
	env = setCaptureEnv(env, "CUDAPROF_RESILIENCE_GPU", "1")
	r, err := runCapture(context.Background(), captureOptions{command: []string{os.Args[0], "-test.run=^TestCaptureResilience_Target$"}, Inject: agent, Env: env, FollowChildren: true, RecoverUnfinalized: true, Timeout: 40 * time.Second, DrainTimeout: time.Second, Stdout: &out, Stderr: &stderr})
	for name, data := range map[string][]byte{"stdout": out.Bytes(), "stderr": stderr.Bytes()} {
		if e := os.WriteFile(filepath.Join(dir, name), data, 0600); e != nil {
			t.Fatal(e)
		}
	}
	encoded, _ := json.Marshal(r)
	if e := os.WriteFile(filepath.Join(dir, "result.json"), encoded, 0600); e != nil {
		t.Fatal(e)
	}
	if err != nil {
		t.Fatalf("native capture: %v; %s", err, stderr.Bytes())
	}
	if r.TargetExitCode != 0 {
		t.Fatalf("target failed: %d %s", r.TargetExitCode, stderr.Bytes())
	}
	var kernels int
	for _, a := range r.Activities {
		if a.Subtype == "kernel" && a.Name == "resilience_store" && a.Valid && a.End > a.Start {
			kernels++
		}
	}
	want := 1
	if mode == "pair" {
		want = 2
	}
	if kernels != want {
		t.Fatalf("actual kernels=%d want%d; outcome=%s capture=%s", kernels, want, r.Outcome, r.CaptureError)
	}
	path := filepath.Join(dir, "capture.cudaprof")
	if _, err := writeCapture(path, r, captureArtifactBudget{FileBytes: 16 << 20, TotalBytes: 64 << 20}); err != nil {
		t.Fatal(err)
	}
	cap, err := readCapturePath(path)
	if err != nil {
		t.Fatal(err)
	}
	var matched int
	for _, sample := range cap.Profile.Sample {
		if names := sample.Label["name"]; len(names) == 1 && strings.Contains(names[0], "resilience_store") {
			if states := sample.Label["cupti.external.stack_status"]; len(states) == 1 && states[0] == "matched" {
				matched++
			}
		}
	}
	if matched != want {
		t.Fatalf("actual correlated kernel stacks=%d want%d", matched, want)
	}
	t.Logf("actual target PID=%d session=%s kernels=%d raw scratch=%s retained=%s", r.TargetPID, r.Session, kernels, r.ScratchDir, dir)
	return r
}

func resilienceNativeDir(t *testing.T) string {
	t.Helper()
	base := os.Getenv("CUDAPROF_RESILIENCE_OUTPUT")
	if base == "" {
		if os.Getenv("CUDAPROF_RESILIENCE_NATIVE") == "1" {
			t.Fatal("persistent native evidence directory required")
		}
		return t.TempDir()
	}
	dir := filepath.Join(base, t.Name())
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestCaptureResilience_NativeReexecTwoProcessesAndKilledWriter(t *testing.T) {
	dir := resilienceNativeDir(t)
	r := resilienceNativeCapture(t, "pair", dir)
	table, got := resilienceTable(t, r)
	if len(table.Stacks) < 4 || !got.damaged || table.DiscardedRecords < 1 {
		t.Fatal("native torn writer evidence lost")
	}
}

func TestCaptureResilience_NativeFreshEpochHasNoOldStacks(t *testing.T) {
	dir := resilienceNativeDir(t)
	for _, arm := range []string{"old", "fresh"} {
		if err := os.Mkdir(filepath.Join(dir, arm), 0700); err != nil {
			t.Fatal(err)
		}
	}
	a := resilienceNativeCapture(t, "old", filepath.Join(dir, "old"))
	b := resilienceNativeCapture(t, "fresh", filepath.Join(dir, "fresh"))
	if a.Session == b.Session || a.TargetPID == b.TargetPID {
		t.Fatal("native epoch reused")
	}
	table, _ := resilienceTable(t, b)
	for _, r := range table.Stacks {
		if r.Process != uint64(b.TargetPID) {
			t.Fatal("old native process stack inherited")
		}
	}
}
