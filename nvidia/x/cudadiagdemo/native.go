package cudadiagdemo

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sync"
	"time"
	"unsafe"

	"github.com/tmc/lib/ebpfhost"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

const nativeSource = `extern "C" __global__ void transform(const float *input, float *output, int n) {
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < n) output[i] = 2.0f * input[i] + 1.0f;
}
`

// NativeConfig selects bounded physical CUDA trials. LockDir defaults to the
// canonical cudalock location. The caller must own CUPTI registration in this
// process. Native capture failures are retained as partial qualification.
type NativeConfig struct {
	Device    int
	Elements  int
	Rounds    int
	HostDelay time.Duration
	LockDir   string
	OutputDir string
}

// NativeCall is a directly measured host wrapper interval in the process-local
// monotonic epoch. These intervals are not kernel durations or scheduler evidence.
type NativeCall struct {
	Name    string `json:"name"`
	StartNS int64  `json:"start_ns"`
	EndNS   int64  `json:"end_ns"`
	Error   string `json:"error,omitempty"`
}

// NativeTrial retains a numerical result and host observations for one arm.
// CUPTI brackets belong to CUPTI's domain; no mapping to host spans is inferred.
type NativeTrial struct {
	Arm          string       `json:"arm"`
	Round        int          `json:"round"`
	DelayNS      int64        `json:"delay_ns"`
	WallNS       int64        `json:"wall_ns"`
	OutputSHA256 string       `json:"output_sha256"`
	Correct      bool         `json:"correct"`
	CUPTIBefore  uint64       `json:"cupti_before,omitempty"`
	CUPTIAfter   uint64       `json:"cupti_after,omitempty"`
	Calls        []NativeCall `json:"calls"`
}

// NativeResult separates physical numerical execution from diagnosis admission.
// Scheduler capture and a bounded host/CUPTI clock mapping are required before
// these observations can support a host submission-delay diagnosis.
type NativeResult struct {
	CleanupPending        bool `json:"cleanup_pending"`
	cleanup               *nativeCleanup
	HostClockDomain       string                        `json:"host_clock_domain"`
	HostClockEpochID      string                        `json:"host_clock_epoch_id"`
	NumericalGate         string                        `json:"numerical_gate"`
	ExecutionError        string                        `json:"execution_error,omitempty"`
	Synthetic             bool                          `json:"synthetic"`
	Qualification         string                        `json:"qualification"`
	RemainingGates        []string                      `json:"remaining_gates"`
	Device                int                           `json:"device"`
	Model                 string                        `json:"model"`
	UUID                  string                        `json:"uuid"`
	Architecture          string                        `json:"architecture"`
	DriverVersion         int                           `json:"driver_version"`
	Build                 string                        `json:"build"`
	ExecutableSHA256      string                        `json:"executable_sha256"`
	SourceSHA256          string                        `json:"source_sha256"`
	CUBINSHA256           string                        `json:"cubin_sha256"`
	InputSHA256           string                        `json:"input_sha256"`
	LeaseDir              string                        `json:"lease_dir"`
	LeaseOwner            cudalock.Owner                `json:"lease_owner"`
	ClockEpochUTC         time.Time                     `json:"clock_epoch_utc"`
	Trials                []NativeTrial                 `json:"trials"`
	Activity              cuptiactivity.Result          `json:"activity"`
	Annotations           []cudaworkload.HostSubmission `json:"annotations,omitempty"`
	ActivityAnalysis      *cudadiag.ActivityAnalysis    `json:"activity_analysis,omitempty"`
	ActivityAnalysisError string                        `json:"activity_analysis_error,omitempty"`
	ActivitySHA256        string                        `json:"activity_sha256,omitempty"`
	ActivityRun           string                        `json:"activity_run,omitempty"`
	CaptureError          string                        `json:"capture_error,omitempty"`
}

// RunNative executes an NVRTC-compiled vector operation on a physical GPU under
// cudalock, compares every output to an independent CPU oracle, and retains
// genuine CUPTI activity when supported. It never substitutes synthetic records.
// Successful numerical execution is explicitly partial diagnosis qualification.
func RunNative(ctx context.Context, cfg NativeConfig) (result *NativeResult, retErr error) {
	if cfg.Device < 0 || cfg.Elements <= 0 || cfg.Elements > 1<<20 || cfg.Rounds <= 0 || cfg.Rounds > 64 || cfg.HostDelay < 0 || cfg.HostDelay > 30*time.Second {
		return nil, errors.New("cudadiagdemo: invalid native trial bounds")
	}
	result = &NativeResult{HostClockDomain: "process_monotonic", NumericalGate: "unavailable", Qualification: "partial", Device: cfg.Device, SourceSHA256: digestBytes([]byte(nativeSource)), RemainingGates: []string{"scheduler observations", "host/CUPTI clock mapping", "collector-off/on overhead", "MIG scope metadata", "concurrent workload attribution", "device-bottleneck negative control"}}
	var clockID [16]byte
	if _, err := rand.Read(clockID[:]); err != nil {
		return result, err
	}
	result.HostClockEpochID = hex.EncodeToString(clockID[:])

	var cubin []byte
	// Export after all native resources and the lease have been released so the
	// final receipt includes cleanup failures rather than predicting success.
	defer func() {
		if retErr != nil {
			result.ExecutionError = retErr.Error()
			result.RemainingGates = append(result.RemainingGates, "native execution or cleanup failed")
		}
		if cfg.OutputDir != "" {
			retErr = errors.Join(retErr, exportNative(cfg.OutputDir, result, cubin))
		}
	}()

	if info, ok := debug.ReadBuildInfo(); ok {
		result.Build = info.String()
	}
	executable, err := os.Executable()
	if err != nil {
		return result, err
	}
	data, err := os.ReadFile(executable)
	if err != nil {
		return result, err
	}
	result.ExecutableSHA256 = digestBytes(data)
	options := []cudalock.Option{cudalock.WithLabel("gpudiagdemo-native")}
	if cfg.LockDir != "" {
		options = append(options, cudalock.WithDir(cfg.LockDir))
	}
	leaseCtx, cancelLease := context.WithTimeout(ctx, 30*time.Second)
	defer cancelLease()
	lease, err := cudalock.Acquire(leaseCtx, options...)
	if err != nil {
		return result, fmt.Errorf("native lease: %w", err)
	}
	result.LeaseDir, result.LeaseOwner = lease.Dir(), lease.Owner()
	result.cleanup = &nativeCleanup{}
	result.cleanup.add("lease", lease.Release)
	skipCleanup := false
	defer func() {
		if skipCleanup {
			result.CleanupPending = true
			return
		}
		retErr = errors.Join(retErr, result.Close())
	}()
	ownerBytes, err := os.ReadFile(filepath.Join(lease.Dir(), "owner"))
	if err != nil {
		return result, fmt.Errorf("native lease owner: %w", err)
	}
	result.LeaseOwner, err = cudalock.ParseOwner(ownerBytes)
	if err != nil {
		return result, err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev, err := cuda.NewDevice(cfg.Device)
	if err != nil {
		return result, fmt.Errorf("native device: %w", err)
	}
	result.cleanup.add("device", dev.Close)
	result.Model, err = dev.Name()
	if err != nil {
		return result, err
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		return result, err
	}
	result.UUID = hex.EncodeToString(uuid.Bytes[:])
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return result, err
	}
	result.Architecture = nvrtc.SmArch(major, minor)
	result.DriverVersion, err = driver.DriverGetVersion()
	if err != nil {
		return result, err
	}
	program, err := nvrtc.CreateProgram(nativeSource, "transform.cu")
	if err != nil {
		return result, err
	}
	result.cleanup.add("program", program.Destroy)
	if err := program.Compile("--gpu-architecture="+result.Architecture, "--fmad=false"); err != nil {
		log, _ := program.Log()
		return result, fmt.Errorf("native compile: %w: %s", err, log)
	}
	cubin, err = program.CUBIN()
	if err != nil {
		return result, err
	}
	result.CUBINSHA256 = digestBytes(cubin)
	var raw driver.Module
	if err := dev.WithContext(func() error { var err error; raw, err = driver.ModuleLoadData(cubin); return err }); err != nil {
		return result, err
	}
	result.cleanup.add("module", func() error { return dev.WithContext(raw.Close) })
	module, err := cuda.OwnModule(dev, raw)
	if err != nil {
		return result, err
	}
	result.cleanup.actions[len(result.cleanup.actions)-1].close = module.Close
	kernel, err := module.Function("transform")
	if err != nil {
		return result, err
	}
	input, err := cudamem.Alloc[float32](dev, cfg.Elements)
	if err != nil {
		return result, err
	}
	result.cleanup.add("input allocation", input.Free)
	output, err := cudamem.Alloc[float32](dev, cfg.Elements)
	if err != nil {
		return result, err
	}
	result.cleanup.add("output allocation", output.Free)
	host := make([]float32, cfg.Elements)
	got := make([]float32, cfg.Elements)
	for i := range host {
		host[i] = float32(i % 1024)
	}
	result.InputSHA256 = digestFloats(host)
	session, captureErr := cuptiactivity.Start(cuptiactivity.Config{Kinds: []cupti.ActivityKind{cupti.ActivityKindDriver, cupti.ActivityKindConcurrentKernel, cupti.ActivityKindMemcpy, cupti.ActivityKindExternalCorrelation}, BufferBytes: 1 << 20, MaxBuffers: 16, MaxRecords: 100000})
	if session != nil {
		result.cleanup.session = session
		result.cleanup.add("activity session", session.Close)
	}
	result.cleanup.add("synchronize", func() error { return dev.WithContext(driver.CtxSynchronize) })
	if captureErr != nil {
		result.CaptureError = captureErr.Error()
		result.RemainingGates = append(result.RemainingGates, "native activity capture")
		if session != nil {
			// Start already attempted rollback and retained ownership on failure.
			// Returning that ownership never automatically retries its failed gate.
			skipCleanup = true
			result.Activity = session.Result()
			return result, captureErr
		}
	}

	var annotator *cudaworkload.Annotator
	if captureErr == nil && session != nil {
		generation, err := ebpfhost.ProcessStartTime(os.Getpid())
		if err != nil {
			result.CaptureError = fmt.Sprintf("process generation: %v", err)
			return result, err
		}
		result.ActivityRun = "native-" + result.HostClockEpochID
		annotator = cudaworkload.NewAnnotator(cudaworkload.ProcessGenID{PID: uint32(os.Getpid()), Generation: generation}, "gpudiagdemo-native", cudaworkload.WithRunID(result.ActivityRun))
	}
	epoch := time.Now()
	result.ClockEpochUTC = epoch.UTC()
	for round := 0; round < cfg.Rounds; round++ {
		arms := []string{"baseline", "candidate"}
		if round%2 != 0 {
			arms[0], arms[1] = arms[1], arms[0]
		}
		for _, arm := range arms {
			if err := ctx.Err(); err != nil {
				return result, err
			}
			trial := NativeTrial{Arm: arm, Round: round + 1}
			start := time.Now()
			if arm == "baseline" {
				delayStart := time.Now()
				timer := time.NewTimer(cfg.HostDelay)
				select {
				case <-ctx.Done():
					timer.Stop()
					return result, ctx.Err()
				case <-timer.C:
				}
				trial.DelayNS = time.Since(delayStart).Nanoseconds()
			}
			call := func(name string, fn func() error) error {
				observation := NativeCall{Name: name, StartNS: time.Since(epoch).Nanoseconds()}
				err := fn()
				observation.EndNS = time.Since(epoch).Nanoseconds()
				if err != nil {
					observation.Error = err.Error()
				}
				trial.Calls = append(trial.Calls, observation)
				return err
			}
			if captureErr == nil {
				trial.CUPTIBefore, err = cupti.GetTimestamp()
				if err != nil {
					return result, err
				}
			}
			if err := call("cuMemcpyHtoD_v2", func() error { return input.CopyFromHost(host) }); err != nil {
				return result, err
			}
			inPtr, outPtr, n := input.Ptr(), output.Ptr(), int32(cfg.Elements)
			launch := func() error {
				return call("cuLaunchKernel", func() error {
					return kernel.Launch([3]uint32{uint32((cfg.Elements + 127) / 128), 1, 1}, [3]uint32{128, 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&inPtr), unsafe.Pointer(&outPtr), unsafe.Pointer(&n)})
				})
			}
			if annotator == nil {
				if err := launch(); err != nil {
					return result, err
				}
			} else {
				_, scope := annotator.Begin(ctx, fmt.Sprintf("%s-%d", arm, round))
				sub, err := scope.NativeOperation(ctx, cudaworkload.ContextStreamID{}, "cuLaunchKernel", result.SourceSHA256, launch)
				scope.End()
				if err != nil {
					return result, err
				}
				result.Annotations = append(result.Annotations, sub)
			}

			if err := call("cuCtxSynchronize", func() error { return dev.WithContext(driver.CtxSynchronize) }); err != nil {
				return result, err
			}
			if err := call("cuMemcpyDtoH_v2", func() error { return output.CopyToHost(got) }); err != nil {
				return result, err
			}
			if captureErr == nil {
				trial.CUPTIAfter, err = cupti.GetTimestamp()
				if err != nil {
					return result, err
				}
			}
			trial.WallNS = time.Since(start).Nanoseconds()
			trial.OutputSHA256 = digestFloats(got)
			if err := checkNativeOutput(host, got); err != nil {
				result.NumericalGate = "failed"
				result.Trials = append(result.Trials, trial)
				return result, err
			}
			trial.Correct = true
			result.Trials = append(result.Trials, trial)
		}
	}
	result.NumericalGate = "passed"
	return result, nil
}

func nativeCaptureComplete(r cuptiactivity.Result) bool {
	return r.Closed && len(r.Records) > 0 && r.PendingBuffers == 0 && r.RefusedBuffers == 0 && r.IncompleteRecords == 0 && r.DiscardedRecords == 0 && r.DroppedRecords == 0 && r.CallbackErrors == 0 && r.FirstError == nil
}
func digestBytes(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }
func digestFloats(values []float32) string {
	h := sha256.New()
	var b [4]byte
	for _, value := range values {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(value))
		h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}
func checkNativeOutput(input, output []float32) error {
	if len(input) == 0 || len(input) != len(output) {
		return errors.New("cudadiagdemo: native output length mismatch")
	}
	for i, value := range input {
		want := 2*value + 1
		if math.IsNaN(float64(output[i])) || math.IsInf(float64(output[i]), 0) || output[i] != want {
			return fmt.Errorf("cudadiagdemo: native output %d: got %g, want %g", i, output[i], want)
		}
	}
	return nil
}

func exportNative(dir string, result *NativeResult, cubin []byte) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	var activityRaw bytes.Buffer
	if len(result.Annotations) > 0 && len(result.Activity.Records) > 0 && !result.CleanupPending {
		writer := &nativeActivityWriter{buffer: &activityRaw, remaining: 32 << 20}
		for _, record := range result.Activity.Records {
			if err := json.NewEncoder(writer).Encode(record); err != nil {
				return err
			}
		}
		result.ActivitySHA256 = digestBytes(activityRaw.Bytes())
		gates := make(map[string]tracecontract.Gate)
		for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
			gates[name] = tracecontract.Gate{Disposition: "unchecked", Reason: "not independently qualified by activity capture", Provenance: "native demo"}
		}
		receipt := tracecontract.Receipt{Version: 1, Command: "gpudiagdemo run -native", InputPath: "activity.jsonl", InputSHA256: result.ActivitySHA256, Kind: "capture", Partial: true, ClockDomain: "cupti", ClockProvenance: "CUPTI timestamps in this retained collector session; no host mapping", Backends: []tracecontract.Backend{{Name: "cuptiactivity"}}, Gates: gates}
		batch := cudaworkload.ActivityBatch{Session: result.ActivityRun, Run: result.ActivityRun, ClockEpoch: result.ActivityRun, Process: result.Annotations[0].Process, EvidenceRef: result.ActivitySHA256, ExternalKind: cupti.ExternalCorrelationKindCustom0, MaxRecords: 65536}
		analysis, err := cudadiag.AnalyzeActivityJSONL(bytes.NewReader(activityRaw.Bytes()), batch, result.Annotations, receipt)
		if err != nil {
			result.ActivityAnalysisError = err.Error()
		} else {
			result.ActivityAnalysis = &analysis
		}
	}
	receipt, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	artifacts := []struct {
		name string
		data []byte
	}{{"transform.cu", []byte(nativeSource)}}
	if len(cubin) > 0 {
		artifacts = append(artifacts, struct {
			name string
			data []byte
		}{"transform.cubin", cubin})
	}
	artifacts = append(artifacts, struct {
		name string
		data []byte
	}{"native-receipt.json", receipt})
	if activityRaw.Len() > 0 {
		artifacts = append(artifacts, struct {
			name string
			data []byte
		}{"activity.jsonl", activityRaw.Bytes()})
	}
	for _, artifact := range artifacts {
		f, err := os.OpenFile(filepath.Join(dir, artifact.name), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
		if err != nil {
			return err
		}
		_, writeErr := f.Write(artifact.data)
		if err := errors.Join(writeErr, f.Close()); err != nil {
			return err
		}
	}
	return nil
}

type nativeCleanupAction struct {
	name  string
	close func() error
}
type nativeCleanup struct {
	mu      sync.Mutex
	actions []nativeCleanupAction
	session *cuptiactivity.Session
}

func (c *nativeCleanup) add(name string, close func() error) {
	c.actions = append(c.actions, nativeCleanupAction{name, close})
}

// Close explicitly retries pending native cleanup. RunNative attempts cleanup
// once; a failed action retains every remaining resource and the GPU lease.
// A startup session whose rollback failed is returned without an automatic
// retry. Call Close after investigating the failure. Successful actions are
// removed and never retried. The zero result has no cleanup ownership.
func (r *NativeResult) Close() error {
	if r == nil || r.cleanup == nil {
		return nil
	}
	c := r.cleanup
	c.mu.Lock()
	defer c.mu.Unlock()
	var err error
	for len(c.actions) > 0 {
		last := len(c.actions) - 1
		action := c.actions[last]
		if actionErr := action.close(); actionErr != nil {
			err = fmt.Errorf("native cleanup %s: %w", action.name, actionErr)
			break
		}
		c.actions = c.actions[:last]
	}
	r.CleanupPending = len(c.actions) > 0
	if c.session != nil {
		r.Activity = c.session.Result()
		if r.Activity.FirstError != nil {
			r.CaptureError = r.Activity.FirstError.Error()
		}
		if !nativeCaptureComplete(r.Activity) {
			addNativeGate(r, "complete loss-free activity capture")
		}
	}
	return err
}
func addNativeGate(r *NativeResult, gate string) {
	for _, existing := range r.RemainingGates {
		if existing == gate {
			return
		}
	}
	r.RemainingGates = append(r.RemainingGates, gate)
}

// nativeActivityWriter refuses an oversized raw export without truncating a record.
type nativeActivityWriter struct {
	buffer    *bytes.Buffer
	remaining int
}

func (w *nativeActivityWriter) Write(p []byte) (int, error) {
	if len(p) > w.remaining {
		return 0, errors.New("cudadiagdemo: activity export exceeds 32 MiB")
	}
	n, err := w.buffer.Write(p)
	w.remaining -= n
	return n, err
}
