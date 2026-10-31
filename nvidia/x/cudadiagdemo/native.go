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
	"github.com/tmc/lib/nvidia/x/cudaregress"
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
//
// Each round plans baseline, candidate and device request attempts, which all
// upload the same round input to their own buffers; see NativeResult.Recipe.
// Every planned wrapper observation, Rounds*(DeviceIterations+11), must be at
// most 65536; launches are never dropped to fit. Concurrency
// bounds the request workers: zero means 4, 1 runs attempts serially in plan
// order, and the maximum is 8. Each attempt owns its stream and input and
// output buffers, so the workers that can run at once (at most 3*Rounds) must
// keep 8*Elements bytes each within 32 MiB of device buffers.
//
// A nonempty OutputDir retains each attempt's input, CPU reference and
// downloaded output vectors there; see NativeNumerics. Their worst case,
// 36*Rounds*Elements bytes, must be at most 64 MiB. An empty OutputDir
// retains no files.
type NativeConfig struct {
	Device    int
	Elements  int
	Rounds    int
	HostDelay time.Duration
	LockDir   string
	OutputDir string
	// DeviceIterations repeats the same kernel in the device-negative arm. Zero means 16; otherwise 2 through 1024.
	DeviceIterations int
	Concurrency      int
	Composition      NativeCompositionOptions
}

// NativeCall is a directly measured host wrapper interval in the process-local
// monotonic epoch, taken immediately before and after the wrapped call. These
// intervals are not kernel durations, CUDA API entry times or scheduler
// evidence. With a CUPTI timestamp source, Before and After are raw samples
// taken just outside the interval; a failed sample never changes Error, the
// call's own failure.
type NativeCall struct {
	Name    string        `json:"name"`
	StartNS int64         `json:"start_ns"`
	EndNS   int64         `json:"end_ns"`
	Error   string        `json:"error,omitempty"`
	Before  *NativeSample `json:"before,omitempty"`
	After   *NativeSample `json:"after,omitempty"`
}

// NativeTrial retains a numerical result and host observations for one request
// attempt. CUPTI brackets belong to CUPTI's domain; no mapping to host spans is
// inferred. Trials are in plan order, not execution order; Worker identifies the
// request worker that ran the attempt. Each attempt has its own input, checked
// by its own oracle; all arms of a round have the same input values. Error
// records why a failed or canceled attempt stopped. Timing is the attempt's
// raw launch-ready and timestamp observations.
//
// Correlation records how the launches relate to CUPTI external correlation:
// "complete" when the native operation succeeded; "partial" when its launch
// function ran under correlation and then the operation failed; "not_entered"
// when it failed before the launch function ran, so no launch was made and
// the thread is unknown; "uninstrumented" without capture; and empty when the
// attempt stopped before its launches. A partial annotation is the scope's
// observation at entry on the launching thread; no native correlation ID is
// inferred.
type NativeTrial struct {
	// ContextStream contains independently queried CUPTI identifiers, not raw
	// driver handles or values copied from the activity records being joined.
	// Nil means no successful identity query was retained. It establishes no activity
	// coverage, numerical correctness or native qualification.
	ContextStream     *cudaworkload.ContextStreamID `json:"context_stream,omitempty"`
	OperationName     string                        `json:"operation_name,omitempty"`
	RequestedLaunches int                           `json:"requested_launches"`
	KernelLaunches    int                           `json:"kernel_launches"`
	RequestID         string                        `json:"request_id"`
	Attempt           string                        `json:"attempt"`
	Arm               string                        `json:"arm"`
	Round             int                           `json:"round"`
	Worker            int                           `json:"worker"`
	DelayNS           int64                         `json:"delay_ns"`
	WallNS            int64                         `json:"wall_ns"`
	InputSHA256       string                        `json:"input_sha256"`
	OutputSHA256      string                        `json:"output_sha256"`
	Correct           bool                          `json:"correct"`
	Correlation       string                        `json:"correlation,omitempty"`
	CUPTIBefore       uint64                        `json:"cupti_before,omitempty"`
	CUPTIAfter        uint64                        `json:"cupti_after,omitempty"`
	Calls             []NativeCall                  `json:"calls"`
	Error             string                        `json:"error,omitempty"`
	Timing            *NativeTiming                 `json:"timing,omitempty"`
}

// NativeResult separates physical numerical execution from diagnosis admission.
// Scheduler capture and a bounded host/CUPTI clock mapping are required before
// these observations can support a host submission-delay diagnosis.
// Requests and Trials retain every dispatched attempt, including failed and
// canceled ones, in plan order; after a failure, later planned requests are not
// dispatched. Concurrency is the number of request workers used. InputSHA256
// is the input of the first planned request.
//
// Recipe is launch-ready-v1: every attempt enqueues its upload before its
// launch-ready observation and the baseline's intentional wait, and all arms
// of round r use the same input seed, r-1. Both differ from unversioned
// earlier results, which this recipe does not qualify. Timing references the
// exported native-timing.jsonl, one raw line per dispatched attempt; on a
// failed export it is nil and TimingError records why.
type NativeResult struct {
	Requests              []NativeRequestLog `json:"requests"`
	PlannedRequests       int                `json:"planned_requests"`
	Concurrency           int                `json:"concurrency"`
	Composition           *NativeComposition `json:"composition,omitempty"`
	CleanupPending        bool               `json:"cleanup_pending"`
	cleanup               *nativeCleanup
	vectors               []nativeVectors
	HostClockDomain       string                        `json:"host_clock_domain"`
	HostClockEpochID      string                        `json:"host_clock_epoch_id"`
	NumericalGate         string                        `json:"numerical_gate"`
	Numerics              NativeNumerics                `json:"numerics"`
	Recipe                string                        `json:"recipe,omitempty"`
	Timing                *cudaregress.ArtifactRef      `json:"timing,omitempty"`
	TimingError           string                        `json:"timing_error,omitempty"`
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
// With an OutputDir, the retained vectors replay each attempt's numerical
// check; they do not qualify diagnosis.
// Every request worker quiesces before capture close, resource teardown and
// lease release.
func RunNative(ctx context.Context, cfg NativeConfig) (result *NativeResult, retErr error) {
	if ctx == nil {
		return nil, errors.New("cudadiagdemo: nil native context")
	}
	if cfg.Device < 0 || cfg.Elements <= 0 || cfg.Elements > 1<<20 || cfg.Rounds <= 0 || cfg.Rounds > 64 || cfg.HostDelay < 0 || cfg.HostDelay > 30*time.Second || cfg.DeviceIterations < 0 || cfg.DeviceIterations == 1 || cfg.DeviceIterations > 1024 {
		return nil, errors.New("cudadiagdemo: invalid native trial bounds")
	}
	workers, err := nativeWorkers(cfg)
	if err != nil {
		return nil, err
	}
	numerics, err := nativeNumericsFor(cfg)
	if err != nil {
		return nil, err
	}
	if cfg.DeviceIterations == 0 {
		cfg.DeviceIterations = 16
	}
	if _, err := nativeTimingCalls(cfg.Rounds, cfg.DeviceIterations); err != nil {
		return nil, err
	}
	plan := planNativeRequests(cfg.Rounds, cfg.DeviceIterations)
	result = &NativeResult{PlannedRequests: len(plan), Concurrency: workers, HostClockDomain: "process_monotonic", NumericalGate: "unavailable", Qualification: "partial", Numerics: numerics, Recipe: nativeRecipe, Device: cfg.Device, SourceSHA256: digestBytes([]byte(nativeSource)), RemainingGates: []string{"scheduler observations", "host/CUPTI clock mapping", "collector-off/on overhead", "MIG scope metadata", "concurrent workload attribution", "physical device-negative attribution"}}
	var clockID [16]byte
	if _, err := rand.Read(clockID[:]); err != nil {
		return result, err
	}
	result.HostClockEpochID = hex.EncodeToString(clockID[:])
	result.ActivityRun = "native-" + result.HostClockEpochID

	var cubin []byte
	// Export after all native resources and the lease have been released so the
	// final receipt includes cleanup failures rather than predicting success.
	defer func() {
		if retErr != nil {
			result.ExecutionError = retErr.Error()
			result.RemainingGates = append(result.RemainingGates, "native execution or cleanup failed")
		}
		composition, compositionErr := ComposeNativeEvidence(ctx, result, cfg.Composition)
		result.Composition = &composition
		retErr = errors.Join(retErr, compositionErr)
		if compositionErr != nil {
			result.ExecutionError = retErr.Error()
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
	result.cleanup = &nativeCleanup{}
	skipCleanup := false
	defer func() {
		retErr = finishNativeCleanup(result, retErr, skipCleanup)
	}()
	lease, err := constructNativeOwner(result.cleanup, "lease", func() (*cudalock.Handle, error) {
		return cudalock.Acquire(leaseCtx, options...)
	}, func(owner *cudalock.Handle) bool {
		return owner != nil
	}, func(owner **cudalock.Handle) error {
		return (*owner).Release()
	})
	if err != nil {
		return result, fmt.Errorf("native lease: %w", err)
	}
	result.LeaseDir, result.LeaseOwner = lease.Dir(), lease.Owner()
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
	dev, err := constructNativeOwner(result.cleanup, "device", func() (*cuda.Device, error) {
		return cuda.NewDevice(cfg.Device)
	}, func(owner *cuda.Device) bool {
		return owner != nil
	}, func(owner **cuda.Device) error {
		return (*owner).Close()
	})
	if err != nil {
		return result, fmt.Errorf("native device: %w", err)
	}
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
	program, err := constructNativeOwner(result.cleanup, "program", func() (nvrtc.Program, error) {
		return nvrtc.CreateProgram(nativeSource, "transform.cu")
	}, func(owner nvrtc.Program) bool {
		return owner != 0
	}, func(owner *nvrtc.Program) error {
		return owner.Destroy()
	})
	if err != nil {
		return result, err
	}
	if err := program.Compile("--gpu-architecture="+result.Architecture, "--fmad=false"); err != nil {
		log, _ := program.Log()
		return result, fmt.Errorf("native compile: %w: %s", err, log)
	}
	cubin, err = program.CUBIN()
	if err != nil {
		return result, err
	}
	result.CUBINSHA256 = digestBytes(cubin)
	raw, err := constructNativeOwner(result.cleanup, "module", func() (driver.Module, error) {
		var loaded driver.Module
		err := dev.WithContext(func() error {
			var loadErr error
			loaded, loadErr = driver.ModuleLoadData(cubin)
			return loadErr
		})
		return loaded, err
	}, func(owner driver.Module) bool {
		return owner != 0
	}, func(owner *driver.Module) error {
		return closeNativeModule(dev.WithContext, owner, (*driver.Module).Close)
	})
	if err != nil {
		return result, err
	}
	if raw == 0 {
		return result, errors.New("native module load returned zero module")
	}
	moduleCleanup := len(result.cleanup.actions) - 1
	module, err := cuda.OwnModule(dev, raw)
	if err != nil {
		return result, err
	}
	result.cleanup.actions[moduleCleanup].close = module.Close
	kernel, err := module.Function("transform")
	if err != nil {
		return result, err
	}
	result.InputSHA256 = digestFloats(nativeInput(cfg.Elements, 0))
	session, captureErr := constructNativeOwner(result.cleanup, "activity session", func() (*cuptiactivity.Session, error) {
		return cuptiactivity.Start(cuptiactivity.Config{Kinds: []cupti.ActivityKind{cupti.ActivityKindDriver, cupti.ActivityKindConcurrentKernel, cupti.ActivityKindMemcpy, cupti.ActivityKindExternalCorrelation}, BufferBytes: 1 << 20, MaxBuffers: 16, MaxRecords: 100000})
	}, func(owner *cuptiactivity.Session) bool {
		return owner != nil
	}, func(owner **cuptiactivity.Session) error {
		return (*owner).Close()
	})
	if session != nil {
		result.cleanup.session = session
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
	executor := &nativeExecutor{
		open: func(_ nativeRequest, own *nativeCleanup) (nativeStream, error) {
			return openCUDANativeStream(dev, kernel, cfg.Elements, own)
		},
		elements:  cfg.Elements,
		hostDelay: cfg.HostDelay,
		evidence:  result.SourceSHA256,
		retain:    cfg.OutputDir != "",
	}
	if captureErr == nil {
		executor.timestamp, executor.timestampAPI = cupti.GetTimestamp, "cupti.GetTimestamp"
	}
	if annotator != nil {
		executor.identity = func(stream nativeStream) (cudaworkload.ContextStreamID, error) {
			owned, ok := stream.(*cudaNativeStream)
			if !ok {
				return cudaworkload.ContextStreamID{}, errors.New("cudadiagdemo: missing CUDA stream identity provider")
			}
			return owned.attributionIdentity()
		}
		executor.annotators = make([]*cudaworkload.Annotator, workers)
		for i := range executor.annotators {
			executor.annotators[i] = annotator
		}
	}
	executor.epoch = time.Now()
	result.ClockEpochUTC = executor.epoch.UTC()
	attempts, err := executor.run(ctx, workers, plan, result.cleanup)
	result.recordAttempts(attempts)
	if err != nil {
		return result, err
	}
	result.NumericalGate = "passed"
	return result, nil
}

// cudaNativeStream is one request attempt's CUDA stream and device buffers.
// Kernel arguments point at its own fields, never at another attempt's.
type cudaNativeStream struct {
	kernel        *cuda.Kernel
	stream        *cuda.Stream
	input, output cudamem.Slice[float32]
	in, out       driver.DevicePtr
	n             int32
}

// openCUDANativeStream allocates one attempt's buffers and non-blocking stream
// on dev's primary context. Closing own synchronizes and destroys the stream
// before freeing the buffers it used.
func openCUDANativeStream(dev *cuda.Device, kernel *cuda.Kernel, elements int, own *nativeCleanup) (nativeStream, error) {
	s := &cudaNativeStream{kernel: kernel, n: int32(elements)}
	alloc := func() (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, elements) }
	owns := func(owner cudamem.Slice[float32]) bool { return owner.Ptr() != 0 }
	free := func(owner *cudamem.Slice[float32]) error { return owner.Free() }
	var err error
	if s.input, err = constructNativeOwner(own, "request input allocation", alloc, owns, free); err != nil {
		return nil, err
	}
	if s.output, err = constructNativeOwner(own, "request output allocation", alloc, owns, free); err != nil {
		return nil, err
	}
	s.stream, err = constructNativeOwner(own, "request stream", func() (*cuda.Stream, error) {
		return dev.NewStreamWithFlags(driver.StreamNonBlocking)
	}, func(owner *cuda.Stream) bool {
		return owner != nil
	}, func(owner **cuda.Stream) error {
		return (*owner).Close()
	})
	if err != nil {
		return nil, err
	}
	s.in, s.out = s.input.Ptr(), s.output.Ptr()
	return s, nil
}

// upload enqueues the copy on the attempt's stream, ordering it before launches.
func (s *cudaNativeStream) upload(input []float32) error {
	return s.input.CopyFromHostAsync(input, s.stream)
}

func (s *cudaNativeStream) launch() error {
	return s.stream.WithHandle(func(handle driver.Stream) error {
		return s.kernel.Launch([3]uint32{uint32((s.n + 127) / 128), 1, 1}, [3]uint32{128, 1, 1}, 0, handle, []unsafe.Pointer{unsafe.Pointer(&s.in), unsafe.Pointer(&s.out), unsafe.Pointer(&s.n)})
	})
}

func (s *cudaNativeStream) synchronize(ctx context.Context) error {
	return s.stream.Synchronize(ctx)
}

// download runs after synchronize; the stream does not order the legacy-stream copy.
func (s *cudaNativeStream) download(output []float32) error {
	return s.output.CopyToHost(output)
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

// exportNative writes transform.cu, result's retained vectors and timing,
// then its receipt and remaining artifacts to dir. A reused dir fails at
// transform.cu before any vector is written. A vector or timing failure is
// recorded in the receipt, which is still written, and returned.
func exportNative(dir string, result *NativeResult, cubin []byte) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	if err := writeNativeFile(createNativeFile, filepath.Join(dir, "transform.cu"), []byte(nativeSource)); err != nil {
		return err
	}
	retainErr := errors.Join(retainNativeVectors(dir, result, createNativeFile), retainNativeTiming(dir, result, createNativeFile))
	var activityRaw bytes.Buffer
	if len(result.Annotations) > 0 && len(result.Activity.Records) > 0 && !result.CleanupPending {
		writer := &nativeActivityWriter{buffer: &activityRaw, remaining: 32 << 20}
		for _, record := range result.Activity.Records {
			if err := json.NewEncoder(writer).Encode(record); err != nil {
				return errors.Join(retainErr, err)
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
		return errors.Join(retainErr, err)
	}
	var artifacts []struct {
		name string
		data []byte
	}
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
		if err := writeNativeFile(createNativeFile, filepath.Join(dir, artifact.name), artifact.data); err != nil {
			return errors.Join(retainErr, err)
		}
	}
	return retainErr
}

type nativeCleanupAction struct {
	name  string
	close func() error
}
type nativeCleanup struct {
	mu      sync.Mutex
	actions []nativeCleanupAction
	session *cuptiactivity.Session
	// adopted reports that a request's failed drain left owners here.
	adopted bool
}

func constructNativeOwner[T any](cleanup *nativeCleanup, name string, construct func() (T, error), owns func(T) bool, close func(*T) error) (T, error) {
	owner, err := construct()
	if owns(owner) {
		cleanup.add(name, func() error { return close(&owner) })
	}
	return owner, err
}

func closeNativeModule(withContext func(func() error) error, module *driver.Module, unload func(*driver.Module) error) error {
	if module == nil || *module == 0 {
		return nil
	}
	return withContext(func() error { return unload(module) })
}

func (c *nativeCleanup) add(name string, close func() error) {
	c.actions = append(c.actions, nativeCleanupAction{name, close})
}

// drain closes actions in reverse order. A failed action stops the drain and
// remains pending with every action before it.
func (c *nativeCleanup) drain() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.drainLocked()
}

func (c *nativeCleanup) drainLocked() error {
	for len(c.actions) > 0 {
		last := len(c.actions) - 1
		action := c.actions[last]
		if err := action.close(); err != nil {
			return fmt.Errorf("native cleanup %s: %w", action.name, err)
		}
		c.actions = c.actions[:last]
	}
	return nil
}

// adopt moves the pending actions of from to the top of c, so that they close
// before the actions c already holds.
func (c *nativeCleanup) adopt(from *nativeCleanup) {
	from.mu.Lock()
	actions := from.actions
	from.actions = nil
	from.mu.Unlock()
	c.mu.Lock()
	c.actions = append(c.actions, actions...)
	c.adopted = true
	c.mu.Unlock()
}

// finishNativeCleanup is RunNative's single automatic cleanup attempt. It
// does not retry a failed activity-start rollback (skip) or the owners of a
// request whose drain failed: those owners and every dependency, through the
// lease, stay pending for an explicit Close.
func finishNativeCleanup(result *NativeResult, primary error, skip bool) error {
	c := result.cleanup
	c.mu.Lock()
	adopted := c.adopted
	c.mu.Unlock()
	if skip || adopted {
		result.CleanupPending = true
		return primary
	}
	return joinNativeCleanup(result, primary)
}

func joinNativeCleanup(result *NativeResult, primary error) error {
	return errors.Join(primary, result.Close())
}

// Close explicitly retries pending native cleanup. RunNative attempts cleanup
// once; a failed action retains every remaining resource and the GPU lease.
// A startup session whose rollback failed is returned without an automatic
// retry, as are the owners of a request whose cleanup failed. Call Close
// after investigating the failure. Successful actions are
// removed and never retried. The zero result has no cleanup ownership.
func (r *NativeResult) Close() error {
	if r == nil || r.cleanup == nil {
		return nil
	}
	c := r.cleanup
	c.mu.Lock()
	defer c.mu.Unlock()
	err := c.drainLocked()
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
