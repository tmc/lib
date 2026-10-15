package cudadiagdemo

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// Mode identifies the execution mode for the service fixture.
type Mode string

const (
	// ModeBaseline represents ordinary baseline execution with nominal host and device timings.
	ModeBaseline Mode = "baseline"
	// ModeHostSubmissionDelay injects controlled delay before native API entry.
	ModeHostSubmissionDelay Mode = "host_submission_delay"
	// ModeDeviceBottleneck models a long synthetic device interval while keeping
	// host API work unchanged. It does not execute physical competing work.
	ModeDeviceBottleneck Mode = "device_bottleneck"
)

// Request defines a work item submitted to the GPU service fixture.
type Request struct {
	ID            string        `json:"id"`
	WorkloadName  string        `json:"workload_name"`
	Data          []float64     `json:"data"`
	Mode          Mode          `json:"mode"`
	Delay         time.Duration `json:"delay,omitempty"`
	CorrelationID uint32        `json:"correlation_id,omitempty"`
	TID           uint64        `json:"tid,omitempty"`
}

// GroundTruthLog records request identity and controls separately from diagnostic
// input. Its timing copies are for audit, not the source of a finding.
type GroundTruthLog struct {
	RequestID       string                        `json:"request_id"`
	WorkloadName    string                        `json:"workload_name"`
	OpID            cudaworkload.OpID             `json:"op_id"`
	ProcessGen      cudaworkload.ProcessGenID     `json:"process_gen"`
	Workload        cudaworkload.WorkloadIdentity `json:"workload"`
	TID             uint64                        `json:"tid"`
	Mode            Mode                          `json:"mode"`
	HostPrepDelay   time.Duration                 `json:"host_prep_delay"`
	HostAPIDuration time.Duration                 `json:"host_api_duration"`
	KernelDuration  time.Duration                 `json:"kernel_duration"`
	TotalDuration   time.Duration                 `json:"total_duration"`
	OutputChecksum  string                        `json:"output_checksum"`
}

// Response records the output and performance characteristics of a completed request.
type Response struct {
	RequestID    string            `json:"request_id"`
	WorkloadName string            `json:"workload_name"`
	OpID         cudaworkload.OpID `json:"op_id"`
	Output       []float64         `json:"output"`
	Duration     time.Duration     `json:"duration"`
	GroundTruth  GroundTruthLog    `json:"ground_truth"`
	Observation  SubmissionTiming  `json:"observation"`
	Error        error             `json:"error,omitempty"`
}

// ServiceFixture is a portable concurrent service with synthetic device activity,
// actual application timing and separate request ground truth.
type ServiceFixture struct {
	annotator   *cudaworkload.Annotator
	recorder    *cudaflightrecorder.Recorder
	coordinator *cudaflightrecorder.EscalationCoordinator
	joiner      *cudaworkload.Joiner
	process     cudaworkload.ProcessGenID

	weight float64
	bias   float64

	mu            sync.Mutex
	logs          []GroundTruthLog
	timings       []SubmissionTiming
	timingDropped uint64
	timingLimit   int
	epoch         time.Time
	source        TimingSource

	nextTID  atomic.Uint64
	nextCorr atomic.Uint32
}

// FixtureOption configures a ServiceFixture.
type FixtureOption func(*ServiceFixture)

// WithWeightBias sets custom weights and biases for the numerical transform.
func WithWeightBias(w, b float64) FixtureOption {
	return func(sf *ServiceFixture) {
		sf.weight = w
		sf.bias = b
	}
}

// WithProcess sets the process generation ID.
func WithProcess(proc cudaworkload.ProcessGenID) FixtureOption {
	return func(sf *ServiceFixture) {
		sf.process = proc
	}
}

// NewServiceFixture instantiates a service fixture.
func NewServiceFixture(
	annotator *cudaworkload.Annotator,
	recorder *cudaflightrecorder.Recorder,
	coordinator *cudaflightrecorder.EscalationCoordinator,
	joiner *cudaworkload.Joiner,
	opts ...FixtureOption,
) (*ServiceFixture, error) {
	if annotator == nil {
		return nil, errors.New("cudadiagdemo: nil annotator")
	}
	if recorder == nil {
		return nil, errors.New("cudadiagdemo: nil recorder")
	}
	if joiner == nil {
		return nil, errors.New("cudadiagdemo: nil joiner")
	}

	pid := os.Getpid()
	if pid <= 0 {
		pid = 12345
	}
	proc := cudaworkload.ProcessGenID{
		PID:        uint32(pid),
		Generation: 1,
	}

	sf := &ServiceFixture{
		annotator:   annotator,
		recorder:    recorder,
		coordinator: coordinator,
		joiner:      joiner,
		process:     proc,
		weight:      2.0,
		bias:        1.0,
	}
	sf.nextTID.Store(1000)
	sf.nextCorr.Store(1)
	sf.epoch = time.Now()
	sf.timingLimit = maxTimingRecords
	sf.source = portableTimingSource(fmt.Sprintf("process-%d-%d", pid, sf.epoch.UnixNano()))

	for _, opt := range opts {
		opt(sf)
	}

	return sf, nil
}

// ComputeReferenceOutput calculates the expected numerical transform $y_i = w \cdot x_i + b$.
func ComputeReferenceOutput(input []float64, weight, bias float64) []float64 {
	out := make([]float64, len(input))
	for i, x := range input {
		out[i] = weight*x + bias
	}
	return out
}

// CheckNumericalCorrectness compares candidate outputs against reference outputs within a tolerance.
func CheckNumericalCorrectness(reference, candidate []float64, tolerance float64) error {
	if len(reference) == 0 || len(candidate) == 0 {
		return errors.New("cudadiagdemo: empty output vector for oracle evaluation")
	}
	if len(reference) != len(candidate) {
		return fmt.Errorf("cudadiagdemo: output length mismatch: reference %d, candidate %d",
			len(reference), len(candidate))
	}
	for i, refVal := range reference {
		candVal := candidate[i]
		if math.IsNaN(refVal) || math.IsNaN(candVal) {
			return fmt.Errorf("cudadiagdemo: NaN encountered at index %d", i)
		}
		if math.IsInf(refVal, 0) || math.IsInf(candVal, 0) {
			return fmt.Errorf("cudadiagdemo: Inf encountered at index %d", i)
		}
		diff := math.Abs(refVal - candVal)
		if diff > tolerance {
			return fmt.Errorf("cudadiagdemo: numerical tolerance exceeded at index %d: diff %g > tolerance %g",
				i, diff, tolerance)
		}
	}
	return nil
}

// Process executes a single request through the service pipeline.
func (sf *ServiceFixture) Process(ctx context.Context, req Request) (resp Response, retErr error) {
	if ctx == nil || req.ID == "" || len(req.ID) > 256 || len(req.WorkloadName) > 256 || len(req.Data) == 0 || len(req.Data) > 1<<20 || req.Delay < 0 || req.Delay > time.Second || (req.Mode != "" && req.Mode != ModeBaseline && req.Mode != ModeHostSubmissionDelay && req.Mode != ModeDeviceBottleneck) {
		return Response{}, errors.New("cudadiagdemo: invalid request bounds")
	}
	start := time.Now()
	workloadName := req.WorkloadName
	if workloadName == "" {
		workloadName = "inference"
	}
	scopedCtx, scope := sf.annotator.Begin(ctx, workloadName)
	defer scope.End()
	corrID := req.CorrelationID
	if corrID == 0 {
		corrID = sf.nextCorr.Add(1)
	}
	timing := SubmissionTiming{RequestID: req.ID, Service: tracecontract.Interval{Start: start.Sub(sf.epoch).Nanoseconds(), Clock: portableClock}}
	defer func() {
		timing.Service.End = time.Since(sf.epoch).Nanoseconds()
		resp.RequestID, resp.WorkloadName, resp.OpID = req.ID, workloadName, scope.ID()
		resp.Duration = time.Since(start)
		resp.Error = retErr
		resp.Observation = timing
		sf.retainTiming(timing)
	}()
	ready := time.Since(sf.epoch).Nanoseconds()
	if req.Mode == ModeHostSubmissionDelay {
		delay := req.Delay
		if delay == 0 {
			delay = 20 * time.Millisecond
		}
		timer := time.NewTimer(delay)
		select {
		case <-scopedCtx.Done():
			timer.Stop()
			return resp, scopedCtx.Err()
		case <-timer.C:
		}
	}
	if err := scopedCtx.Err(); err != nil {
		return resp, err
	}
	// NativeCall pins only this callback. Its TID is not attributed to ready's
	// earlier unpinned interval. The portable launch and device are synthetic.
	if err := scope.NativeCall(func() error {
		entry := time.Since(sf.epoch).Nanoseconds()
		timing.Host = scope.NewSubmission(corrID, cudaworkload.ContextStreamID{Context: 1, Stream: 1}, cudaworkload.GraphReplayGeneration{}, "portable_launch", "")
		timing.Host.Provenance = cudaworkload.ProvenanceSynthetic
		timing.Host.TimestampNs = uint64(entry)
		timing.Host.TimestampClock, timing.Host.TimestampEpoch = portableClock, sf.source.ClockEpoch
		if req.TID != 0 && req.TID != timing.Host.TID {
			return errors.New("cudadiagdemo: requested TID contradicts observed API thread")
		}
		timing.PreEntry = tracecontract.Interval{Start: ready, End: entry, Clock: portableClock}
		apiStart := time.Since(sf.epoch).Nanoseconds()
		// Keep the annotation outside the simulated API entry/return bracket.
		timing.Host.TimestampNs = uint64(apiStart)
		timing.PreEntry.End = apiStart
		time.Sleep(15 * time.Microsecond)
		timing.API = cudadiag.HostCallEvent{Name: timing.Host.Symbol, HasReturn: true, Span: tracecontract.Interval{Start: apiStart, End: time.Since(sf.epoch).Nanoseconds(), Clock: portableClock}}
		return nil
	}); err != nil {
		return resp, err
	}
	kernelDuration := 50 * time.Microsecond
	if req.Mode == ModeDeviceBottleneck {
		kernelDuration = req.Delay
		if kernelDuration == 0 {
			kernelDuration = 250 * time.Microsecond
		}
	}
	deviceStart := time.Since(sf.epoch).Nanoseconds()
	// This wait bounds a portable simulated activity inside service time. It is
	// host work, not a kernel timer or evidence that a GPU executed.
	timer := time.NewTimer(kernelDuration)
	select {
	case <-scopedCtx.Done():
		timer.Stop()
		return resp, scopedCtx.Err()
	case <-timer.C:
	}
	timing.Device = cudaworkload.DeviceActivity{Process: sf.process, NativeCorrelationID: corrID, ExternalCorrelationID: uint64(scope.ID()), Kind: "kernel", Name: "portable_transform", Context: timing.Host.Context, StartNs: uint64(deviceStart), EndNs: uint64(deviceStart) + uint64(kernelDuration), DeviceID: 0}
	sf.joiner.SubmitHost(timing.Host)
	sf.joiner.SubmitDevice(timing.Device)
	// The actual portable transform is separate from the reference helper.
	output := make([]float64, len(req.Data))
	for i, value := range req.Data {
		output[i] = value*sf.weight + sf.bias
	}
	h := sha256.New()
	for _, value := range output {
		bits := math.Float64bits(value)
		for shift := uint(0); shift < 64; shift += 8 {
			h.Write([]byte{byte(bits >> shift)})
		}
	}
	timing.Complete = true
	sf.recorder.Push(cudaflightrecorder.Record{TimestampNs: time.Now().UnixNano(), KernelName: timing.Device.Name, CorrelationID: uint64(corrID), DurationNs: uint64(kernelDuration), Labels: map[string]string{"activity.kind": "synthetic_kernel", "source_class": "synthetic", "op_id": scope.ID().String()}})
	logEntry := GroundTruthLog{RequestID: req.ID, WorkloadName: workloadName, OpID: scope.ID(), ProcessGen: sf.process, Workload: scope.Workload(), TID: timing.Host.TID, Mode: req.Mode, HostPrepDelay: time.Duration(timing.PreEntry.Duration()), HostAPIDuration: time.Duration(timing.API.Span.Duration()), KernelDuration: kernelDuration, TotalDuration: time.Since(start), OutputChecksum: hex.EncodeToString(h.Sum(nil))}
	sf.mu.Lock()
	if len(sf.logs) < maxTimingRecords {
		saved := logEntry
		saved.Workload = logEntry.Workload.Clone()
		sf.logs = append(sf.logs, saved)
	}
	sf.mu.Unlock()
	resp.Output, resp.GroundTruth = output, logEntry
	return resp, nil
}

// BatchProcess executes requests concurrently across a worker pool.
func (sf *ServiceFixture) BatchProcess(ctx context.Context, reqs []Request, concurrency int) ([]Response, error) {
	if ctx == nil || len(reqs) > maxTimingRecords || concurrency > 64 {
		return nil, errors.New("cudadiagdemo: batch bounds exceeded")
	}
	if concurrency <= 0 {
		concurrency = 4
	}
	resps := make([]Response, len(reqs))
	jobs := make(chan int, len(reqs))
	for i := range reqs {
		jobs <- i
	}
	close(jobs)

	var wg sync.WaitGroup
	var firstErr error
	var errMu sync.Mutex

	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				if ctx.Err() != nil {
					errMu.Lock()
					if firstErr == nil {
						firstErr = ctx.Err()
					}
					errMu.Unlock()
					return
				}
				resp, err := sf.Process(ctx, reqs[idx])
				if err != nil {
					errMu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					errMu.Unlock()
				}
				resps[idx] = resp
			}
		}()
	}

	wg.Wait()
	return resps, firstErr
}

// GroundTruthLogs returns a copy of all accumulated execution logs.
func (sf *ServiceFixture) GroundTruthLogs() []GroundTruthLog {
	sf.mu.Lock()
	defer sf.mu.Unlock()
	cp := make([]GroundTruthLog, len(sf.logs))
	copy(cp, sf.logs)
	for i := range cp {
		cp[i].Workload = cp[i].Workload.Clone()
	}
	return cp
}

// ProcessGen returns the fixture's process generation identity.
func (sf *ServiceFixture) ProcessGen() cudaworkload.ProcessGenID {
	return sf.process
}

func curGoroutineID() uint64 {
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	s := strings.TrimPrefix(string(buf[:n]), "goroutine ")
	idx := strings.IndexByte(s, ' ')
	if idx > 0 {
		id, _ := strconv.ParseUint(s[:idx], 10, 64)
		return id
	}
	return 0
}

// CorrelationHookMock provides an in-memory correlation hook for testing external correlation.
type CorrelationHookMock struct {
	mu     sync.Mutex
	stacks map[uint64][]uint64
}

// NewCorrelationHookMock creates an initialized CorrelationHookMock.
func NewCorrelationHookMock() *CorrelationHookMock {
	return &CorrelationHookMock{
		stacks: make(map[uint64][]uint64),
	}
}

// Push appends an ID to the caller's goroutine correlation stack.
func (c *CorrelationHookMock) Push(kind cupti.ExternalCorrelationKind, id uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	gid := curGoroutineID()
	if c.stacks == nil {
		c.stacks = make(map[uint64][]uint64)
	}
	c.stacks[gid] = append(c.stacks[gid], id)
	return nil
}

// Pop removes and returns the top ID from the caller's goroutine correlation stack.
func (c *CorrelationHookMock) Pop(kind cupti.ExternalCorrelationKind) (uint64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	gid := curGoroutineID()
	s := c.stacks[gid]
	if len(s) == 0 {
		return 0, errors.New("empty correlation stack")
	}
	top := s[len(s)-1]
	c.stacks[gid] = s[:len(s)-1]
	if len(c.stacks[gid]) == 0 {
		delete(c.stacks, gid)
	}
	return top, nil
}
