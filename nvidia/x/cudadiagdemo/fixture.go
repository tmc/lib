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
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// Mode identifies the execution mode for the service fixture.
type Mode string

const (
	// ModeBaseline represents ordinary baseline execution with nominal host and device timings.
	ModeBaseline Mode = "baseline"
	// ModeHostSubmissionDelay injects controlled delay before native API entry.
	ModeHostSubmissionDelay Mode = "host_submission_delay"
	// ModeDeviceBottleneck injects device execution contention while host submission remains nominal.
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

// GroundTruthLog records the independent, verified ground truth of an execution.
type GroundTruthLog struct {
	RequestID       string                    `json:"request_id"`
	WorkloadName    string                    `json:"workload_name"`
	OpID            cudaworkload.OpID         `json:"op_id"`
	ProcessGen      cudaworkload.ProcessGenID `json:"process_gen"`
	TID             uint64                    `json:"tid"`
	Mode            Mode                      `json:"mode"`
	HostPrepDelay   time.Duration             `json:"host_prep_delay"`
	HostAPIDuration time.Duration             `json:"host_api_duration"`
	KernelDuration  time.Duration             `json:"kernel_duration"`
	TotalDuration   time.Duration             `json:"total_duration"`
	OutputChecksum  string                    `json:"output_checksum"`
}

// Response records the output and performance characteristics of a completed request.
type Response struct {
	RequestID    string            `json:"request_id"`
	WorkloadName string            `json:"workload_name"`
	OpID         cudaworkload.OpID `json:"op_id"`
	Output       []float64         `json:"output"`
	Duration     time.Duration     `json:"duration"`
	GroundTruth  GroundTruthLog    `json:"ground_truth"`
	Error        error             `json:"error,omitempty"`
}

// ServiceFixture represents a concurrent GPU service supporting telemetry collection,
// request correlation, and controlled bottleneck injection.
type ServiceFixture struct {
	annotator   *cudaworkload.Annotator
	recorder    *cudaflightrecorder.Recorder
	coordinator *cudaflightrecorder.EscalationCoordinator
	joiner      *cudaworkload.Joiner
	process     cudaworkload.ProcessGenID

	weight float64
	bias   float64

	mu   sync.Mutex
	logs []GroundTruthLog

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
func (sf *ServiceFixture) Process(ctx context.Context, req Request) (Response, error) {
	if len(req.Data) == 0 {
		return Response{}, errors.New("cudadiagdemo: empty request data")
	}
	start := time.Now()

	// 1. Begin scoped workload operation
	workloadName := req.WorkloadName
	if workloadName == "" {
		workloadName = "inference"
	}
	scopedCtx, scope := sf.annotator.Begin(ctx, workloadName)
	defer scope.End()

	tid := req.TID
	if tid == 0 {
		tid = sf.nextTID.Add(1)
	}

	corrID := req.CorrelationID
	if corrID == 0 {
		corrID = sf.nextCorr.Add(1)
	}

	var hostPrepDelay time.Duration
	var hostAPIDuration time.Duration
	var kernelDuration time.Duration

	// 2. Controlled host-submission-delay: injected before API entry
	if req.Mode == ModeHostSubmissionDelay {
		delay := req.Delay
		if delay <= 0 {
			delay = 20 * time.Millisecond
		}
		hostPrepDelay = delay
		// In portable test harness, sleep or simulate delay
		select {
		case <-scopedCtx.Done():
			return Response{RequestID: req.ID, Error: scopedCtx.Err()}, scopedCtx.Err()
		case <-time.After(delay):
		}
	} else {
		hostPrepDelay = 10 * time.Microsecond
	}

	// 3. Native submission call with thread pinning via runtime.LockOSThread in Scope.NativeCall
	apiStart := time.Now()
	var sub cudaworkload.HostSubmission
	err := scope.NativeCall(func() error {
		// Simulate native driver launch API call duration
		time.Sleep(15 * time.Microsecond)
		hostAPIDuration = time.Since(apiStart)

		sub = cudaworkload.HostSubmission{
			Process:             sf.process,
			OpID:                scope.ID(),
			Workload:            scope.Workload(),
			NativeCorrelationID: corrID,
			Provenance:          cudaworkload.ProvenanceCUPTI,
			TID:                 tid,
			TimestampNs:         uint64(time.Now().UnixNano()),
			Symbol:              "gemm_kernel",
		}
		return nil
	})
	if err != nil {
		return Response{RequestID: req.ID, Error: err}, err
	}

	// Submit host submission to joiner and push to continuous recorder
	sf.joiner.SubmitHost(sub)
	sf.recorder.Push(cudaflightrecorder.Record{
		TimestampNs:   int64(sub.TimestampNs),
		KernelName:    sub.Symbol,
		CorrelationID: uint64(sub.NativeCorrelationID),
		DurationNs:    uint64(hostAPIDuration.Nanoseconds()),
		Labels: map[string]string{
			"op_id": fmt.Sprintf("%d", scope.ID()),
			"tid":   fmt.Sprintf("%d", tid),
		},
	})

	// 4. Device activity simulation
	if req.Mode == ModeDeviceBottleneck {
		delay := req.Delay
		if delay <= 0 {
			delay = 250 * time.Microsecond
		}
		kernelDuration = delay
	} else {
		kernelDuration = 50 * time.Microsecond
	}

	actStartNs := uint64(time.Now().UnixNano())
	actEndNs := actStartNs + uint64(kernelDuration.Nanoseconds())
	act := cudaworkload.DeviceActivity{
		Process:               sf.process,
		NativeCorrelationID:   corrID,
		ExternalCorrelationID: uint64(scope.ID()),
		Kind:                  "kernel",
		Name:                  "gemm_kernel",
		StartNs:               actStartNs,
		EndNs:                 actEndNs,
		DeviceID:              0,
	}

	sf.joiner.SubmitDevice(act)
	sf.recorder.Push(cudaflightrecorder.Record{
		TimestampNs:   int64(act.StartNs),
		KernelName:    act.Name,
		CorrelationID: uint64(act.NativeCorrelationID),
		DurationNs:    uint64(kernelDuration.Nanoseconds()),
		Labels: map[string]string{
			"activity.kind": "kernel",
			"op_id":         fmt.Sprintf("%d", scope.ID()),
		},
	})

	// 5. Numerical computation: $y_i = w \cdot x_i + b$
	output := ComputeReferenceOutput(req.Data, sf.weight, sf.bias)

	// Checksum of output
	h := sha256.New()
	for _, v := range output {
		bits := math.Float64bits(v)
		h.Write([]byte{
			byte(bits), byte(bits >> 8), byte(bits >> 16), byte(bits >> 24),
			byte(bits >> 32), byte(bits >> 40), byte(bits >> 48), byte(bits >> 56),
		})
	}
	checksum := hex.EncodeToString(h.Sum(nil))

	totalDuration := time.Since(start)

	logEntry := GroundTruthLog{
		RequestID:       req.ID,
		WorkloadName:    workloadName,
		OpID:            scope.ID(),
		ProcessGen:      sf.process,
		TID:             tid,
		Mode:            req.Mode,
		HostPrepDelay:   hostPrepDelay,
		HostAPIDuration: hostAPIDuration,
		KernelDuration:  kernelDuration,
		TotalDuration:   totalDuration,
		OutputChecksum:  checksum,
	}

	sf.mu.Lock()
	sf.logs = append(sf.logs, logEntry)
	sf.mu.Unlock()

	return Response{
		RequestID:    req.ID,
		WorkloadName: workloadName,
		OpID:         scope.ID(),
		Output:       output,
		Duration:     totalDuration,
		GroundTruth:  logEntry,
	}, nil
}

// BatchProcess executes requests concurrently across a worker pool.
func (sf *ServiceFixture) BatchProcess(ctx context.Context, reqs []Request, concurrency int) ([]Response, error) {
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
