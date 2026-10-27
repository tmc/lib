package cudadiagdemo

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// Request worker bounds. NativeConfig.Concurrency zero selects
// nativeDefaultConcurrency workers.
const (
	nativeDefaultConcurrency = 4
	nativeMaxConcurrency     = 8
	// nativeMaxLiveBytes bounds the input and output device buffers held
	// by all concurrently executing requests.
	nativeMaxLiveBytes = 32 << 20
)

var nativeArms = [...]string{"baseline", "candidate", "device"}

// errNativeRequestFailed is the cause recorded for requests left
// undispatched or canceled because another request failed.
var errNativeRequestFailed = errors.New("cudadiagdemo: another native request failed")

// nativeWorkers returns the request worker count for cfg, whose other
// fields must already be valid. It never exceeds the planned request count.
func nativeWorkers(cfg NativeConfig) (int, error) {
	workers := cfg.Concurrency
	if workers == 0 {
		workers = nativeDefaultConcurrency
	}
	if workers < 1 || workers > nativeMaxConcurrency {
		return 0, errors.New("cudadiagdemo: invalid native concurrency")
	}
	workers = min(workers, len(nativeArms)*cfg.Rounds)
	if int64(workers)*2*4*int64(cfg.Elements) > nativeMaxLiveBytes {
		return 0, errors.New("cudadiagdemo: native concurrency exceeds live buffer bound")
	}
	return workers, nil
}

// nativeRequest is one planned request attempt. Index is its position in
// the plan, which fixes result order; it is not an execution order.
type nativeRequest struct {
	index    int
	round    int
	arm      string
	launches int
}

// planNativeRequests returns every request attempt in result order: rounds
// in order, with baseline first in odd rounds and candidate first in even
// rounds. Device attempts launch deviceIterations kernels; others launch one.
func planNativeRequests(rounds, deviceIterations int) []nativeRequest {
	plan := make([]nativeRequest, 0, len(nativeArms)*rounds)
	for round := 1; round <= rounds; round++ {
		arms := nativeArms
		if round%2 == 0 {
			arms[0], arms[1] = arms[1], arms[0]
		}
		for _, arm := range arms {
			launches := 1
			if arm == "device" {
				launches = deviceIterations
			}
			plan = append(plan, nativeRequest{index: len(plan), round: round, arm: arm, launches: launches})
		}
	}
	return plan
}

// nativeInput returns the input of the request at plan index i. Inputs
// differ between requests so that output read from another request's
// buffers fails the numerical oracle.
func nativeInput(elements, i int) []float32 {
	input := make([]float32, elements)
	for j := range input {
		input[j] = float32((j + i) % 1024)
	}
	return input
}

// nativeStream is the device state owned by one request attempt: its own
// stream and input and output buffers. One goroutine calls its methods.
type nativeStream interface {
	upload(input []float32) error
	launch() error
	synchronize(ctx context.Context) error
	download(output []float32) error
}

// nativeExecutor runs request attempts. Open constructs one attempt's
// stream and registers every owner it constructs on own, including when it
// returns an error. Annotators is indexed by worker and is nil when capture
// is unavailable, as is timestamp. Retain keeps each attempt's encoded input,
// CPU reference and completely downloaded output. Fields are read-only while
// requests run.
type nativeExecutor struct {
	open       func(own *nativeCleanup) (nativeStream, error)
	annotators []*cudaworkload.Annotator
	timestamp  func() (uint64, error)
	elements   int
	hostDelay  time.Duration
	evidence   string
	epoch      time.Time
	retain     bool
}

// nativeAttempt is the retained record of one dispatched request attempt.
type nativeAttempt struct {
	trial      NativeTrial
	log        NativeRequestLog
	annotation *cudaworkload.HostSubmission
	incorrect  bool
	vectors    *nativeVectors
}

// run executes plan on at most workers goroutines. Each worker runs one
// attempt at a time on its own locked OS thread and releases the attempt's
// owners after it returns. A failed release moves the remaining owners to
// retain for an explicit Close, which never resubmits work.
//
// The first failure cancels in-flight attempts and stops dispatch. Every
// dispatched attempt is returned in plan order, including failed and
// canceled ones; the error joins each failure with its request identity.
// Run returns only after every worker has quiesced.
func (e *nativeExecutor) run(ctx context.Context, workers int, plan []nativeRequest, retain *nativeCleanup) ([]nativeAttempt, error) {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	attempts := make([]nativeAttempt, len(plan))
	errs := make([]error, len(plan))
	dispatched := make([]bool, len(plan))
	var (
		mu   sync.Mutex
		next int
	)
	// take refuses dispatch after any failure has canceled ctx.
	take := func() (int, bool) {
		mu.Lock()
		defer mu.Unlock()
		if next == len(plan) || ctx.Err() != nil {
			return 0, false
		}
		next++
		dispatched[next-1] = true
		return next - 1, true
	}
	var wg sync.WaitGroup
	for worker := range workers {
		wg.Go(func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			for {
				i, ok := take()
				if !ok {
					return
				}
				own := &nativeCleanup{}
				attempts[i], errs[i] = e.attempt(ctx, worker, plan[i], own)
				if errs[i] != nil {
					cancel(errNativeRequestFailed)
				}
				if err := own.drain(); err != nil {
					retain.adopt(own)
					errs[i] = errors.Join(errs[i], fmt.Errorf("cleanup: %w", err))
					cancel(errNativeRequestFailed)
				}
				if errs[i] != nil {
					attempts[i].trial.Error = errs[i].Error()
				}
			}
		})
	}
	wg.Wait()

	var out []nativeAttempt
	var err error
	for i, a := range attempts {
		if !dispatched[i] {
			continue
		}
		out = append(out, a)
		if errs[i] != nil {
			err = errors.Join(err, fmt.Errorf("native request %s/%s: %w", a.trial.RequestID, a.trial.Attempt, errs[i]))
		}
	}
	if len(out) < len(plan) {
		err = errors.Join(err, fmt.Errorf("native requests: %d of %d not dispatched: %w", len(plan)-len(out), len(plan), context.Cause(ctx)))
	}
	return out, err
}

// attempt executes req with stream and buffers owned by own. The returned
// record is complete even when err is non-nil.
func (e *nativeExecutor) attempt(ctx context.Context, worker int, req nativeRequest, own *nativeCleanup) (a nativeAttempt, err error) {
	trial := &a.trial
	*trial = NativeTrial{RequestedLaunches: req.launches, RequestID: fmt.Sprintf("request-%d", req.round), Attempt: req.arm, Arm: req.arm, Round: req.round, Worker: worker}
	mode := ModeBaseline
	if req.arm == "baseline" {
		mode = ModeHostSubmissionDelay
	} else if req.arm == "device" {
		mode = ModeDeviceBottleneck
	}
	a.log = NativeRequestLog{Attempt: req.arm, StartedNS: time.Since(e.epoch).Nanoseconds(), GroundTruth: GroundTruthLog{RequestID: trial.RequestID, WorkloadName: "gpudiagdemo-native", Mode: mode}}
	defer func() {
		a.log.FinishedNS = time.Since(e.epoch).Nanoseconds()
		a.log.Complete = trial.Correct
		a.log.GroundTruth.TotalDuration = time.Duration(a.log.FinishedNS - a.log.StartedNS)
		trial.WallNS = a.log.GroundTruth.TotalDuration.Nanoseconds()
	}()
	if err := ctx.Err(); err != nil {
		return a, err
	}
	host := nativeInput(e.elements, req.index)
	got := make([]float32, e.elements)
	trial.InputSHA256 = digestFloats(host)
	if e.retain {
		a.vectors = &nativeVectors{index: req.index, input: encodeFloats(host), reference: encodeFloats(nativeReference(host))}
	}
	stream, err := e.open(own)
	if err != nil {
		return a, err
	}
	if req.arm == "baseline" {
		delayStart := time.Now()
		timer := time.NewTimer(e.hostDelay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return a, ctx.Err()
		case <-timer.C:
		}
		trial.DelayNS = time.Since(delayStart).Nanoseconds()
	}
	call := func(name string, fn func() error) error {
		observation := NativeCall{Name: name, StartNS: time.Since(e.epoch).Nanoseconds()}
		err := fn()
		observation.EndNS = time.Since(e.epoch).Nanoseconds()
		if err != nil {
			observation.Error = err.Error()
		}
		trial.Calls = append(trial.Calls, observation)
		return err
	}
	if e.timestamp != nil {
		if trial.CUPTIBefore, err = e.timestamp(); err != nil {
			return a, err
		}
	}
	if err := call("cuMemcpyHtoDAsync_v2", func() error { return stream.upload(host) }); err != nil {
		return a, err
	}
	launchRequest := func() error {
		for range req.launches {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := call("cuLaunchKernel", stream.launch); err != nil {
				return err
			}
			trial.KernelLaunches++
		}
		return nil
	}
	var annotator *cudaworkload.Annotator
	if e.annotators != nil {
		annotator = e.annotators[worker]
	}
	if annotator == nil {
		trial.Correlation = "uninstrumented"
		if err := launchRequest(); err != nil {
			return a, err
		}
	} else {
		_, scope := annotator.Begin(ctx, trial.RequestID+"/"+trial.Attempt)
		var entry *cudaworkload.HostSubmission
		sub, err := scope.NativeOperation(ctx, cudaworkload.ContextStreamID{}, "cuLaunchKernel", e.evidence, func() error {
			// NativeOperation calls this only after correlation entry, on
			// the pinned thread that makes the launches.
			observed := scope.NewSubmission(0, cudaworkload.ContextStreamID{}, cudaworkload.GraphReplayGeneration{}, "cuLaunchKernel", e.evidence)
			entry = &observed
			return launchRequest()
		})
		scope.End()
		// The application identity is known whether or not the operation
		// succeeded; the thread is known only once the launches began.
		workload := scope.Workload()
		a.log.GroundTruth.OpID = scope.ID()
		a.log.GroundTruth.ProcessGen = workload.Process
		a.log.GroundTruth.Workload = workload
		switch {
		case err == nil:
			trial.Correlation = "complete"
			a.annotation = &sub
			a.log.GroundTruth.Workload = sub.Workload
			a.log.GroundTruth.TID = sub.TID
		case entry != nil:
			trial.Correlation = "partial"
			a.annotation = entry
			a.log.GroundTruth.TID = entry.TID
		default:
			trial.Correlation = "not_entered"
		}
		if err != nil {
			return a, err
		}
	}
	if err := call("cuStreamSynchronize", func() error { return stream.synchronize(ctx) }); err != nil {
		return a, err
	}
	if err := call("cuMemcpyDtoH_v2", func() error { return stream.download(got) }); err != nil {
		// A failed download may have partly written got; it is not output.
		return a, err
	}
	if a.vectors != nil {
		a.vectors.output = encodeFloats(got)
	}
	if e.timestamp != nil {
		if trial.CUPTIAfter, err = e.timestamp(); err != nil {
			return a, err
		}
	}
	trial.OutputSHA256 = digestFloats(got)
	if err := checkNativeOutput(host, got); err != nil {
		a.incorrect = true
		return a, err
	}
	trial.Correct = true
	a.log.GroundTruth.OutputChecksum = trial.OutputSHA256
	a.log.GroundTruth.HostPrepDelay = time.Duration(trial.DelayNS)
	return a, nil
}
