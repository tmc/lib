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

// nativeInput returns the input with the given seed. Under nativeRecipe
// every attempt of round r uses seed r-1, so a round's arms upload the same
// values to their own buffers. Rounds differ, so output read from another
// round's buffers fails the numerical oracle.
func nativeInput(elements, seed int) []float32 {
	input := make([]float32, elements)
	for j := range input {
		input[j] = float32((j + seed) % 1024)
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

// nativeExecutor runs request attempts. Open constructs the stream of req
// and registers every owner it constructs on own, including when it returns
// an error; RunNative's open ignores req, which identifies the attempt to
// in-package fakes. Annotators is indexed by worker and is nil when capture
// is unavailable, as is timestamp, the function timestampAPI names. Retain
// keeps each attempt's encoded input, CPU reference and completely
// downloaded output. Now and wait, when set, replace the monotonic clock
// since epoch and the baseline's timer. Fields are read-only while requests
// run. Identity, when set, queries the owned stream before launch-ready; nil
// retains unknown legacy identity rather than inventing activity IDs.
type nativeExecutor struct {
	open         func(req nativeRequest, own *nativeCleanup) (nativeStream, error)
	annotators   []*cudaworkload.Annotator
	identity     func(nativeStream) (cudaworkload.ContextStreamID, error)
	timestamp    func() (uint64, error)
	timestampAPI string
	elements     int
	hostDelay    time.Duration
	evidence     string
	epoch        time.Time
	retain       bool
	now          func() int64
	wait         func(ctx context.Context, d time.Duration) error
}

// since returns the nanoseconds since e.epoch on the monotonic clock.
func (e *nativeExecutor) since() int64 {
	if e.now != nil {
		return e.now()
	}
	return time.Since(e.epoch).Nanoseconds()
}

// hostWait is the baseline's intentional host delay before its first launch.
func (e *nativeExecutor) hostWait(ctx context.Context) error {
	if e.wait != nil {
		return e.wait(ctx, e.hostDelay)
	}
	timer := time.NewTimer(e.hostDelay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
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

// attempt executes req with stream and buffers owned by own, in the order of
// nativeRecipe: it opens the stream, enqueues the upload, begins the
// application operation, observes launch-ready, waits if req is a baseline,
// and then makes the launches, synchronizes and downloads. The returned
// record is complete even when err is non-nil; it holds every wrapper that
// was called, including a failed one, and none that was not.
func (e *nativeExecutor) attempt(ctx context.Context, worker int, req nativeRequest, own *nativeCleanup) (a nativeAttempt, err error) {
	trial := &a.trial
	*trial = NativeTrial{RequestedLaunches: req.launches, RequestID: fmt.Sprintf("request-%d", req.round), Attempt: req.arm, Arm: req.arm, Round: req.round, Worker: worker}
	timing := &NativeTiming{Recipe: nativeRecipe, TimestampAPI: "unavailable"}
	if e.timestamp != nil {
		timing.TimestampAPI, timing.TimestampMethod = e.timestampAPI, "unknown"
	}
	trial.Timing = timing
	trial.OperationName = "cuLaunchKernel"
	mode := ModeBaseline
	if req.arm == "baseline" {
		mode = ModeHostSubmissionDelay
	} else if req.arm == "device" {
		mode = ModeDeviceBottleneck
	}
	a.log = NativeRequestLog{Attempt: req.arm, StartedNS: e.since(), GroundTruth: GroundTruthLog{RequestID: trial.RequestID, WorkloadName: "gpudiagdemo-native", Mode: mode}}
	defer func() {
		a.log.FinishedNS = e.since()
		a.log.Complete = trial.Correct
		a.log.GroundTruth.TotalDuration = time.Duration(a.log.FinishedNS - a.log.StartedNS)
		trial.WallNS = a.log.GroundTruth.TotalDuration.Nanoseconds()
	}()
	if err := ctx.Err(); err != nil {
		return a, err
	}
	host := nativeInput(e.elements, req.round-1)
	got := make([]float32, e.elements)
	trial.InputSHA256 = digestFloats(host)
	if e.retain {
		a.vectors = &nativeVectors{index: req.index, input: encodeFloats(host), reference: encodeFloats(nativeReference(host))}
	}
	stream, err := e.open(req, own)
	if err != nil {
		return a, err
	}
	// call brackets fn with monotonic times immediately around it and, with
	// a timestamp source, raw samples just outside those. A sample failure
	// is recorded in the sample, never as fn's error.
	call := func(name string, fn func() error) error {
		observation := NativeCall{Name: name}
		if e.timestamp != nil {
			observation.Before = e.sample()
		}
		observation.StartNS = e.since()
		err := fn()
		observation.EndNS = e.since()
		if e.timestamp != nil {
			observation.After = e.sample()
		}
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
	var annotator *cudaworkload.Annotator
	if e.annotators != nil {
		annotator = e.annotators[worker]
	}
	var contextStream cudaworkload.ContextStreamID
	if annotator != nil && e.identity != nil {
		if err := ctx.Err(); err != nil {
			return a, err
		}
		contextStream, err = e.identity(stream)
		if err != nil {
			return a, fmt.Errorf("cudadiagdemo: query submission identity: %w", err)
		}
		trial.ContextStream = &contextStream
	}
	var scope *cudaworkload.Scope
	if annotator != nil {
		// The application identity is known from here whether or not the
		// operation is entered; its entry thread is known only once the
		// launches begin.
		_, scope = annotator.Begin(ctx, trial.OperationName)
		workload := scope.Workload()
		a.log.GroundTruth.OpID = scope.ID()
		a.log.GroundTruth.ProcessGen = workload.Process
		a.log.GroundTruth.Workload = workload
		identity := scope.Workload()
		timing.OpID, timing.Workload = scope.ID(), &identity
	}
	e.observeReady(timing, scope)
	if req.arm == "baseline" {
		delayStart := time.Now()
		if err := e.hostWait(ctx); err != nil {
			if scope != nil {
				scope.End()
			}
			return a, err
		}
		trial.DelayNS = time.Since(delayStart).Nanoseconds()
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
	if scope == nil {
		trial.Correlation = "uninstrumented"
		if err := launchRequest(); err != nil {
			return a, err
		}
	} else {
		var entry *cudaworkload.HostSubmission
		sub, err := scope.NativeOperation(ctx, contextStream, "cuLaunchKernel", e.evidence, func() error {
			// NativeOperation calls this only after correlation entry, on
			// the pinned thread that makes the launches.
			observed := scope.NewSubmission(0, contextStream, cudaworkload.GraphReplayGeneration{}, "cuLaunchKernel", e.evidence)
			entry = &observed
			return launchRequest()
		})
		scope.End()
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
