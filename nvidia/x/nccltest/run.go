package nccltest

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

// Options controls a finite single-process run. Each device gets one rank.
// Iterations and Checks must be positive; Warmup may be zero.
type Options struct {
	SyncIterations bool   // Synchronize after each aggregated timed iteration (reference -z 3).
	Offset         int    // Leading elements before send and receive buffers (reference -u).
	GraphLaunches  int    // Zero disables capture; positive values replay the timed graph.
	Register       bool   // Register allocated buffers with NCCL.
	DefaultStream  bool   // Use CUDA stream zero instead of creating a stream.
	Timing         Timing // How durations are combined across local ranks.
	Devices        []int32
	Warmup         int
	Iterations     int
	Aggregation    int // Timed collectives per iteration; zero means one.
	Checks         int
}

// Result records correctness and CUDA-event timing for one effective case.
// A successful run requires positive Comparisons and zero Wrong values.
type Result struct {
	SyncIterations   bool
	Offset           int
	GraphLaunches    int
	Register         bool
	DefaultStream    bool
	Timing           Timing
	Case             Case
	Layout           Layout
	Ranks            int
	NCCLVersion      int32
	Iterations       int
	Warmup           int
	Checks           int
	Aggregation      int
	TimedOperations  int
	GraphComparisons int // Replay comparisons, also included in Comparisons.
	GraphWrong       int // Replay mismatches, also included in Wrong.
	Comparisons      int
	Wrong            int
	GPUTime          time.Duration
	AlgorithmGBs     float64
	BusGBs           float64
}

type rankState struct {
	retainResources                    bool // Failed graph cleanup or communicator abort may leave live references.
	sendRegistration, recvRegistration nccl.Registration
	device                             driver.Device
	context                            driver.Context
	comm                               nccl.Comm
	stream                             driver.Stream
	start, end                         driver.Event
	send, recv                         driver.DevicePtr
	sendArg, recvArg                   driver.DevicePtr
}

// Run executes a correctness-checked case on the selected CUDA devices.
// It locks its OS thread throughout context selection and grouped submissions.
// Library or device failures are returned as errors, never converted into skips.
// Run blocks during native calls; execute it in a subprocess for a hard timeout.
func Run(c Case, o Options) (Result, error) { return run(c, o, false) }

// omit is used only by the physical negative-control test. It leaves receive
// storage poisoned while exercising the same allocation, copies, and checker.
func run(c Case, o Options, omit bool) (Result, error) {
	return runCycles(c, o, omit, 1, nil)
}

// RunCycles repeats a case using the same communicators and allocations.
// Count must be positive and yield must be non-nil. Each cycle resets inputs,
// checks correctness, warms up, and measures independently. Yield runs on the
// locked CUDA thread after each successful cycle; its error stops execution.
// The returned error also includes cleanup failures after the last yield.
func RunCycles(c Case, o Options, count int, yield func(Result) error) error {
	if yield == nil {
		return fmt.Errorf("nccltest: nil cycle callback")
	}
	_, err := runCycles(c, o, false, count, yield)
	return err
}

func runCycles(c Case, o Options, omit bool, count int, yield func(Result) error) (Result, error) {
	return runSweep(c, o, omit, []int{c.Bytes}, count, yield)
}

// RunSweep repeats sizes in order for each cycle, reusing communicators and
// allocations sized for the largest case. Sizes and count must be positive.
// Each yielded Result identifies its size through Case.Bytes. The callback
// and cleanup error rules are the same as RunCycles.
func RunSweep(c Case, o Options, sizes []int, count int, yield func(Result) error) error {
	if yield == nil {
		return fmt.Errorf("nccltest: nil sweep callback")
	}
	_, err := runSweep(c, o, false, sizes, count, yield)
	return err
}

// RunSweepContext is RunSweep with cancellation between samples. A zero count
// repeats until cancellation or a callback error; a negative count is invalid.
// Cancellation returns ctx.Err(), joined with any cleanup failure. A sample
// already in progress completes and is yielded before cancellation is checked.
// Native calls and callbacks cannot be interrupted; use a subprocess for a hard
// timeout. The context and callback must be non-nil.
func RunSweepContext(ctx context.Context, c Case, o Options, sizes []int, count int, yield func(Result) error) error {
	if ctx == nil {
		return fmt.Errorf("nccltest: nil context")
	}
	if yield == nil {
		return fmt.Errorf("nccltest: nil sweep callback")
	}
	_, err := runSweepContext(ctx, c, o, false, sizes, count, yield)
	return err
}

func runSweep(c Case, o Options, omit bool, sizes []int, count int, yield func(Result) error) (Result, error) {
	if count <= 0 {
		return Result{}, fmt.Errorf("nccltest: cycle count must be positive")
	}
	return runSweepContext(context.Background(), c, o, omit, sizes, count, yield)
}

func runSweepContext(ctx context.Context, c Case, o Options, omit bool, sizes []int, count int, yield func(Result) error) (result Result, err error) {
	c = c.copy()
	if count < 0 {
		return result, fmt.Errorf("nccltest: cycle count must not be negative")
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	aggregation, operations, err := iterationCounts(o.Iterations, o.Aggregation)
	if err != nil {
		return result, err
	}
	if o.SyncIterations && o.GraphLaunches != 0 {
		return result, fmt.Errorf("nccltest: synchronized iterations cannot use graphs")
	}
	if o.GraphLaunches < 0 || (o.GraphLaunches > 0 && o.DefaultStream) {
		return result, fmt.Errorf("nccltest: invalid graph configuration")
	}
	if o.GraphLaunches > 0 {
		if operations > int(^uint(0)>>1)/o.GraphLaunches {
			return result, fmt.Errorf("nccltest: overflowing graph operation count")
		}
		operations *= o.GraphLaunches
	}
	checkPasses := o.Checks
	if o.GraphLaunches > 0 {
		if checkPasses > int(^uint(0)>>1)/2 {
			return result, fmt.Errorf("nccltest: overflowing graph check count")
		}
		checkPasses *= 2
	}
	if !o.Timing.valid() {
		return result, fmt.Errorf("nccltest: unknown timing selection %q", o.Timing)
	}
	if len(sizes) == 0 {
		return result, fmt.Errorf("nccltest: empty size sweep")
	}
	sizes = append([]int(nil), sizes...)
	largest := 0
	for _, bytes := range sizes {
		c.Bytes = bytes
		if _, err := c.Layout(len(o.Devices)); err != nil {
			return result, err
		}
		largest = max(largest, bytes)
	}
	c.Bytes = largest
	size := c.DataType.Size()
	l, err := c.Layout(len(o.Devices))
	if err != nil {
		return result, err
	}
	allocationBytes, offsetBytes, err := bufferAllocation(l, size, o.Offset)
	if err != nil {
		return result, err
	}
	if o.Iterations <= 0 || o.Checks <= 0 || o.Warmup < 0 {
		return result, fmt.Errorf("nccltest: invalid iteration counts")
	}
	expectedComparisons, err := comparisonCount(c, l, len(o.Devices), checkPasses)
	if err != nil {
		return result, err
	}
	seen := make(map[int32]bool, len(o.Devices))
	for _, device := range o.Devices {
		if device < 0 || seen[device] {
			return result, fmt.Errorf("nccltest: invalid or repeated device %d", device)
		}
		seen[device] = true
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err = driver.Init(0); err != nil {
		return result, err
	}
	previous, err := driver.NewContextGetCurrent()
	if err != nil {
		return result, err
	}
	defer func() { err = errors.Join(err, previous.SetCurrent()) }()
	version, err := nccl.GetVersion()
	if err != nil {
		return result, err
	}
	states := make([]rankState, len(o.Devices))
	defer func() {
		// Drain the whole clique before releasing any rank's resources: peer
		// work may still reference a rank whose own stream has completed.
		drainErr := drainRanks(states, o.DefaultStream, driver.Context.SetCurrent, driver.Stream.Synchronize)
		if drainErr != nil {
			err = errors.Join(err, fmt.Errorf("nccltest: drain communicator clique (resources retained): %w", drainErr))
			return
		}
		for i := len(states) - 1; i >= 0; i-- {
			s := &states[i]
			if s.retainResources {
				err = errors.Join(err, fmt.Errorf("nccltest: rank %d resources retained after incomplete cleanup", i))
				continue
			}
			if s.context == 0 {
				continue
			}
			if e := s.context.SetCurrent(); e != nil {
				// Context-bound cleanup on another rank's current context is unsafe.
				// Preserve the resources and report the leak rather than freeing there.
				err = errors.Join(err, fmt.Errorf("nccltest: select rank %d context for cleanup (resources retained): %w", i, e))
				continue
			}
			if e := releaseCommunicator(s, nccl.Comm.Deregister, nccl.Comm.Destroy); e != nil {
				err = errors.Join(err, fmt.Errorf("nccltest: release rank %d communicator (resources retained): %w", i, e))
				continue
			}
			if s.start != 0 {
				err = errors.Join(err, s.start.Destroy())
			}
			if s.end != 0 {
				err = errors.Join(err, s.end.Destroy())
			}
			if s.send != 0 {
				err = errors.Join(err, s.send.MemFree())
			}
			if s.recv != 0 {
				err = errors.Join(err, s.recv.MemFree())
			}
			if s.stream != 0 {
				err = errors.Join(err, s.stream.Destroy())
			}
			err = errors.Join(err, s.device.PrimaryCtxRelease())
		}
	}()
	for i, ordinal := range o.Devices {
		s := &states[i]
		s.device, err = driver.NewDeviceGet(int(ordinal))
		if err != nil {
			return result, err
		}
		s.context, err = s.device.PrimaryCtxRetain()
		if err != nil {
			return result, err
		}
		if err = s.context.SetCurrent(); err != nil {
			return result, err
		}
		if !o.DefaultStream {
			s.stream, err = driver.NewStream(1)
			if err != nil {
				return result, err
			}
		}
		s.start, err = driver.NewEvent(0)
		if err != nil {
			return result, err
		}
		s.end, err = driver.NewEvent(0)
		if err != nil {
			return result, err
		}
		s.send, err = driver.MemAlloc(uint(allocationBytes))
		if err != nil {
			return result, err
		}
		s.recv, err = driver.MemAlloc(uint(allocationBytes))
		if err != nil {
			return result, err
		}
		s.sendArg = s.send + driver.DevicePtr(offsetBytes)
		s.recvArg = s.recv + driver.DevicePtr(offsetBytes)
		if c.InPlace {
			s.sendArg = s.recv + driver.DevicePtr(offsetBytes+i*l.SendOffset*size)
			s.recvArg = s.recv + driver.DevicePtr(offsetBytes) + driver.DevicePtr(i*l.RecvOffset*size)
		}
	}
	comms, err := nccl.InitAll(o.Devices)
	if err != nil {
		initErr := fmt.Errorf("nccltest: initialize communicator clique: %w", err)
		cleanupErr := abortPartial(states, comms, func(rank int, comm nccl.Comm) error {
			if err := states[rank].context.SetCurrent(); err != nil {
				return err
			}
			return comm.Abort()
		})
		return result, errors.Join(initErr, cleanupErr)
	}
	for i := range states {
		states[i].comm = comms[i]
	}
	if o.Register {
		for i := range states {
			s := &states[i]
			if err = s.context.SetCurrent(); err != nil {
				return result, err
			}
			bytes := uint(allocationBytes)
			s.sendRegistration, err = s.comm.Register(nccl.Buffer(s.send), bytes)
			if err != nil {
				return result, fmt.Errorf("nccltest: register send rank %d: %w", i, err)
			}
			s.recvRegistration, err = s.comm.Register(nccl.Buffer(s.recv), bytes)
			if err != nil {
				return result, fmt.Errorf("nccltest: register receive rank %d: %w", i, err)
			}
		}
	}
	reset := func() error {
		for rank := range states {
			s := &states[rank]
			if err := s.context.SetCurrent(); err != nil {
				return err
			}
			poison := poisonBuffer(c, max(l.SendCount, l.RecvCount))
			if c.DataType.integer() {
				poisonIntegers(c, l, len(states), rank, poison)
			}
			if err := (s.recv + driver.DevicePtr(offsetBytes)).MemcpyHtoD(unsafe.Pointer(&poison[0]), uint(len(poison))); err != nil {
				return err
			}
			inputData := inputBuffer(c, len(states), rank, l.SendCount)
			if err := s.sendArg.MemcpyHtoD(unsafe.Pointer(&inputData[0]), uint(len(inputData))); err != nil {
				return err
			}
		}
		return nil
	}
	nativeCollectives := version >= 22800
	grouped := len(states) > 1 || c.Operation == AllToAllV || c.Operation == SendRecv || ((c.Operation == Gather || c.Operation == Scatter) && !nativeCollectives) || (c.Operation == AllToAll && !nativeCollectives)
	submit := func() error {
		if omit {
			return nil
		}
		if c.Operation == Hypercube {
			return hypercube(c.DataType, l, states)
		}
		if grouped {
			if err := nccl.GroupStart(); err != nil {
				return err
			}
		}
		var callErr error
		for rank := range states {
			s := &states[rank]
			if len(states) > 1 {
				if e := s.context.SetCurrent(); e != nil {
					callErr = errors.Join(callErr, e)
					break
				}
			}
			callErr = errors.Join(callErr, execute(c, l, rank, states, nativeCollectives))
		}
		if grouped {
			return errors.Join(callErr, nccl.GroupEnd())
		}
		return callErr
	}
	syncAll := func() error {
		for i := range states {
			if err := states[i].context.SetCurrent(); err != nil {
				return err
			}
			if err := states[i].stream.Synchronize(); err != nil {
				return err
			}
		}
		return nil
	}
	cycle := func() (result Result, err error) {
		result = Result{SyncIterations: o.SyncIterations, Offset: o.Offset, GraphLaunches: o.GraphLaunches, Register: o.Register, DefaultStream: o.DefaultStream, Timing: o.Timing, Case: c.copy(), Layout: l, Ranks: len(states), NCCLVersion: version, Iterations: o.Iterations, Warmup: o.Warmup, Checks: o.Checks, Aggregation: aggregation, TimedOperations: operations}
		// Reset for every correctness pass, especially for in-place reductions.
		for check := 0; check < checkPasses; check++ {
			if err = reset(); err != nil {
				return result, err
			}
			if o.GraphLaunches > 0 && check >= o.Checks {
				graphs, captureErr := captureGraphs(states, submit)
				if captureErr == nil {
					captureErr = launchGraphs(states, graphs)
				}
				closeErr := closeGraphs(states, graphs)
				if err = errors.Join(captureErr, closeErr); err != nil {
					return result, err
				}
			} else if err = submit(); err != nil {
				return result, err
			}
			if err = syncAll(); err != nil {
				return result, err
			}
			for rank := range states {
				s := &states[rank]
				if err = s.context.SetCurrent(); err != nil {
					return result, err
				}
				got := make([]byte, l.RecvCount*size)
				if err = s.recvArg.MemcpyDtoH(unsafe.Pointer(&got[0]), uint(len(got))); err != nil {
					return result, err
				}
				count, wrong, e := CheckBytes(c, len(states), rank, got)
				if e != nil {
					return result, e
				}
				if o.GraphLaunches > 0 && check >= o.Checks {
					result.GraphComparisons += count
					result.GraphWrong += wrong
				}
				result.Comparisons += count
				result.Wrong += wrong
			}
		}
		if result.Comparisons != expectedComparisons {
			return result, fmt.Errorf("nccltest: checked %d values, want %d", result.Comparisons, expectedComparisons)
		}
		if result.Wrong != 0 {
			return result, fmt.Errorf("nccltest: %d wrong values in %d comparisons", result.Wrong, result.Comparisons)
		}
		if err = reset(); err != nil {
			return result, err
		}
		for i := 0; i < o.Warmup; i++ {
			if err = submit(); err != nil {
				return result, err
			}
		}
		if err = syncAll(); err != nil {
			return result, err
		}
		timedSubmit := func() error {
			var synchronize func() error
			if o.SyncIterations {
				synchronize = syncAll
			}
			return timedIterationsSync(o.Iterations, aggregation, submit, nccl.GroupStart, nccl.GroupEnd, synchronize)
		}
		var timedGraphs []rankGraph
		if o.GraphLaunches > 0 {
			timedGraphs, err = captureGraphs(states, timedSubmit)
			defer func() { err = errors.Join(err, closeGraphs(states, timedGraphs)) }()
			if err != nil {
				return result, err
			}
		}
		for i := range states {
			s := &states[i]
			if err = s.context.SetCurrent(); err != nil {
				return result, err
			}
			if err = s.start.Record(s.stream); err != nil {
				return result, err
			}
		}
		if o.GraphLaunches > 0 {
			for launch := 0; launch < o.GraphLaunches; launch++ {
				if err = launchGraphs(states, timedGraphs); err != nil {
					return result, err
				}
			}
		} else if err = timedSubmit(); err != nil {
			return result, err
		}
		for i := range states {
			s := &states[i]
			if err = s.context.SetCurrent(); err != nil {
				return result, err
			}
			if err = s.end.Record(s.stream); err != nil {
				return result, err
			}
		}
		if err = syncAll(); err != nil {
			return result, err
		}
		times := make([]float64, len(states))
		for i := range states {
			s := &states[i]
			if err = s.context.SetCurrent(); err != nil {
				return result, err
			}
			ms, e := s.start.ElapsedTime(s.end)
			if e != nil {
				return result, e
			}
			times[i] = float64(ms)
		}
		elapsed, err := rankTime(times, o.Timing)
		if err != nil {
			return result, err
		}
		result.GPUTime = time.Duration(elapsed / float64(operations) * float64(time.Millisecond))
		result.AlgorithmGBs, result.BusGBs, err = c.Bandwidth(len(states), result.GPUTime)
		return result, err
	}
	return repeatSweepContext(ctx, count, sizes, func(bytes int) (Result, error) {
		c.Bytes = bytes
		l, err = c.Layout(len(states))
		if err != nil {
			return Result{}, err
		}
		expectedComparisons, err = comparisonCount(c, l, len(states), checkPasses)
		if err != nil {
			return Result{}, err
		}
		for i := range states {
			s := &states[i]
			s.sendArg = s.send + driver.DevicePtr(offsetBytes)
			s.recvArg = s.recv + driver.DevicePtr(offsetBytes)
			if c.InPlace {
				s.sendArg = s.recv + driver.DevicePtr(offsetBytes+i*l.SendOffset*size)
				s.recvArg = s.recv + driver.DevicePtr(offsetBytes+i*l.RecvOffset*size)
			}
		}
		return cycle()
	}, yield)
}

func execute(c Case, l Layout, rank int, states []rankState, nativeCollectives bool) (err error) {
	s := &states[rank]
	size, datatype := c.DataType.Size(), c.DataType.native()
	send, recv := nccl.Buffer(s.sendArg), nccl.Buffer(s.recvArg)
	stream := nccl.Stream(s.stream)
	count := uint(l.ParamCount)
	n := len(states)
	op := nccl.Sum
	switch c.Reduction {
	case Product:
		op = nccl.Prod
	case Minimum:
		op = nccl.Min
	case Maximum:
		op = nccl.Max
	case Average:
		op = nccl.Avg
	case PreMulSum:
		op, err = createPreMulSum(s.comm, rank, c.DataType)
		if err != nil {
			return err
		}
		// The host-immediate scalar has been copied; upstream also destroys
		// the operator after the collective API returns, before GroupEnd.
		defer func() { err = errors.Join(err, s.comm.RedOpDestroy(op)) }()
	}
	switch c.Operation {
	case AllReduce:
		return s.comm.AllReduce(send, recv, count, datatype, op, stream)
	case AllGather:
		return s.comm.AllGather(send, recv, count, datatype, stream)
	case ReduceScatter:
		return s.comm.ReduceScatter(send, recv, count, datatype, op, stream)
	case Broadcast:
		return s.comm.Broadcast(send, recv, count, datatype, int32(c.Root), stream)
	case Reduce:
		return s.comm.Reduce(send, recv, count, datatype, op, int32(c.Root), stream)
	case AllToAll:
		if nativeCollectives {
			return s.comm.AlltoAll(send, recv, count, datatype, stream)
		}
		for peer := 0; peer < n; peer++ {
			offset := nccl.Buffer(peer * l.ParamCount * size)
			if err := s.comm.Send(send+offset, count, datatype, int32(peer), stream); err != nil {
				return err
			}
			if err := s.comm.Recv(recv+offset, count, datatype, int32(peer), stream); err != nil {
				return err
			}
		}
	case AllToAllV:
		sendOffset, recvOffset := 0, 0
		for peer := 0; peer < n; peer++ {
			ns, nr := c.peerElements(l.ParamCount, n, rank, peer, size), c.peerElements(l.ParamCount, n, peer, rank, size)
			if ns > 0 {
				if err := s.comm.Send(send+nccl.Buffer(sendOffset*size), uint(ns), datatype, int32(peer), stream); err != nil {
					return err
				}
			}
			if nr > 0 {
				if err := s.comm.Recv(recv+nccl.Buffer(recvOffset*size), uint(nr), datatype, int32(peer), stream); err != nil {
					return err
				}
			}
			sendOffset += ns
			recvOffset += nr
		}
	case SendRecv:
		if err := s.comm.Send(send, count, datatype, int32((rank+1)%n), stream); err != nil {
			return err
		}
		return s.comm.Recv(recv, count, datatype, int32((rank+n-1)%n), stream)
	case Gather:
		if nativeCollectives {
			return s.comm.Gather(send, recv, count, datatype, int32(c.Root), stream)
		}
		if err := s.comm.Send(send, count, datatype, int32(c.Root), stream); err != nil {
			return err
		}
		if rank == c.Root {
			for peer := 0; peer < n; peer++ {
				if err := s.comm.Recv(recv+nccl.Buffer(peer*l.ParamCount*size), count, datatype, int32(peer), stream); err != nil {
					return err
				}
			}
		}
	case Scatter:
		if nativeCollectives {
			return s.comm.Scatter(send, recv, count, datatype, int32(c.Root), stream)
		}
		if rank == c.Root {
			for peer := 0; peer < n; peer++ {
				if err := s.comm.Send(send+nccl.Buffer(peer*l.ParamCount*size), count, datatype, int32(peer), stream); err != nil {
					return err
				}
			}
		}
		return s.comm.Recv(recv, count, datatype, int32(c.Root), stream)
	case Hypercube:
		return fmt.Errorf("nccltest: hypercube submission requires stage barriers")
	default:
		return fmt.Errorf("nccltest: unsupported operation %q", c.Operation)
	}
	return nil
}

func hypercube(d DataType, l Layout, states []rankState) error {
	size, datatype := d.Size(), d.native()
	for rank := range states {
		s := &states[rank]
		if err := s.context.SetCurrent(); err != nil {
			return err
		}
		target := s.recvArg + driver.DevicePtr(rank*l.ParamCount*size)
		if target != s.sendArg {
			if err := target.MemcpyDtoDAsync(s.sendArg, uint(l.ParamCount*size), s.stream); err != nil {
				return err
			}
		}
	}
	for mask := 1; mask < len(states); mask <<= 1 {
		if err := nccl.GroupStart(); err != nil {
			return err
		}
		var callErr error
		for rank := range states {
			s := &states[rank]
			if err := s.context.SetCurrent(); err != nil {
				callErr = errors.Join(callErr, err)
				break
			}
			sendRank := rank &^ (mask - 1)
			recvRank := sendRank ^ mask
			callErr = errors.Join(callErr, s.comm.Send(nccl.Buffer(s.recvArg)+nccl.Buffer(sendRank*l.ParamCount*size), uint(l.ParamCount*mask), datatype, int32(rank^mask), nccl.Stream(s.stream)))
			callErr = errors.Join(callErr, s.comm.Recv(nccl.Buffer(s.recvArg)+nccl.Buffer(recvRank*l.ParamCount*size), uint(l.ParamCount*mask), datatype, int32(rank^mask), nccl.Stream(s.stream)))
		}
		if err := errors.Join(callErr, nccl.GroupEnd()); err != nil {
			return err
		}
	}
	return nil
}

// Pin the host-immediate scalar because the raw binding passes its address as
// an integer. NCCL copies it before returning; no device-lifetime pin is needed.
func createPreMulSum(comm nccl.Comm, rank int, d DataType) (nccl.RedOp, error) {
	if f, ok := d.smallFormat(); ok {
		scalar := f.encode(float64(1 + rank%2))
		if d.Size() == 1 {
			return createScalar(comm, uint8(scalar), d.native())
		}
		return createScalar(comm, scalar, d.native())
	}
	if d.integer() {
		scalar := uint64(1 + rank%2)
		switch d.Size() {
		case 1:
			return createScalar(comm, uint8(scalar), d.native())
		case 4:
			return createScalar(comm, uint32(scalar), d.native())
		case 8:
			return createScalar(comm, scalar, d.native())
		}
	}
	if d == Float64 {
		return createScalar(comm, float64(premulScalar(rank)), nccl.Float64)
	}
	return createScalar(comm, premulScalar(rank), nccl.Float32)
}

func createScalar[T ~float32 | ~float64 | ~uint8 | ~uint16 | ~uint32 | ~uint64](comm nccl.Comm, scalar T, datatype nccl.DataType) (nccl.RedOp, error) {
	var pins runtime.Pinner
	pins.Pin(&scalar)
	defer pins.Unpin()
	return comm.RedOpCreatePreMulSum(nccl.Buffer(uintptr(unsafe.Pointer(&scalar))), datatype, nccl.ScalarHostImmediate)
}

// Failed grouped initialization can leave allocated communicators. Abort them
// rather than invoking the collective normal-destruction path.
func abortPartial(states []rankState, comms []nccl.Comm, abort func(int, nccl.Comm) error) error {
	var result error
	for rank, comm := range comms {
		if comm == 0 {
			continue
		}
		if err := abort(rank, comm); err != nil {
			result = errors.Join(result, fmt.Errorf("nccltest: abort partial rank %d: %w", rank, err))
		}
	}
	if result != nil {
		// A partial clique may still own work or references on peer ranks.
		// Keep every rank alive when any abort could not be completed.
		for i := range states {
			states[i].retainResources = true
		}
	}
	return result
}

func bufferAllocation(l Layout, size, offset int) (bytes, shift int, err error) {
	n := max(l.SendCount, l.RecvCount)
	limit := int(^uint(0) >> 1)
	if size <= 0 || offset < 0 || n <= 0 || offset > limit/size || n > limit/size-offset {
		return 0, 0, fmt.Errorf("nccltest: invalid or overflowing buffer offset")
	}
	return (n + offset) * size, offset * size, nil
}

func repeatCycles(count int, cycle func() (Result, error), yield func(Result) error) (Result, error) {
	var result Result
	for i := 0; i < count; i++ {
		var err error
		result, err = cycle()
		if err != nil {
			return result, err
		}
		if yield != nil {
			if err := yield(result); err != nil {
				return result, err
			}
		}
	}
	return result, nil
}

func comparisonCount(c Case, l Layout, ranks, passes int) (int, error) {
	if c.Operation == Reduce || c.Operation == Gather {
		ranks = 1
	}
	n := 1
	for _, factor := range []int{l.RecvCount, ranks, passes} {
		if factor <= 0 || n > int(^uint(0)>>1)/factor {
			return 0, fmt.Errorf("nccltest: invalid or overflowing comparison count")
		}
		n *= factor
	}
	return n, nil
}

func repeatSweep(count int, sizes []int, run func(int) (Result, error), yield func(Result) error) (Result, error) {
	return repeatSweepContext(context.Background(), count, sizes, run, yield)
}

func repeatSweepContext(ctx context.Context, count int, sizes []int, run func(int) (Result, error), yield func(Result) error) (Result, error) {
	var result Result
	for remaining := count; count == 0 || remaining > 0; {
		for _, bytes := range sizes {
			if err := ctx.Err(); err != nil {
				return result, err
			}
			var err error
			result, err = run(bytes)
			if err != nil {
				return result, err
			}
			if yield != nil {
				if err := yield(result); err != nil {
					return result, err
				}
			}
		}
		if count > 0 {
			remaining--
		}
	}
	return result, ctx.Err()
}
