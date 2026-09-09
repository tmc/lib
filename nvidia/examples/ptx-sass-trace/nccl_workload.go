package main

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

type ncclRankContext struct {
	ordinal int
	dev     *cuda.Device
	stream  *cuda.Stream
	start   *cuda.Event
	end     *cuda.Event
	sendPtr driver.DevicePtr
	recvPtr driver.DevicePtr
	uuid    driver.Uuid
}

type ncclResult struct {
	Mode         string                     `json:"mode"`
	Devices      []int                      `json:"devices"`
	Ranks        int                        `json:"ranks"`
	Elements     int                        `json:"elements"`
	Passes       int                        `json:"passes"`
	Verified     map[string]map[int]int     `json:"verified_by_rank"`
	Intervals    map[string]map[int][]int64 `json:"gpu_event_nanoseconds_by_rank"`
	HostDuration map[string][]int64         `json:"host_nanoseconds"`
}

func runNCCLWorkload(ctx context.Context, ordinals []int, mode string, elements, passes int, prof *cudaprof.Session) (*ncclResult, error) {
	if mode == "off" || len(ordinals) == 0 {
		return nil, nil
	}
	if len(ordinals) < 2 {
		return nil, fmt.Errorf("nccl requires at least 2 distinct devices, got %d", len(ordinals))
	}
	if elements < 1 || passes < 1 {
		return nil, fmt.Errorf("nccl requires elements >= 1 and passes >= 1")
	}

	ranks := len(ordinals)
	res := &ncclResult{
		Mode:         mode,
		Devices:      ordinals,
		Ranks:        ranks,
		Elements:     elements,
		Passes:       passes,
		Verified:     make(map[string]map[int]int),
		Intervals:    make(map[string]map[int][]int64),
		HostDuration: make(map[string][]int64),
	}

	for _, op := range []string{"allreduce", "allgather"} {
		res.Verified[op] = make(map[int]int)
		res.Intervals[op] = make(map[int][]int64)
	}

	// 1. Initialize CUDA devices, contexts, streams, and events
	rankCtxs := make([]*ncclRankContext, ranks)
	var cleanupMu sync.Mutex
	cleanup := func() {
		cleanupMu.Lock()
		defer cleanupMu.Unlock()
		for _, rc := range rankCtxs {
			if rc == nil {
				continue
			}
			if rc.sendPtr != 0 {
				_ = rc.sendPtr.MemFree()
				rc.sendPtr = 0
			}
			if rc.recvPtr != 0 {
				_ = rc.recvPtr.MemFree()
				rc.recvPtr = 0
			}
			if rc.start != nil {
				_ = rc.start.Destroy()
				rc.start = nil
			}
			if rc.end != nil {
				_ = rc.end.Destroy()
				rc.end = nil
			}
			if rc.stream != nil {
				_ = rc.stream.Destroy()
				rc.stream = nil
			}
			if rc.dev != nil {
				_ = rc.dev.Close()
				rc.dev = nil
			}
		}
	}
	defer cleanup()

	seenUUIDs := make(map[driver.Uuid]int)
	for r, ord := range ordinals {
		dev, err := cuda.NewDevice(ord)
		if err != nil {
			return nil, fmt.Errorf("open device %d: %w", ord, err)
		}
		var uuid driver.Uuid
		drvDev := driver.Device(ord)
		if err := drvDev.GetUuid(&uuid); err != nil {
			dev.Close()
			return nil, fmt.Errorf("get device %d UUID: %w", ord, err)
		}
		if origRank, exists := seenUUIDs[uuid]; exists {
			dev.Close()
			return nil, fmt.Errorf("device ordinal %d has duplicate UUID with rank %d", ord, origRank)
		}
		seenUUIDs[uuid] = r

		stream, err := dev.NewStream()
		if err != nil {
			dev.Close()
			return nil, fmt.Errorf("create stream on device %d: %w", ord, err)
		}
		start, err := dev.NewEvent()
		if err != nil {
			stream.Destroy()
			dev.Close()
			return nil, fmt.Errorf("create start event on device %d: %w", ord, err)
		}
		end, err := dev.NewEvent()
		if err != nil {
			start.Destroy()
			stream.Destroy()
			dev.Close()
			return nil, fmt.Errorf("create end event on device %d: %w", ord, err)
		}

		rankCtxs[r] = &ncclRankContext{
			ordinal: ord,
			dev:     dev,
			stream:  stream,
			start:   start,
			end:     end,
			uuid:    uuid,
		}
	}

	// 2. Initialize NCCL communicators
	comms := make([]nccl.Comm, ranks)
	devInt32s := make([]int32, ranks)
	for i, ord := range ordinals {
		devInt32s[i] = int32(ord)
	}

	var pin runtime.Pinner
	pin.Pin(&comms[0])
	pin.Pin(&devInt32s[0])
	defer pin.Unpin()

	if err := nccl.CommInitAll(&comms[0], int32(ranks), &devInt32s[0]); err != nil {
		return nil, fmt.Errorf("nccl CommInitAll: %w", err)
	}
	defer func() {
		for _, comm := range comms {
			if comm != 0 {
				_ = comm.Destroy()
			}
		}
	}()

	// Verify comms
	for r, comm := range comms {
		if comm == 0 {
			return nil, fmt.Errorf("nccl communicator rank %d is null", r)
		}
		uRank, err := comm.UserRank()
		if err != nil || int(uRank) != r {
			return nil, fmt.Errorf("nccl communicator %d reported rank %d: %v", r, uRank, err)
		}
	}

	// 3. Allocate device memory and initialize host inputs
	maxRecvElems := elements
	if mode == "allgather" || mode == "both" {
		maxRecvElems = elements * ranks
	}

	sendBytes := uint(elements * 4)
	recvBytes := uint(maxRecvElems * 4)

	for r, rc := range rankCtxs {
		var err error
		_ = rc.dev.WithContext(func() error {
			rc.sendPtr, err = driver.MemAlloc(sendBytes)
			if err != nil {
				return err
			}
			rc.recvPtr, err = driver.MemAlloc(recvBytes)
			return err
		})
		if err != nil {
			return nil, fmt.Errorf("allocate memory for rank %d: %w", r, err)
		}

		// Upload rank-dependent input
		hostIn := make([]int32, elements)
		for i := range hostIn {
			hostIn[i] = ncclInput(r, i)
		}
		_ = rc.dev.WithContext(func() error {
			return rc.sendPtr.MemcpyHtoD(unsafe.Pointer(&hostIn[0]), sendBytes)
		})
	}

	// Operations to run
	var ops []string
	switch mode {
	case "allreduce":
		ops = []string{"allreduce"}
	case "allgather":
		ops = []string{"allgather"}
	case "both":
		ops = []string{"allreduce", "allgather"}
	default:
		return nil, fmt.Errorf("unknown nccl mode %q", mode)
	}

	for _, op := range ops {
		for pass := -1; pass < passes; pass++ {
			isWarmup := pass < 0

			// Reset receive sentinels
			for _, rc := range rankCtxs {
				_ = rc.dev.WithContext(func() error {
					return rc.recvPtr.MemsetD32(0xdeadbeef, uint(maxRecvElems))
				})
			}

			// Group enqueue on locked OS thread
			t0 := time.Now()
			enqueueErr := func() error {
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()

				for _, rc := range rankCtxs {
					if err := rc.start.Record(rc.stream); err != nil {
						return fmt.Errorf("record start event: %w", err)
					}
				}

				if err := nccl.GroupStart(); err != nil {
					return fmt.Errorf("GroupStart: %w", err)
				}

				var collectiveErr error
				for r, rc := range rankCtxs {
					_ = rc.dev.WithContext(func() error {
						var err error
						if op == "allreduce" {
							err = comms[r].AllReduce(
								nccl.Buffer(rc.sendPtr),
								nccl.Buffer(rc.recvPtr),
								uint(elements),
								nccl.Int32,
								nccl.Sum,
								nccl.Stream(rc.stream.Raw()),
							)
						} else {
							err = comms[r].AllGather(
								nccl.Buffer(rc.sendPtr),
								nccl.Buffer(rc.recvPtr),
								uint(elements),
								nccl.Int32,
								nccl.Stream(rc.stream.Raw()),
							)
						}
						if err != nil {
							collectiveErr = err
						}
						return err
					})
				}

				endErr := nccl.GroupEnd()
				return errors.Join(collectiveErr, endErr)
			}()

			if enqueueErr != nil {
				return nil, fmt.Errorf("%s pass %d: %w", op, pass, enqueueErr)
			}

			// Record stop event and synchronize each stream
			for _, rc := range rankCtxs {
				if err := rc.end.Record(rc.stream); err != nil {
					return nil, err
				}
				if err := rc.end.Synchronize(ctx); err != nil {
					return nil, err
				}
			}
			hostElapsed := time.Since(t0)
			if !isWarmup {
				res.HostDuration[op] = append(res.HostDuration[op], hostElapsed.Nanoseconds())
			}

			// Measure intervals and validate results
			for r, rc := range rankCtxs {
				elapsed, err := rc.end.ElapsedSince(rc.start)
				if err != nil {
					return nil, err
				}
				if !isWarmup {
					res.Intervals[op][r] = append(res.Intervals[op][r], elapsed.Nanoseconds())
				}

				// Download output and verify against CPU oracle
				numExpected := elements
				if op == "allgather" {
					numExpected = elements * ranks
				}
				hostOut := make([]int32, numExpected)
				if err := rc.dev.WithContext(func() error {
					return rc.recvPtr.MemcpyDtoH(unsafe.Pointer(&hostOut[0]), uint(numExpected*4))
				}); err != nil {
					return nil, fmt.Errorf("%s rank %d memcpy DtoH: %w", op, r, err)
				}

				for i, got := range hostOut {
					var want int32
					if op == "allreduce" {
						var err error
						want, err = allReduceExpected(ranks, i)
						if err != nil {
							return nil, err
						}
					} else {
						want = allGatherExpected(elements, i)
					}
					if got != want {
						return nil, fmt.Errorf("%s rank %d elem %d: got %d, want %d", op, r, i, got, want)
					}
				}

				if !isWarmup {
					res.Verified[op][r] += numExpected
					// Record pprof sample
					if prof != nil {
						vals := cudaprof.SampleValues{
							GPUNanos: elapsed.Nanoseconds(),
						}
						_, _, callerLine, _ := runtime.Caller(0)
						prof.RecordStackSample(ctx, vals, []cudaprof.Frame{
							{FunctionName: "nccl." + op, FileName: "x/nccl/nccl.go", Line: 0},
							{FunctionName: "main.runNCCLWorkload", FileName: "nccl_workload.go", Line: int64(callerLine)},
						})
					}
				}
			}
		}
	}

	return res, nil
}
