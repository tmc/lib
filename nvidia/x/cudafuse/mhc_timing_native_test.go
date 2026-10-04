//go:build native

package cudafuse

import (
	"context"
	"errors"
	"io"
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

type mhcTimingReceipt struct {
	Config                                                MHCConfig
	Variant, EventRegion                                  string
	Trial                                                 int
	GPUEvent, CUPTI                                       mhcTimeMeasurement
	CompileLoadWallNanoseconds, AllocationWallNanoseconds int64
	KernelReceipts                                        []KernelReceipt
	Stats                                                 ExecutionStats
	EveryOutputElementChecked                             bool
}

func mhcTimingTrial(t *testing.T, dev *cuda.Device, stream *cuda.Stream, f mhcFixture, in MHCInputs, fused bool, trial int) mhcTimingReceipt {
	t.Helper()
	sources, err := MHCSource(f.config)
	if err != nil {
		t.Fatal(err)
	}
	list := sources[:]
	variant, region := "two-stage", "ordered normalization and post kernels"
	if fused {
		s, err := MHCFusedSource(f.config)
		if err != nil {
			t.Fatal(err)
		}
		list = []FusedKernel{s}
		variant, region = "fused", "single fused kernel"
	}
	base := nativeExecutionOps(dev, stream)
	// This owner retains only modules not yet transferred into the execution.
	// A launched execution retains its own transferred modules on failed drain.
	untransferred := &Execution{ops: base}
	t.Cleanup(func() {
		if err := untransferred.Close(); err != nil {
			t.Error(err)
		}
	})
	loaded := map[string]*executionKernel{}
	compileStart := time.Now()
	for _, s := range list {
		k, err := base.compile(s)
		if k != nil {
			untransferred.kernels = append(untransferred.kernels, k)
		}
		if err != nil {
			var pending io.Closer
			if errors.As(err, &pending) {
				untransferred.pending = append(untransferred.pending, pending)
			}
			t.Fatal(err)
		}
		loaded[s.Name] = k
	}
	compileWall := time.Since(compileStart)
	start, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := start.Close(); err != nil {
			t.Error(err)
		}
	})
	end, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := end.Close(); err != nil {
			t.Error(err)
		}
	})
	ptrs, releases, err := mhcBorrowFP32(dev, f.config, in)
	if err != nil {
		t.Fatal(err)
	}
	var allocationWall time.Duration
	ops := base
	launches := 0
	ops.alloc = func(n int) (*executionBuffer, error) {
		begin := time.Now()
		b, err := base.alloc(n)
		allocationWall += time.Since(begin)
		return b, err
	}
	ops.compile = func(s FusedKernel) (*executionKernel, error) {
		k := loaded[s.Name]
		if k == nil {
			return nil, errors.New("mhc: timing entry not precompiled")
		}
		delete(loaded, s.Name)
		for i, owner := range untransferred.kernels {
			if owner == k {
				untransferred.kernels[i] = nil
				break
			}
		}
		launch := k.launch
		k.launch = func(ptrs []driver.DevicePtr, n int) error {
			if launches == 0 {
				if err := start.Record(stream); err != nil {
					return err
				}
			}
			if err := launch(ptrs, n); err != nil {
				return err
			}
			launches++
			if launches == len(list) {
				return end.Record(stream)
			}
			return nil
		}
		return k, nil
	}
	var e *Execution
	if fused {
		e, err = mhcFusedExecute(context.Background(), f.config, ptrs, ops, releases)
	} else {
		e, err = mhcExecute(context.Background(), f.config, ptrs, ops, releases)
	}
	if e != nil {
		t.Cleanup(func() {
			if err := e.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	// Compilation/load and all allocations precede the first recorded event.
	// Readback/validation and cleanup follow the end event; neither is a wall
	// fallback for measured execution. Stream completion precedes elapsed read.
	duration, err := end.ElapsedSince(start)
	if err != nil {
		t.Fatal(err)
	}
	measurement, err := mhcGPUEventTime(duration)
	if err != nil {
		t.Fatal(err)
	}
	out, err := e.Readback(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := mhcOracle(f)
	receipts := e.Receipts()
	source := ""
	for _, s := range list {
		source += s.Source
	}
	target := receipts[0].Target
	if err := cudatest.CompareCUDAOutputFP32(out, want.out, mhcPolicy(f, source, target, len(out))); err != nil {
		t.Fatal("timed variant failed reference", err)
	}
	if e.Stats().Launches != len(list) || launches != len(list) {
		t.Fatal("timed launch count mismatch")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	return mhcTimingReceipt{f.config, variant, region, trial, measurement, mhcTimeMeasurement{Status: "Unmeasured"}, int64(compileWall), int64(allocationWall), receipts, e.Stats(), true}
}

func TestNativeMHCEventTiming(t *testing.T) {
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("native mhc timing requires gpu-lock")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	})
	// Two matched configs, one unreported warmup and three interleaved trials
	// per variant: 24 launches total. Every run checks the same FP64 oracle.
	for _, f := range mhcFixtures() {
		if f.config.Repeat != 10 || (f.config.Width != 33 && f.config.Width != 129) {
			continue
		}
		t.Run(f.name, func(t *testing.T) {
			in := mhcAllocInputs(t, dev, f)
			var receipts []mhcTimingReceipt
			for trial := -1; trial < 3; trial++ {
				order := []bool{false, true}
				if trial%2 == 1 {
					order = []bool{true, false}
				}
				for _, fused := range order {
					r := mhcTimingTrial(t, dev, stream, f, in.inputs, fused, trial)
					in.check(t)
					if trial >= 0 {
						receipts = append(receipts, r)
					}
				}
			}
			mhcWriteJSON(t, os.Getenv("MHC_TIMING_DIR"), f.name+".json", receipts)
			for _, r := range receipts {
				t.Logf("%s trial=%d GPU event=%dns compile/load wall=%dns allocation wall=%dns CUPTI=%s", r.Variant, r.Trial, *r.GPUEvent.Nanoseconds, r.CompileLoadWallNanoseconds, r.AllocationWallNanoseconds, r.CUPTI.Status)
			}
		})
	}
}
