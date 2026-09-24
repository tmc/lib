package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalineinfo"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// Line attribution answers where in forward.cu the GPU spent its time.
//
// The hardware samples the program counter of one warp scheduler every 2^n
// cycles and CUPTI reports each sample as a kernel name and a byte offset
// into that kernel. nvdisasm prints the same offsets with the file and line
// each instruction was compiled from, so the two join with no relocation:
// the offset CUPTI reports is the address nvdisasm printed. That join is
// what cudalineinfo does, and it holds only if the module that ran is the
// module that was disassembled, which is why --lines runs the model on the
// module cudalineinfo compiled rather than compiling a second one.
//
// This costs a separate run under the sampler. It measures nothing the
// default path does, and the default path is unchanged by it.

// linesPeriod is the sampling period --lines starts at: one sample per 2^16
// GPU cycles. A whole request is tens of milliseconds of device time across
// thousands of launches, which is four orders of magnitude more than the
// single-kernel workloads shorter periods are meant for.
const linesPeriod = 16

// predictLines answers the request and reports the answer's cost per line of
// forward.cu. The request is answered twice: once to pay for the module's
// first launch of each kernel, and once under the sampler. A non-empty pprof
// names a file the same samples are written to as a profile.
func predictLines(dir, input string, prec precision, period uint32, pprof string) error {
	req, err := readInput(input)
	if err != nil {
		return err
	}

	// forward.cu is hand-written and holds many kernels, which is neither
	// of the things a unit was once assumed to be. Kernels names them all,
	// so every function in the module is attributed rather than all but one
	// of them filed as subroutines; carrying Source with no line map is what
	// tells cudalineinfo there is no generated tier below this file, so its
	// own lines are the program's.
	unit := cudalineinfo.Unit{
		Name:    forwardUnit,
		Source:  forwardCUDA,
		Kernels: forwardKernels(),
	}
	var a *cudalineinfo.Attribution
	m, err := openModel(dir, prec, func(dev *cuda.Device) (*kernels, error) {
		major, minor, err := dev.ComputeCapability()
		if err != nil {
			return nil, err
		}
		a, err = cudalineinfo.Compile(context.Background(), unit, cudalineinfo.Options{
			Arch: fmt.Sprintf("sm_%d%d", major, minor),
		})
		if err != nil {
			return nil, fmt.Errorf("compile forward kernels for attribution: %w", err)
		}
		return loadKernels(a.Module, dev)
	})
	if err != nil {
		return err
	}
	defer m.Close()

	cuCtx, err := m.dev.PrimaryContext()
	if err != nil {
		return err
	}
	defer cuCtx.Release()

	if _, err := m.Predict(req); err != nil {
		return err
	}
	var res []byte
	report, err := samplePCs(cupti.Context(cuCtx.Raw()), period, cuCtx.Synchronize, func() error {
		res, err = m.Predict(req)
		return err
	})
	if err != nil {
		return err
	}
	fmt.Println(string(res))

	p := cudalineinfo.CountSamples(a, report.inventory, report.Batches...)
	if err := cudalineinfo.WriteProfile(os.Stderr, a, p); err != nil {
		return err
	}
	if p.Unresolved > 0 {
		// Foreign PCs are expected here and the profile names them: most of
		// a request's device time is in the matrix products, which run in
		// cuBLAS. Unresolved ones are not. They are PCs in forward.cu's own
		// kernels at offsets those kernels do not have, and the module that
		// ran is the module that was disassembled, so there should be none.
		fmt.Fprintf(os.Stderr, "  %d samples fell in forward.cu's kernels at offsets they do not have\n", p.Unresolved)
	}
	if report.Loss() {
		fmt.Fprintf(os.Stderr, "  lost: %d samples dropped, %d left in hardware; raise the period above 2^%d\n",
			report.DroppedSamples, report.RemainingSamples, period)
	}
	if pprof != "" {
		if err := writePprof(pprof, a, report); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "  wrote %s\n", pprof)
	}
	return nil
}

// writePprof writes the run's samples to path as a pprof profile. The sample
// types are cudalineinfo's, derived from the collector's own reason list, so
// nothing here names one: a copy of that list kept by hand is what WritePprof
// exists to avoid.
func writePprof(path string, a *cudalineinfo.Attribution, report sampleReport) (err error) {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create pprof file: %w", err)
	}
	// The profile is gzipped and flushed by Close, so a close error is a
	// truncated profile and not a detail to drop.
	defer func() { err = errors.Join(err, f.Close()) }()
	if err := cudalineinfo.WritePprof(context.Background(), f, a, report.inventory, report.Batches...); err != nil {
		return fmt.Errorf("write pprof profile: %w", err)
	}
	return nil
}

// A sampleReport is a phase report with the reason inventory the samples
// were counted against, which is needed to read them and is only readable
// while sampling is enabled.
type sampleReport struct {
	cuptipcsampling.PhaseReport
	inventory cuptipcsampling.ReasonInventory
}

// samplePCs collects hardware PC samples while workload runs. The hardware
// buffer is fixed and a request outlives it many times over, so the phase
// drains on a timer while the workload runs and once more after sync; what
// the drain did not keep up with comes back in the report rather than
// silently shortening the profile.
func samplePCs(cuCtx cupti.Context, period uint32, sync func() error, workload func() error) (sampleReport, error) {
	var r sampleReport
	inv, err := stallReasons(cuCtx)
	if err != nil {
		return r, fmt.Errorf("stall reasons: %w", err)
	}
	r.inventory = inv
	cfg := samplingConfig(len(inv.Reasons), period)
	for _, reason := range inv.Reasons {
		cfg.ReasonIDs = append(cfg.ReasonIDs, reason.Index)
	}
	r.PhaseReport, err = cuptipcsampling.CapturePhase(context.Background(), cuCtx,
		cuptipcsampling.PhaseConfig{
			Config: cfg,
			Limits: cuptipcsampling.PhaseLimits{
				MaxBatches:   maxBatches,
				DrainTimeout: 30 * time.Second,
				PollInterval: 5 * time.Millisecond,
			},
		},
		sync,
		func(context.Context) error { return workload() })
	if err == nil {
		err = errors.Join(r.WorkloadErr, r.ConsumerErr)
	}
	return r, err
}

// stallReasons reads the native reason inventory, which is only readable
// while sampling is enabled on the context.
func stallReasons(cuCtx cupti.Context) (inv cuptipcsampling.ReasonInventory, err error) {
	disable, err := cuptipcsampling.Enable(cuCtx)
	if disable != nil {
		defer func() { err = errors.Join(err, disable()) }()
	}
	if err != nil {
		return inv, err
	}
	return cuptipcsampling.StallReasons(cuCtx, 1<<10)
}

// maxBatches bounds the drains one request is allowed. At the poll interval
// above it is several times what a request needs, and it must not exceed the
// session's read budget.
const maxBatches = 1 << 12

// samplingConfig sizes the collector for a reason count. The output holds two
// buffers, each with an 80 byte header and one 64 byte record per program
// counter plus 8 bytes per selected reason. A request touches every kernel in
// the module and every cuBLAS kernel it calls, so the buffers are sized for
// far more distinct counters than a single-kernel capture needs.
func samplingConfig(reasons int, period uint32) cuptipcsampling.Config {
	const outputBytes = 8 << 20
	stride := uint64(64) + uint64(reasons)*8
	capacity := min((outputBytes/2-80)/stride, 1<<14)
	return cuptipcsampling.Config{
		Period:            period,
		WorkerSleepMS:     5,
		MaxReasons:        reasons,
		ScratchBytes:      32 << 20,
		HardwareBytes:     32 << 20,
		MaxNativeBytes:    128 << 20,
		PCsPerBuffer:      int(capacity),
		MaxOutputBytes:    outputBytes,
		MaxReads:          maxBatches * 4,
		MaxNames:          1 << 16,
		MaxNameBytes:      256,
		MaxTotalNameBytes: 1 << 16 * 256,
	}
}
