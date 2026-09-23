package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"text/tabwriter"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudalineinfo"
	"github.com/tmc/lib/nvidia/x/cudaprofagent"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// profileConfig selects the runtime measurement. Both collectors need CUPTI
// and exclusive use of the device's profiling hardware, so neither is on by
// default and they run one after the other when both are asked for.
type profileConfig struct {
	pcSampling  bool   // sample warp program counters
	sassMetrics bool   // count executed instructions exactly
	pprof       string // write the PC samples as a pprof profile here
	period      uint   // PC sampling period exponent: one sample per 2^period cycles
	n           int    // elements per launch
	launches    int
}

// profile runs the first kernel under the selected collectors and reports
// each measure per Go line. The static table says which instructions a line
// became; this says how often the GPU was on them.
func profile(ctx context.Context, cfg profileConfig, a *cudalineinfo.Attribution, fk cudafuse.FusedKernel) error {
	if !cfg.pcSampling && !cfg.sassMetrics {
		return nil
	}
	if err := cuda.Init(); err != nil {
		return err
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		return err
	}
	cuCtx, err := dev.PrimaryContext()
	if err != nil {
		return err
	}
	r, err := newRun(dev, a, fk, inputs(fk, cfg.n))
	if err != nil {
		return err
	}
	defer r.close()
	// Each collector's workload is timed so the report can say what the
	// measurement cost. The time a collector spends reading its buffers
	// between launches counts: the samples are not there without it.
	var elapsed time.Duration
	workload := func(between func() error) error {
		start := time.Now()
		defer func() { elapsed = time.Since(start) }()
		for i := 0; i < cfg.launches; i++ {
			if err := r.launch(); err != nil {
				return err
			}
			if between != nil {
				if err := between(); err != nil {
					return err
				}
			}
		}
		return nil
	}

	// The baseline is the same launches with nothing attached, after a
	// warm-up run that pays for the module's first launch.
	if err := workload(nil); err != nil {
		return err
	}
	if err := workload(nil); err != nil {
		return err
	}
	costs := []cost{{"none", elapsed}}

	if cfg.pcSampling {
		fmt.Println()
		p, period, err := samplePCsAdaptive(ctx, cupti.Context(cuCtx.Raw()), a, cfg.period, cfg.pprof, workload)
		if err != nil {
			return fmt.Errorf("pc sampling: %w", err)
		}
		if period != cfg.period {
			fmt.Printf("  sampling period raised from 2^%d to 2^%d cycles: the hardware buffer overflowed\n", cfg.period, period)
		}
		costs = append(costs, cost{"pcsampling", elapsed})
		cudalineinfo.WriteProfile(os.Stdout, a, p)
		if cfg.pprof != "" {
			fmt.Printf("  pprof profile written to %s\n", cfg.pprof)
		}
	}
	if cfg.sassMetrics {
		fmt.Println()
		// The SASS metrics API works on the calling thread's current
		// context, so the whole collection runs with it pushed.
		var p cudalineinfo.Profile
		var counts []cudaprofagent.Count
		err := dev.WithContext(func() (err error) {
			p, counts, err = countExecuted(uintptr(cuCtx.Raw()), a, workload)
			return err
		})
		if err != nil {
			return fmt.Errorf("sass metrics: %w", err)
		}
		costs = append(costs, cost{"sassmetrics", elapsed})
		cudalineinfo.WriteProfile(os.Stdout, a, p)
		// Under cudaprof these exact counts become a sample type of the
		// capture, over the same instructions the agent sampled.
		if _, err := cudaprofagent.WriteCounts(a.Unit.Name, a.Module.CUBIN(), executedMetric, counts); err != nil {
			return fmt.Errorf("publish counts: %w", err)
		}
	}
	writeCost(os.Stdout, a.Unit.Name, cfg.launches, costs)
	return nil
}

// A cost is the wall time one collector took over the same launches.
type cost struct {
	collector string
	elapsed   time.Duration
}

// writeCost reports each collector against the uninstrumented baseline,
// which is the first entry.
func writeCost(w io.Writer, kernel string, launches int, costs []cost) {
	if len(costs) < 2 {
		return
	}
	base := costs[0].elapsed
	fmt.Fprintf(w, "\n%s: wall time of %d launches under each collector\n", kernel, launches)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  COLLECTOR\tWALL\tPER LAUNCH\tOVERHEAD")
	for _, c := range costs {
		over := "-"
		if c.collector != costs[0].collector {
			over = fmt.Sprintf("%+.1f%%", 100*(float64(c.elapsed)/float64(base)-1))
		}
		fmt.Fprintf(tw, "  %s\t%.1fms\t%.3fms\t%s\n", c.collector,
			float64(c.elapsed.Microseconds())/1000, float64(c.elapsed.Microseconds())/1000/float64(launches), over)
	}
	tw.Flush()
}

// samplePCsAdaptive collects samples, lengthening the period when the
// hardware buffer overflows. The buffer is fixed, so a short period over a
// long workload fills it and CUPTI reports running out of memory. Each
// retry samples a quarter as often; the period that worked is returned, so
// a report says what the numbers were taken at rather than a run failing on
// a knob the reader has to find by hand.
func samplePCsAdaptive(ctx context.Context, cuCtx cupti.Context, a *cudalineinfo.Attribution, period uint, pprofPath string, workload func(between func() error) error) (cudalineinfo.Profile, uint, error) {
	for {
		p, err := samplePCs(ctx, cuCtx, a, period, pprofPath, workload)
		if !errors.Is(err, cupti.ErrOutOfMemory) || period >= maxPeriod {
			return p, period, err
		}
		period = min(period+2, maxPeriod)
	}
}

// maxPeriod is the longest period the hardware accepts: one sample per 2^31
// cycles. A workload that overflows the buffer at that rate is not one this
// sampler can measure.
const maxPeriod = 31

// samplePCs collects hardware PC samples while workload runs and joins them
// to the Go lines of a. The hardware buffer is small and fills within a few
// launches, so the session is read between launches and once more after it
// stops. With pprofPath set, the same samples also go to a pprof profile
// whose stacks run from the SASS instruction up through the Go frames that
// built the site.
func samplePCs(ctx context.Context, cuCtx cupti.Context, a *cudalineinfo.Attribution, period uint, pprofPath string, workload func(between func() error) error) (p cudalineinfo.Profile, err error) {
	inv, err := stallReasons(cuCtx)
	if err != nil {
		return p, fmt.Errorf("stall reasons: %w", err)
	}
	config := samplingConfig(len(inv.Reasons), uint32(period))
	for _, r := range inv.Reasons {
		config.ReasonIDs = append(config.ReasonIDs, r.Index)
	}
	session, err := cuptipcsampling.Start(cuCtx, config)
	if err != nil {
		if session != nil {
			err = errors.Join(err, session.Close())
		}
		return p, fmt.Errorf("start: %w", err)
	}

	var batches []cuptipcsampling.Batch
	drain := func() error {
		for {
			batch, err := session.Read()
			batches = append(batches, batch)
			if err != nil {
				return fmt.Errorf("read: %w", err)
			}
			if batch.RemainingPCs == 0 {
				return nil
			}
		}
	}
	err = workload(drain)
	err = errors.Join(err, session.Stop())
	if err == nil {
		err = drain()
	}
	err = errors.Join(err, session.Close())
	if final, ok := session.Final(); ok {
		batches = append(batches, final)
	}
	if err != nil {
		return p, err
	}

	p = cudalineinfo.CountSamples(a, inv, batches...)
	if pprofPath != "" {
		err = writePprof(ctx, pprofPath, a, inv, batches)
	}
	return p, err
}

// writePprof records the batches into a pprof profile. The assembly -- the
// sample types, the resolver, the bridge -- is cudalineinfo.WritePprof's; the
// profile carries every column the bridge fills, including the inst_executed
// this example selected nowhere while the bridge counted into it.
func writePprof(ctx context.Context, path string, a *cudalineinfo.Attribution, inv cuptipcsampling.ReasonInventory, batches []cuptipcsampling.Batch) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := cudalineinfo.WritePprof(ctx, f, a, inv, batches...); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// countExecuted patches the kernel to count executed instructions per PC
// while workload runs, then joins the counts to the Go lines of a.
func countExecuted(cuCtx uintptr, a *cudalineinfo.Attribution, workload func(between func() error) error) (p cudalineinfo.Profile, counts []cudaprofagent.Count, err error) {
	chip, err := cuptisass.DeviceChipName(0)
	if err != nil {
		return p, nil, err
	}
	metric, err := cuptisass.MetricProperties(chip, executedMetric)
	if err != nil {
		return p, nil, err
	}
	session, err := cuptisass.NewSession(cuCtx, cuptisass.Config{
		Metrics: []cuptisass.MetricConfig{{MetricID: metric.ID, Granularity: cuptisass.GranularityGPU}},
	})
	if err != nil {
		return p, nil, err
	}
	defer func() { err = errors.Join(err, session.Close()) }()
	if err := session.Enable(); err != nil {
		return p, nil, err
	}
	err = workload(nil)
	records, flushErr := session.Flush()
	_, disableErr := session.Disable()
	if err = errors.Join(err, flushErr, disableErr); err != nil {
		return p, nil, err
	}
	return cudalineinfo.CountExecuted(a, metric.ID, records...), perPC(metric.ID, records), nil
}

// executedMetric is the CUPTI metric counting the instructions a program
// counter executed, summed over the warps that reached it.
const executedMetric = "smsp__sass_inst_executed"

// perPC is the same records in the form a capture carries them: one count
// per program counter, with the values of every other metric dropped.
func perPC(metricID uint64, records []cuptisass.Record) []cudaprofagent.Count {
	counts := make([]cudaprofagent.Count, 0, len(records))
	for _, rec := range records {
		var n int64
		for _, v := range rec.Values {
			if v.MetricID == metricID {
				n += int64(v.Value)
			}
		}
		if n == 0 {
			continue
		}
		counts = append(counts, cudaprofagent.Count{
			Function: rec.FunctionName,
			Offset:   uint64(rec.PCOffset),
			Value:    n,
		})
	}
	return counts
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

// samplingConfig sizes the collector for a reason count. The output buffer
// holds two halves, each with an 80 byte header and one 64 byte record per
// PC plus 8 bytes per selected reason. The hardware buffer is fixed, so a
// short period over a long workload overflows it; the period is the knob.
func samplingConfig(reasons int, period uint32) cuptipcsampling.Config {
	const outputBytes = 1 << 20
	stride := uint64(64) + uint64(reasons)*8
	capacity := min((outputBytes/2-80)/stride, 2048)
	return cuptipcsampling.Config{
		Period:            period,
		WorkerSleepMS:     10,
		MaxReasons:        reasons,
		ScratchBytes:      4 << 20,
		HardwareBytes:     4 << 20,
		MaxNativeBytes:    16 << 20,
		PCsPerBuffer:      int(capacity),
		MaxOutputBytes:    outputBytes,
		MaxReads:          1 << 12,
		MaxNames:          4096,
		MaxNameBytes:      256,
		MaxTotalNameBytes: 4096 * 256,
	}
}
