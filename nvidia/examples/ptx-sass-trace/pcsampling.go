package main

// PC sampling is off by default. The tool's primary measurement is a
// whole-launch CUDA event interval, and enabling the hardware PC collector
// perturbs exactly that number: it adds per-launch collection overhead, and
// serialized collection changes kernel concurrency outright. The two
// measurements answer different questions, so they are separate runs rather
// than one default.
//
// Enabling it also carries preconditions the interval path does not have.
// Collection is admitted only on Linux amd64/arm64 against a CUPTI whose
// version is exactly 130001 (CUDA 13.0), the process must own PC sampling
// exclusively and cannot detect other registrants such as nsys or ncu, and the
// driver must permit non-admin profiling
// (NVreg_RestrictProfilingToAdminUsers=0). A default that fails or silently
// degrades under any of those is worse than an explicit flag.

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// pcSamplingResult reports what a sampling run actually collected. A high
// resolution fraction over a batch that dropped most of its samples is not a
// working attribution path, so the loss counters travel with the ratio.
type pcSamplingResult struct {
	PeriodExponent     uint32 `json:"period_exponent"`
	SelectedReasons    int    `json:"selected_reasons"`
	Batches            int    `json:"batches"`
	PCs                int    `json:"pcs"`
	TotalSamples       uint64 `json:"total_samples"`
	DroppedSamples     uint64 `json:"dropped_samples"`
	NonUserSamples     uint64 `json:"non_user_samples"`
	HardwareBufferFull bool   `json:"hardware_buffer_full"`
	PCsResolvedToPTX   int    `json:"pcs_resolved_to_ptx"`
	// RecordErrors counts batches that were collected but failed to reach the
	// profile. Those samples are in total_samples and in no sample type, so the
	// reconciliation identity stops closing by exactly their count; this field
	// says so directly rather than leaving the identity as the only detector.
	RecordErrors   int              `json:"record_errors,omitempty"`
	UnmappedStalls map[string]int64 `json:"unmapped_stalls,omitempty"`
	// ReasonRouting records where every reason the hardware inventory names
	// would be counted, sampled on this run or not. The reconciliation identity
	// sums all the stall types, so it stays closed when a reason is counted in
	// the wrong one; this map is what makes that auditable from the bundle.
	ReasonRouting map[string]string `json:"reason_routing,omitempty"`
	// ReasonSamples records how much each reason actually contributed, which is
	// what makes the routing falsifiable against this same file. It is also the
	// only place the long_scoreboard and short_scoreboard split survives: both
	// reach stall_scoreboard, and only the first means the warps were waiting
	// on memory data.
	ReasonSamples map[string]int64 `json:"reason_samples,omitempty"`
}

// Resolution reports the fraction of sampled PCs that resolved to a PTX
// statement. It is the number that distinguishes a working bridge from one
// whose resolver rejects everything.
func (r *pcSamplingResult) Resolution() float64 {
	if r == nil || r.PCs == 0 {
		return 0
	}
	return float64(r.PCsResolvedToPTX) / float64(r.PCs)
}

// pcSamplingBudget sizes the native buffers for a reason count. The output
// buffer holds two halves, each carrying an 80 byte header and one 64 byte
// record per PC plus 8 bytes per selected reason, so the PC capacity follows
// from the byte budget rather than being asserted independently of it.
func pcSamplingBudget(reasons int, period uint32) (cuptipcsampling.Config, error) {
	if reasons < 1 {
		return cuptipcsampling.Config{}, errors.New("pc sampling: empty stall reason inventory")
	}
	if period < 5 || period > 31 {
		return cuptipcsampling.Config{}, fmt.Errorf("pc sampling: period exponent %d outside 5 through 31", period)
	}
	const (
		outputBytes = 1 << 20
		nameBytes   = 256
		maxNames    = 4096
	)
	stride := uint64(64) + uint64(reasons)*8
	capacity := (outputBytes/2 - 80) / stride
	if capacity > maxNames/2 {
		capacity = maxNames / 2
	}
	if capacity < 1 {
		return cuptipcsampling.Config{}, fmt.Errorf("pc sampling: %d reasons exceed the output budget", reasons)
	}
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
		MaxNames:          maxNames,
		MaxNameBytes:      nameBytes,
		MaxTotalNameBytes: maxNames * nameBytes,
	}, nil
}

// stallReasons reads the native reason inventory for cuCtx, enabling sampling
// for the duration of the query and disabling it again.
func stallReasons(cuCtx cupti.Context) (inv cuptipcsampling.ReasonInventory, err error) {
	disable, err := cuptipcsampling.Enable(cuCtx)
	if err != nil {
		return cuptipcsampling.ReasonInventory{}, err
	}
	defer func() { err = errors.Join(err, disable()) }()
	return cuptipcsampling.StallReasons(cuCtx, 1<<10)
}

// withPCSampling collects PC samples for the duration of fn and records them
// into prof. It starts the collector before fn and drains after, because Read
// never synchronizes the device and Final is a separate native buffer rather
// than a replacement for the batches already read.
//
// The returned result is valid even when fn fails, so a partial run still
// reports what it collected.
func withPCSampling(ctx context.Context, cuCtx cupti.Context, code *cudacapture.Code, cubin []byte, rec *launchRecorder, period uint32, fn func(context.Context)) (res *pcSamplingResult, err error) {
	// The reason inventory cannot be read on a context where sampling is off,
	// and the configuration Start needs is built from that inventory, so this
	// enable is scoped to the query alone. Start enables the context again on
	// its own terms and owns the matching disable.
	inv, err := stallReasons(cuCtx)
	if err != nil {
		return nil, fmt.Errorf("query stall reasons: %w", err)
	}
	config, err := pcSamplingBudget(len(inv.Reasons), period)
	if err != nil {
		return nil, err
	}
	// Select every reason the inventory reports; the stall breakdown is the
	// reason to sample at all, and the budget above is sized for this count.
	for _, r := range inv.Reasons {
		config.ReasonIDs = append(config.ReasonIDs, r.Index)
	}

	session, err := cuptipcsampling.Start(cuCtx, config)
	if err != nil {
		// Start returns a nonnil session with an error when it retains cleanup
		// ownership; closing it is the caller's responsibility either way.
		if session != nil {
			err = errors.Join(err, session.Close())
		}
		return nil, fmt.Errorf("start pc sampling: %w", err)
	}

	resolver := cuptipcsampling.NewCaptureResolverFromCubin(code, cubin)
	// HostFrames resolves per kernel at record time rather than here: the
	// samples are drained after fn returns, so a stack taken now is the
	// recorder's own, identical on every row, and a constant stack filters
	// like attribution without being any.
	bridge := &cuptipcsampling.ProfileBridge{
		Session:    rec.prof,
		Inventory:  inv,
		Resolver:   resolver,
		PTXOffsets: rec.ptxLines,
		HostFrames: func(loc cuptipcsampling.CodeLocation) []cudaprof.Frame {
			return rec.launchStack(loc.KernelName)
		},
		// Same argument as HostFrames: the labelled context the workload ran
		// under is gone by the time CUPTI hands the samples over, so the labels
		// have to come from what the launches shared.
		LabelFunc: func(loc cuptipcsampling.CodeLocation) map[string]string {
			return rec.launchLabels(loc.KernelName)
		},
	}
	res = &pcSamplingResult{PeriodExponent: period, SelectedReasons: len(config.ReasonIDs)}

	fn(ctx)

	if stopErr := session.Stop(); stopErr != nil {
		err = errors.Join(err, fmt.Errorf("stop pc sampling: %w", stopErr))
	}
	for {
		batch, readErr := session.Read()
		record(ctx, res, bridge, resolver, batch)
		if readErr != nil {
			err = errors.Join(err, fmt.Errorf("read pc samples: %w", readErr))
			break
		}
		if batch.RemainingPCs == 0 {
			break
		}
	}
	closeErr := session.Close()
	if final, ok := session.Final(); ok {
		record(ctx, res, bridge, resolver, final)
	}
	if closeErr != nil {
		err = errors.Join(err, fmt.Errorf("close pc sampling: %w", closeErr))
	}
	res.UnmappedStalls = bridge.UnmappedStalls
	res.ReasonRouting = inv.Routing()
	res.ReasonSamples = bridge.ReasonSamples
	return res, err
}

// record accumulates one batch into res and routes it into the bridge. It
// counts resolutions separately from recording them: RecordBatch resolves
// internally without reporting, and the resolved fraction is the evidence that
// attribution reached PTX at all.
func record(ctx context.Context, res *pcSamplingResult, bridge *cuptipcsampling.ProfileBridge, resolver cuptipcsampling.CodeResolver, batch cuptipcsampling.Batch) {
	res.DroppedSamples += batch.DroppedSamples
	res.NonUserSamples += batch.NonUserSamples
	if batch.HardwareBufferFull != 0 {
		res.HardwareBufferFull = true
	}
	if !batch.Valid {
		return
	}
	res.Batches++
	res.TotalSamples += batch.TotalSamples
	for _, pc := range batch.PCs {
		if pc.Name == "" {
			continue
		}
		res.PCs++
		loc := resolver.ResolvePC(cuptipcsampling.ModuleID{CRC: pc.CRC}, pc.Name, pc.Offset)
		if loc.PTXLine > 0 {
			res.PCsResolvedToPTX++
		}
	}
	// A batch that fails to record is a reporting failure, not a collection
	// failure: the counters above already hold what it contained, so the capture
	// keeps going rather than discarding a workload that ran. The count travels
	// with it because the loss is otherwise visible only as the reconciliation
	// identity falling short, and a capture path without that identity would
	// lose the samples silently.
	if err := bridge.RecordBatch(ctx, batch); err != nil {
		res.RecordErrors++
	}
}
