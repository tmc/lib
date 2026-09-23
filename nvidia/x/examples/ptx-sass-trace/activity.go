package main

// Activity tracing is off by default, for the same reason PC sampling is: it
// needs preconditions the CUDA event path does not have. cuptiactivity.Start
// takes process-wide ownership of the CUPTI activity enable/disable and
// buffer-callback APIs and cannot detect another registrant such as nsys or
// ncu, and it admits only CUPTI 13.x libraries whose native record sizes match
// the generated Go layouts. A default that fails or degrades under either is
// worse than an explicit flag.
//
// What it adds is the one thing neither of the other tiers can report. A CUDA
// event interval is per launch and only where this example chose to record one;
// a PC sample says where execution was and never for how long, so a share of
// samples is a share of samples and never a share of elapsed time. Activity
// records carry a device start and end per kernel, so they establish how many
// kernels the device actually ran, how long each ran, and -- against a
// timestamp bracket taken from the same CUPTI clock -- how much of the span no
// kernel was executing at all.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// activityStatus values. Only statusCollected means the numbers below may be
// read; every other value means the tier did not establish what it reports, and
// the capture is refused rather than written with a plausible zero in it.
const (
	statusCollected = "collected"
	statusLost      = "lossy"
	statusEmpty     = "empty"
	statusFailed    = "failed"
)

// kernelActivity is one kernel's device-side totals. Launches counts activity
// records, not host launch calls: a launch the device never ran produces no
// record, and a graph replay produces one record per node.
type kernelActivity struct {
	Name     string `json:"name"`
	Launches int    `json:"launches"`
	// BusyNanos sums this kernel's device intervals. Two launches of the same
	// kernel can overlap on one device, so this is a sum of intervals and not
	// an elapsed duration.
	BusyNanos int64 `json:"busy_nanoseconds"`
}

// activityResult reports what the CUPTI activity tier established around one
// workload. Every field is labelled measured, derived, or UNMEASURED, because
// the three are not interchangeable and the difference is invisible once they
// are printed side by side.
type activityResult struct {
	Status string `json:"status"`

	// MEASURED. KernelLaunches is the number of concurrent-kernel activity
	// records with a usable interval; IncompleteKernels is the number without
	// one, which are counted as launches and contribute no duration.
	KernelLaunches    int `json:"kernel_launches"`
	IncompleteKernels int `json:"incomplete_kernels"`
	// MEASURED. SpanNanos is a CUPTI timestamp bracket taken immediately
	// before and after the workload, in the same clock domain as the record
	// timestamps below. It is not Go wall-clock time and the two must not be
	// differenced.
	SpanNanos int64 `json:"span_nanoseconds"`

	// DERIVED. BusySumNanos sums per-kernel intervals. It exceeds the span
	// whenever kernels overlap, so it is not a fraction of anything; compare
	// OccupiedNanos to the span instead.
	BusySumNanos int64 `json:"kernel_busy_sum_nanoseconds"`
	// DERIVED. OccupiedNanos is the union of the kernel intervals: the time
	// during which at least one kernel was executing. This is the only
	// device-time quantity here measured in the same units as the span.
	OccupiedNanos int64 `json:"device_occupied_nanoseconds"`
	// DERIVED. IdleNanos is SpanNanos-OccupiedNanos: span time with no kernel
	// executing. It does not mean the device was doing nothing -- copies,
	// memset and driver work are not kernels -- and it says nothing about how
	// many SMs were busy while a kernel ran.
	IdleNanos int64 `json:"device_idle_nanoseconds"`
	// DERIVED. OverlapNanos is BusySumNanos-OccupiedNanos, the amount by which
	// concurrency makes the sum incomparable to the span. Zero means the
	// kernels ran one at a time and the sum is the occupancy.
	OverlapNanos int64 `json:"kernel_overlap_nanoseconds"`

	// SMOccupancy is UNMEASURED and says so in the profile rather than being
	// omitted. Activity records carry no per-SM residency: one block resident
	// on one of N SMs makes the device wall-busy and 1/N SM-busy, and nothing
	// collected here distinguishes the two. Deriving an SM-seconds number from
	// grid dimensions would be a model, not a measurement.
	SMOccupancy string `json:"sm_occupancy"`

	// Losses. A record CUPTI dropped, a buffer the pool refused, or a callback
	// that failed all reduce the counts above without changing their shape, so
	// they travel with the numbers rather than being checked once and dropped.
	PendingBuffers    int    `json:"pending_buffers"`
	RefusedBuffers    uint64 `json:"refused_buffers"`
	DroppedRecords    uint64 `json:"dropped_records"`
	DiscardedRecords  uint64 `json:"discarded_records"`
	IncompleteRecords uint64 `json:"incomplete_records"`
	CallbackErrors    uint64 `json:"callback_errors"`
	Closed            bool   `json:"session_closed"`
	CollectorError    string `json:"collector_error,omitempty"`

	// TotalRecords counts every activity record retained, kernels included.
	TotalRecords int              `json:"total_records"`
	Kernels      []kernelActivity `json:"kernels,omitempty"`
}

// activityConfig bounds collection. The kernel tier is the only kind requested:
// driver and runtime API records would multiply the record count by the number
// of CUDA calls without adding a device-side quantity this report states.
func activityConfig() cuptiactivity.Config {
	return cuptiactivity.Config{
		Kinds:       []cupti.ActivityKind{cupti.ActivityKindConcurrentKernel},
		BufferBytes: 1 << 20,
		MaxBuffers:  16,
		MaxRecords:  1 << 16,
	}
}

// withActivity collects kernel activity records for the duration of fn.
//
// The timestamp bracket is taken inside the session so the span cannot be
// shorter than the interval the records come from. Close is called
// unconditionally and before this function returns, which is what forces
// CUPTI's buffers to be flushed: a Go process that reaches runtime exit with
// records still buffered loses them with no diagnostic at all. fn is expected
// to have synchronized its own device work -- Close flushes activity buffers,
// it does not wait for the GPU.
//
// The returned result is valid even when fn or the collector failed, so a
// partial run still reports what it collected and what it lost.
// unrequestedKindRefusal explains a Start failure that rejected the record
// layout of an activity kind this tier never asked to collect.
//
// cuptiactivity validates every kind it knows how to decode, not only the ones
// in Config.Kinds, so a CUPTI whose Memcpy record grew refuses a session that
// collects nothing but kernels. Without this note the flag fails with a message
// about a record type the user did not request and cannot avoid, which reads as
// a bug in the tier rather than as a toolkit constraint.
//
// The refused kind comes from the error's own Kind field, so this cannot
// disagree with the list the package actually validates. Returns "" when the
// failure is anything else, including a refusal of a kind that was requested:
// that one really is about what this tier collects.
func unrequestedKindRefusal(err error, requested []cupti.ActivityKind) string {
	var layout *cuptiactivity.LayoutError
	if !errors.As(err, &layout) {
		return ""
	}
	for _, k := range requested {
		if k == layout.Kind {
			return ""
		}
	}
	names := make([]string, 0, len(requested))
	for _, k := range requested {
		names = append(names, k.String())
	}
	return fmt.Sprintf("%s is not among the kinds this tier collects (%s); "+
		"the CUPTI on the loader path emits a %d-byte record where this build is generated for %d. "+
		"Point LD_LIBRARY_PATH at a CUPTI whose layouts this build was generated against",
		layout.Kind, strings.Join(names, ", "), layout.NativeSize, layout.GoSize)
}

func withActivity(ctx context.Context, fn func(context.Context)) (res *activityResult, err error) {
	session, err := cuptiactivity.Start(activityConfig())
	if err != nil {
		// Start returns a nonnil session with an error when it retains cleanup
		// ownership; closing it is the caller's responsibility either way.
		if session != nil {
			err = errors.Join(err, session.Close())
		}
		if note := unrequestedKindRefusal(err, activityConfig().Kinds); note != "" {
			err = fmt.Errorf("%w (%s)", err, note)
		}
		return &activityResult{Status: statusFailed, SMOccupancy: "UNMEASURED", CollectorError: err.Error()},
			fmt.Errorf("start activity collection: %w", err)
	}

	start, err := cupti.GetTimestamp()
	if err != nil {
		return &activityResult{Status: statusFailed, SMOccupancy: "UNMEASURED", CollectorError: err.Error()},
			errors.Join(fmt.Errorf("read cupti timestamp: %w", err), session.Close())
	}

	fn(ctx)

	end, tsErr := cupti.GetTimestamp()
	closeErr := session.Close()
	r := session.Result()
	res = summarizeActivity(r, start, end)
	if tsErr != nil {
		res.Status = statusFailed
		err = errors.Join(err, fmt.Errorf("read cupti timestamp: %w", tsErr))
	}
	if closeErr != nil {
		res.Status = statusFailed
		err = errors.Join(err, fmt.Errorf("close activity collection: %w", closeErr))
	}
	if res.Status != statusCollected && err == nil {
		err = errors.New(res.problem())
	}
	return res, err
}

// summarizeActivity reduces a collection result and a CUPTI timestamp bracket
// to the report. It is separate from the native session so the accounting is
// checkable without a device, and it decides the status: only a closed, clean,
// nonempty collection is readable, because a zero record count from a working
// tier and a zero from a tier that collected nothing are the same zero.
func summarizeActivity(r cuptiactivity.Result, start, end uint64) *activityResult {
	res := &activityResult{
		Status:            statusCollected,
		SMOccupancy:       "UNMEASURED",
		PendingBuffers:    r.PendingBuffers,
		RefusedBuffers:    r.RefusedBuffers,
		DroppedRecords:    r.DroppedRecords,
		DiscardedRecords:  r.DiscardedRecords,
		IncompleteRecords: r.IncompleteRecords,
		CallbackErrors:    r.CallbackErrors,
		Closed:            r.Closed,
		TotalRecords:      len(r.Records),
	}
	if r.FirstError != nil {
		res.CollectorError = r.FirstError.Error()
	}
	if end >= start {
		res.SpanNanos = int64(end - start)
	}

	type interval struct{ start, end uint64 }
	var intervals []interval
	byKernel := make(map[string]*kernelActivity)
	var order []string
	for _, rec := range r.Records {
		if rec.Kind != cupti.ActivityKindConcurrentKernel {
			continue
		}
		k := byKernel[rec.Name]
		if k == nil {
			k = &kernelActivity{Name: rec.Name}
			byKernel[rec.Name] = k
			order = append(order, rec.Name)
		}
		k.Launches++
		if !rec.Complete {
			res.IncompleteKernels++
			continue
		}
		res.KernelLaunches++
		d := int64(rec.End - rec.Start)
		k.BusyNanos += d
		res.BusySumNanos += d
		intervals = append(intervals, interval{rec.Start, rec.End})
	}
	sort.Strings(order)
	for _, name := range order {
		res.Kernels = append(res.Kernels, *byKernel[name])
	}

	// Union of the kernel intervals. This is the quantity comparable to the
	// span; the sum above is not, and the difference between them is the
	// concurrency that makes it so.
	sort.Slice(intervals, func(i, j int) bool { return intervals[i].start < intervals[j].start })
	var union uint64
	var cur interval
	for i, iv := range intervals {
		if i == 0 {
			cur = iv
			continue
		}
		if iv.start > cur.end {
			union += cur.end - cur.start
			cur = iv
			continue
		}
		if iv.end > cur.end {
			cur.end = iv.end
		}
	}
	if len(intervals) > 0 {
		union += cur.end - cur.start
	}
	res.OccupiedNanos = int64(union)
	res.OverlapNanos = res.BusySumNanos - res.OccupiedNanos
	// Idle is stated only when the records fall inside the bracket that is
	// supposed to contain them. A negative remainder means the span and the
	// record timestamps are not the interval and clock this code assumes.
	if res.SpanNanos >= res.OccupiedNanos {
		res.IdleNanos = res.SpanNanos - res.OccupiedNanos
	} else {
		res.Status = statusFailed
	}

	switch {
	case res.Status == statusFailed:
	case !r.Closed || r.FirstError != nil || r.CallbackErrors != 0:
		res.Status = statusFailed
	case r.PendingBuffers != 0 || r.RefusedBuffers != 0 || r.DroppedRecords != 0 || r.DiscardedRecords != 0:
		res.Status = statusLost
	case res.TotalRecords == 0 || res.KernelLaunches == 0:
		res.Status = statusEmpty
	}
	return res
}

// problem states why a result must not be read, in one line. It is the text a
// capture is refused with, so it names the counters rather than the status.
func (r *activityResult) problem() string {
	switch r.Status {
	case statusCollected:
		return ""
	case statusEmpty:
		return fmt.Sprintf("collected %d records and %d kernel launches: an empty collection is "+
			"indistinguishable from a tier that saw nothing, so it is not a measurement",
			r.TotalRecords, r.KernelLaunches)
	case statusLost:
		return fmt.Sprintf("lost records: pending_buffers=%d refused_buffers=%d dropped=%d discarded=%d",
			r.PendingBuffers, r.RefusedBuffers, r.DroppedRecords, r.DiscardedRecords)
	default:
		msg := fmt.Sprintf("collection failed: closed=%v callback_errors=%d incomplete=%d",
			r.Closed, r.CallbackErrors, r.IncompleteRecords)
		if r.CollectorError != "" {
			msg += "; first error: " + r.CollectorError
		}
		return msg
	}
}

// String is the line the capture prints. It names each quantity's provenance
// where it prints it, because a reader seeing six durations in a row has
// nothing else to separate the two that were measured from the four that were
// computed from them.
func (r *activityResult) String() string {
	return fmt.Sprintf("cupti activity (%s): %d kernel launches measured; span %dns measured; "+
		"device occupied %dns, idle %dns, kernel-interval sum %dns, overlap %dns (all derived); "+
		"sm occupancy UNMEASURED; dropped=%d discarded=%d refused_buffers=%d pending_buffers=%d",
		r.Status, r.KernelLaunches, r.SpanNanos, r.OccupiedNanos, r.IdleNanos, r.BusySumNanos,
		r.OverlapNanos, r.DroppedRecords, r.DiscardedRecords, r.RefusedBuffers, r.PendingBuffers)
}

// activityJSON renders the bundled activity.json.
func activityJSON(r *activityResult) ([]byte, error) {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("encode activity result: %w", err)
	}
	return append(data, '\n'), nil
}
