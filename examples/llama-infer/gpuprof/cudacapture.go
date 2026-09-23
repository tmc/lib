package gpuprof

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// Sample type and label names written by cudaprof captures.
const (
	kernelTimeType = "kernel_time"
	copyTimeType   = "copy_time"
	copyBytesType  = "copy_bytes"
	activityType   = "activity"

	subtypeLabel = "subtype"
	nameLabel    = "name"
	deviceLabel  = "device"

	kernelSubtype = "kernel"
)

// CaptureSummary reports what one CUDA execution capture retained.
//
// Every field is a sum over retained records. A capture may be a partial view
// of its run: the producer drops records when its admission queue fills, and a
// dropped record is not represented anywhere in the profile. Treat these as
// survivor totals, not run totals, unless the capture is separately qualified
// as complete.
type CaptureSummary struct {
	// KernelTimeNS and CopyTimeNS sum the kernel_time and copy_time sample
	// values. They are device-side durations, so they overlap in wall-clock
	// time when work runs concurrently on several streams and must not be
	// added together to form an elapsed time.
	KernelTimeNS int64
	CopyTimeNS   int64

	// CopyBytes sums the copy_bytes sample values.
	CopyBytes int64

	// Activities sums the activity counter, which counts every retained
	// record, including kinds this summary does not otherwise decode.
	Activities int64

	// KernelLaunches counts retained records labeled as kernel activity, and
	// KernelNames lists the distinct kernel names among them. A capture
	// written without name labels yields launches with no names, which is a
	// representation limit and not evidence that the kernels were anonymous.
	KernelLaunches int
	KernelNames    []string

	// Devices lists the distinct device labels observed, in sorted order.
	Devices []string

	// Qualification is what the capture says about its own completeness.
	// Consult it before reporting any field above as a run total.
	Qualification CaptureQualification
}

// ReadCaptureSummary reads the cudaprof capture at path and summarizes its
// retained activity. Path may be either a capture file or a capture directory.
//
// Reading uses the cudacapture package defaults, which bound how much of an
// untrusted artifact is held in memory.
func ReadCaptureSummary(path string) (*CaptureSummary, error) {
	return ReadCaptureSummaryLimits(path, cudacapture.Limits{})
}

// ReadCaptureSummaryLimits is ReadCaptureSummary with explicit read bounds.
//
// The defaults are what keeps a hostile artifact from exhausting the reader,
// so raise them only for a capture produced by this host's own run. A serving
// capture retains one journal per process generation and can exceed the
// default per-file bound.
func ReadCaptureSummaryLimits(path string, limits cudacapture.Limits) (*CaptureSummary, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("gpuprof: stat capture: %w", err)
	}

	var c *cudacapture.Capture
	if info.IsDir() {
		c, err = cudacapture.ReadDirLimits(path, limits)
	} else {
		var f *os.File
		f, err = os.Open(path)
		if err != nil {
			return nil, fmt.Errorf("gpuprof: open capture: %w", err)
		}
		defer f.Close()
		c, err = cudacapture.ReadLimits(f, limits)
	}
	if err != nil {
		return nil, fmt.Errorf("gpuprof: read capture %s: %w", path, err)
	}
	return summarizeCapture(c)
}

// summarizeCapture walks the capture profile once and accumulates the sample
// types cudaprof writes.
func summarizeCapture(c *cudacapture.Capture) (*CaptureSummary, error) {
	if c == nil || c.Profile == nil {
		return nil, errors.New("gpuprof: capture carries no profile")
	}

	// Resolve sample-type positions by name. A capture that carries none of
	// them is reported as an error rather than as a zeroed summary, because a
	// fabricated zero is indistinguishable from an idle run.
	idx := map[string]int{}
	for i, st := range c.Profile.SampleType {
		if st == nil {
			continue
		}
		switch st.Type {
		case kernelTimeType, copyTimeType, copyBytesType, activityType:
			idx[st.Type] = i
		}
	}
	if len(idx) == 0 {
		return nil, fmt.Errorf("gpuprof: capture has no cudaprof sample types (found %d types)", len(c.Profile.SampleType))
	}

	value := func(vals []int64, name string) int64 {
		i, ok := idx[name]
		if !ok || i >= len(vals) {
			return 0
		}
		return vals[i]
	}

	sum := &CaptureSummary{}
	kernels := map[string]bool{}
	devices := map[string]bool{}
	for _, s := range c.Profile.Sample {
		if s == nil {
			continue
		}
		sum.KernelTimeNS += value(s.Value, kernelTimeType)
		sum.CopyTimeNS += value(s.Value, copyTimeType)
		sum.CopyBytes += value(s.Value, copyBytesType)
		sum.Activities += value(s.Value, activityType)

		if d := firstLabel(s.Label, deviceLabel); d != "" {
			devices[d] = true
		}
		if firstLabel(s.Label, subtypeLabel) != kernelSubtype {
			continue
		}
		sum.KernelLaunches++
		if n := firstLabel(s.Label, nameLabel); n != "" {
			kernels[n] = true
		}
	}
	sum.KernelNames = sortedKeys(kernels)
	sum.Devices = sortedKeys(devices)

	q, err := readQualification(c.Files)
	if err != nil {
		return nil, err
	}
	sum.Qualification = q
	return sum, nil
}

// firstLabel returns the first value of key, or "" when the key is absent.
func firstLabel(labels map[string][]string, key string) string {
	vals := labels[key]
	if len(vals) == 0 {
		return ""
	}
	return vals[0]
}

func sortedKeys(m map[string]bool) []string {
	if len(m) == 0 {
		return nil
	}
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// captureStatusFile is the name cudaprof gives the status document it attaches
// to every capture.
const captureStatusFile = "capture-status.json"

// Whole-run completeness values published by cudaprof.
const (
	runComplete = "complete"
	runPartial  = "partial"
	runFailed   = "failed"
	runUnknown  = "unknown"
)

// CaptureQualification reports what a capture says about its own completeness.
//
// It answers a different question from CaptureSummary. The summary says what
// the capture retained; this says whether what it retained is the whole run.
// A benchmark that reports a duration from an unqualified capture is reporting
// a survivor sum as if it were a total, which is why Qualify exists.
type CaptureQualification struct {
	// Present reports whether the capture retained a status document. A
	// capture without one cannot qualify itself, and the absence is not
	// evidence that the run was complete.
	Present bool

	// Derived reports that RunCompleteness was computed here from the
	// per-generation dispositions rather than published by the writer. A
	// capture written before whole-run completeness existed publishes the
	// dispositions but no verdict.
	Derived bool

	RunCompleteness        string
	CaptureSuccess         bool
	TransportReady         bool
	CollectorFinalized     bool
	WorkerGenerations      int
	UnqualifiedGenerations int

	// ActivityRecords counts retained activity records only. RetainedRecords
	// sums the per-generation record counts across every kind, and is zero
	// when the status reports no per-generation counts.
	ActivityRecords uint64
	RetainedRecords uint64

	// DroppedRecords is kind-agnostic: the producer admits every record kind
	// through one queue and counts drops without separating them, so a drop
	// cannot be attributed to activity records specifically. Dividing it by
	// ActivityRecords would state an activity loss rate the receipt does not
	// support.
	DroppedRecords uint64

	Outcome string
}

// Qualify reports why the capture's totals are not run totals, or nil when
// they are.
//
// Both failure axes are reported, because they are independent: a run whose
// every generation finalized can still have lost records to the producer's
// admission queue, and a run that dropped nothing can still be missing whole
// worker generations.
func (q CaptureQualification) Qualify() error {
	if !q.Present {
		return errors.New("capture retained no status document, so its totals cannot be qualified")
	}
	var reasons []string
	if q.RunCompleteness != runComplete {
		reason := fmt.Sprintf("run completeness is %q", q.RunCompleteness)
		if q.UnqualifiedGenerations > 0 {
			reason += fmt.Sprintf(" (%d of %d worker generations did not qualify)", q.UnqualifiedGenerations, q.WorkerGenerations)
		}
		reasons = append(reasons, reason)
	}
	if q.DroppedRecords > 0 {
		// The denominator has to be the all-kinds retained count, because the
		// drop counter is all-kinds. Pairing it with ActivityRecords would
		// report an activity-specific loss rate the producer never measured.
		if q.RetainedRecords > 0 {
			offered := q.RetainedRecords + q.DroppedRecords
			reasons = append(reasons, fmt.Sprintf("the producer dropped %d of %d offered records (%.2f%% loss, all kinds)",
				q.DroppedRecords, offered, 100*float64(q.DroppedRecords)/float64(offered)))
		} else {
			reasons = append(reasons, fmt.Sprintf("the producer dropped %d records (all kinds; the status reports no retained total to divide by)",
				q.DroppedRecords))
		}
	}
	if len(reasons) == 0 {
		return nil
	}
	return errors.New("capture is not a complete view of its run: " + strings.Join(reasons, "; "))
}

// captureStatus is the subset of the cudaprof status document this package
// reads. Unknown members are ignored, so a newer writer does not break it.
type captureStatus struct {
	CaptureSuccess         bool   `json:"capture_success"`
	TransportReady         bool   `json:"transport_ready"`
	CollectorFinalized     bool   `json:"collector_finalized"`
	RunCompleteness        string `json:"run_completeness"`
	UnqualifiedGenerations uint64 `json:"unqualified_generations"`
	ActivityRecords        uint64 `json:"activity_records"`
	DroppedRecords         uint64 `json:"dropped_records"`
	Outcome                string `json:"outcome"`
	WorkerGenerations      []struct {
		Disposition string `json:"disposition"`
		Failure     string `json:"failure"`
		Records     uint64 `json:"records"`
	} `json:"worker_generations"`
}

// readQualification parses the capture's status document.
//
// The document is written either bare or wrapped in a single "status" member,
// so both shapes are accepted. A capture with no status document yields a zero
// qualification, which Qualify reports as unqualifiable rather than complete.
func readQualification(files map[string][]byte) (CaptureQualification, error) {
	raw, ok := files[captureStatusFile]
	if !ok {
		return CaptureQualification{}, nil
	}

	var envelope struct {
		Status json.RawMessage `json:"status"`
	}
	body := raw
	if err := json.Unmarshal(raw, &envelope); err == nil && len(envelope.Status) != 0 {
		body = envelope.Status
	}

	var st captureStatus
	if err := json.Unmarshal(body, &st); err != nil {
		return CaptureQualification{}, fmt.Errorf("gpuprof: parse %s: %w", captureStatusFile, err)
	}

	q := CaptureQualification{
		Present:            true,
		RunCompleteness:    st.RunCompleteness,
		CaptureSuccess:     st.CaptureSuccess,
		TransportReady:     st.TransportReady,
		CollectorFinalized: st.CollectorFinalized,
		WorkerGenerations:  len(st.WorkerGenerations),
		ActivityRecords:    st.ActivityRecords,
		DroppedRecords:     st.DroppedRecords,
		Outcome:            st.Outcome,
	}

	// A generation qualifies only by finalizing without a reported failure.
	// Zero retained records is not a fault: a finalized generation that ran no
	// GPU work is complete, and nothing here infers lost work from a count.
	for _, w := range st.WorkerGenerations {
		q.RetainedRecords += w.Records
		if w.Disposition != "finalized" || w.Failure != "" {
			q.UnqualifiedGenerations++
		}
	}

	// A published verdict is authoritative. Deriving one only fills in for a
	// capture written before whole-run completeness existed, and must never
	// override what the writer said.
	if q.RunCompleteness == "" {
		q.Derived = true
		q.RunCompleteness = runCompletenessOf(q.TransportReady, q.CollectorFinalized, q.UnqualifiedGenerations)
	} else {
		q.UnqualifiedGenerations = int(st.UnqualifiedGenerations)
	}
	return q, nil
}

// runCompletenessOf is the classification rule cudaprof's controller uses, so a
// verdict derived here agrees with one the writer would have published.
func runCompletenessOf(transportReady, collectorFinalized bool, unqualified int) string {
	switch {
	case !transportReady:
		return runUnknown
	case !collectorFinalized:
		return runFailed
	case unqualified > 0:
		return runPartial
	}
	return runComplete
}
