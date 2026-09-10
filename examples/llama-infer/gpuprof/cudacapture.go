package gpuprof

import (
	"errors"
	"fmt"
	"os"
	"sort"

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
}

// ReadCaptureSummary reads the cudaprof capture at path and summarizes its
// retained activity. Path may be either a capture file or a capture directory.
//
// Reading uses the cudacapture package defaults, which bound how much of an
// untrusted artifact is held in memory.
func ReadCaptureSummary(path string) (*CaptureSummary, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("gpuprof: stat capture: %w", err)
	}

	var c *cudacapture.Capture
	if info.IsDir() {
		c, err = cudacapture.ReadDir(path)
	} else {
		var f *os.File
		f, err = os.Open(path)
		if err != nil {
			return nil, fmt.Errorf("gpuprof: open capture: %w", err)
		}
		defer f.Close()
		c, err = cudacapture.Read(f)
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
