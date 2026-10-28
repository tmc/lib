package nsightsystems

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// A Span is one interval on a timeline: a kernel execution on the device
// or a CUDA API call on the host. Stream is the CUDA stream of a kernel and
// -1 when unknown.
type Span struct {
	Name           string
	StartNS, EndNS int64
	Stream         int64
}

// KernelSpans returns the kernel executions in r in start order.
func KernelSpans(r *Report) ([]Span, error) {
	var out []Span
	for _, e := range r.Events {
		if e.Table != "CUPTI_ACTIVITY_KIND_KERNEL" {
			continue
		}
		if e.EndNS == nil {
			return nil, errors.New("nsys: kernel without end")
		}
		var raw struct {
			StreamID int64 `json:"streamId"`
		}
		if err := json.Unmarshal(e.Raw, &raw); err != nil {
			return nil, fmt.Errorf("nsys: kernel row: %w", err)
		}
		out = append(out, Span{Name: e.Name, StartNS: e.StartNS, EndNS: *e.EndNS, Stream: raw.StreamID})
	}
	sortSpans(out)
	return out, nil
}

// ReadGPUTraceCSV reads the kernel rows of the CSV that
//
//	nsys stats --report cuda_gpu_trace --format csv
//
// writes, in start order. Memory copies and sets, which have no grid, are
// skipped.
func ReadGPUTraceCSV(r io.Reader) ([]Span, error) {
	return readTraceCSV(r, func(row map[string]string) bool { return row["GrdX"] != "" }, "Strm")
}

// ReadAPITraceCSV reads the CSV that
//
//	nsys stats --report cuda_api_trace --format csv
//
// writes, in start order.
func ReadAPITraceCSV(r io.Reader) ([]Span, error) {
	return readTraceCSV(r, func(map[string]string) bool { return true }, "")
}

func readTraceCSV(r io.Reader, keep func(map[string]string) bool, streamCol string) ([]Span, error) {
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = -1
	var head []string
	for head == nil {
		rec, err := cr.Read()
		if err == io.EOF {
			return nil, errors.New("nsys csv: no header")
		}
		if err != nil {
			return nil, fmt.Errorf("nsys csv: %w", err)
		}
		// nsys may print notices before the header.
		if slices.Contains(rec, "Start (ns)") {
			head = rec
		}
	}
	col := make(map[string]int)
	for i, h := range head {
		col[h] = i
	}
	for _, h := range []string{"Start (ns)", "Duration (ns)", "Name"} {
		if _, ok := col[h]; !ok {
			return nil, fmt.Errorf("nsys csv: no %q column", h)
		}
	}
	var out []Span
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("nsys csv: %w", err)
		}
		if len(rec) != len(head) {
			return nil, fmt.Errorf("nsys csv: row has %d fields, header %d", len(rec), len(head))
		}
		row := make(map[string]string, len(head))
		for i, h := range head {
			row[h] = rec[i]
		}
		if !keep(row) {
			continue
		}
		start, err := strconv.ParseInt(row["Start (ns)"], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("nsys csv: start: %w", err)
		}
		dur, err := strconv.ParseInt(row["Duration (ns)"], 10, 64)
		if err != nil || dur < 0 {
			return nil, fmt.Errorf("nsys csv: bad duration %q", row["Duration (ns)"])
		}
		s := Span{Name: row["Name"], StartNS: start, EndNS: start + dur, Stream: -1}
		if v := row[streamCol]; streamCol != "" && v != "" {
			if s.Stream, err = strconv.ParseInt(v, 10, 64); err != nil {
				return nil, fmt.Errorf("nsys csv: stream: %w", err)
			}
		}
		out = append(out, s)
	}
	sortSpans(out)
	return out, nil
}

func sortSpans(s []Span) {
	sort.SliceStable(s, func(i, j int) bool { return s[i].StartNS < s[j].StartNS })
}

// SplitSteps splits kernels, in start order, into steps that each end with a
// kernel whose name contains end. Kernels after the last such kernel are
// dropped: they belong to an unfinished step.
func SplitSteps(kernels []Span, end string) [][]Span {
	var steps [][]Span
	start := 0
	for i, k := range kernels {
		if strings.Contains(k.Name, end) {
			steps = append(steps, kernels[start:i+1])
			start = i + 1
		}
	}
	return steps
}

// LaunchGaps summarizes the device timeline of one step: a sequence of
// kernels that run as a dependent chain.
//
// SpanNS runs from the first kernel's start to the last kernel's end.
// BusyNS is the time covered by at least one kernel and IdleNS the rest of
// the span: the device-side cost of launching and retiring the step's
// kernels, which does not shrink when the kernels get faster. Under
// programmatic dependent launch a kernel may start before its predecessor
// ends; the overlap counts once in BusyNS. Such a kernel's duration includes
// its wait for the predecessor, so the launch latency it hides moves from
// IdleNS into BusyNS: compare SpanNS, not IdleNS, across launch modes.
type LaunchGaps struct {
	Launches int
	SpanNS   int64
	BusyNS   int64
	IdleNS   int64
	MaxGapNS int64

	// Kernels aggregates the step's kernels by name, in descending order of
	// the idle time before them.
	Kernels []KernelGaps
}

// KernelGaps aggregates the launches of one kernel within a step. GapNS is
// the idle time between the end of all earlier kernels and this kernel's
// start, summed over its launches.
type KernelGaps struct {
	Name     string
	Launches int
	BusyNS   int64
	GapNS    int64
}

// AnalyzeLaunchGaps summarizes step, a sequence of kernels in start order.
func AnalyzeLaunchGaps(step []Span) LaunchGaps {
	var g LaunchGaps
	if len(step) == 0 {
		return g
	}
	g.Launches = len(step)
	by := make(map[string]*KernelGaps)
	end := step[0].StartNS
	for _, k := range step {
		kg := by[k.Name]
		if kg == nil {
			kg = &KernelGaps{Name: k.Name}
			by[k.Name] = kg
		}
		kg.Launches++
		kg.BusyNS += k.EndNS - k.StartNS
		if gap := k.StartNS - end; gap > 0 {
			kg.GapNS += gap
			g.IdleNS += gap
			g.MaxGapNS = max(g.MaxGapNS, gap)
		}
		end = max(end, k.EndNS)
	}
	g.SpanNS = end - step[0].StartNS
	g.BusyNS = g.SpanNS - g.IdleNS
	for _, kg := range by {
		g.Kernels = append(g.Kernels, *kg)
	}
	sort.Slice(g.Kernels, func(i, j int) bool {
		a, b := g.Kernels[i], g.Kernels[j]
		if a.GapNS != b.GapNS {
			return a.GapNS > b.GapNS
		}
		return a.Name < b.Name
	})
	return g
}

// GapPerLaunchNS returns the mean idle time between consecutive launches.
func (g LaunchGaps) GapPerLaunchNS() float64 {
	if g.Launches < 2 {
		return 0
	}
	return float64(g.IdleNS) / float64(g.Launches-1)
}

// Project models the step on a device whose kernels together take floorNS,
// the step's bandwidth or compute floor, while the idle time between
// launches stays what it was here: floorNS + IdleNS. It is a model, not a
// measurement: launch and drain latency differ between devices, and a
// faster device can expose host submission cost that this trace hid.
func (g LaunchGaps) Project(floorNS int64) int64 {
	return floorNS + g.IdleNS
}

// ShortKernels returns the number and total duration of the kernels in
// step shorter than maxNS. Such kernels are bound by launch and drain
// latency, not bandwidth, and keep their duration on a faster device.
func ShortKernels(step []Span, maxNS int64) (n int, totalNS int64) {
	for _, k := range step {
		if d := k.EndNS - k.StartNS; d < maxNS {
			n++
			totalNS += d
		}
	}
	return n, totalNS
}

// StepGaps returns the idle time between consecutive steps: from the end of
// each step's last kernel to the start of the next step's first. It is the
// host's turnaround between steps (synchronization, copies, parameter
// updates and the next launch), which no kernel overlaps.
func StepGaps(steps [][]Span) []int64 {
	var gaps []int64
	for i := 1; i < len(steps); i++ {
		end := int64(0)
		for _, k := range steps[i-1] {
			end = max(end, k.EndNS)
		}
		if len(steps[i]) > 0 {
			gaps = append(gaps, steps[i][0].StartNS-end)
		}
	}
	return gaps
}

// APILaunches counts the host calls in api whose names contain one of
// names, and their total duration.
func APILaunches(api []Span, names ...string) (calls int, totalNS int64) {
	for _, s := range api {
		for _, n := range names {
			if strings.Contains(s.Name, n) {
				calls++
				totalNS += s.EndNS - s.StartNS
				break
			}
		}
	}
	return calls, totalNS
}
