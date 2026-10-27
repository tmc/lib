package okcprofile

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Report is one parsed kernel table.
type Report struct {
	Rows []Row // in table order, which is descending time

	// Total is the sum of every kernel's time, hidden groups included.
	Total time.Duration

	// Busy is the time the GPU had any kernel running, printed only when
	// kernels overlapped; zero otherwise. Busy < Total means the rows
	// double-count waits.
	Busy time.Duration

	// Overlap reports that the judge flagged overlapping kernels.
	Overlap bool

	// Hidden is the number of kernel groups behind "... N more".
	Hidden int
}

// Row is one kernel group.
type Row struct {
	Kernel    string        // name as printed, without a trailing "..."
	Truncated bool          // the printed name was cut short
	Time      time.Duration // GPU time, rounded by the judge
	Share     float64       // percent of Total, as printed
	Calls     int
}

// Launches returns the number of launches in visible rows and whether that is
// all of them.
func (r Report) Launches() (n int, exact bool) {
	for _, row := range r.Rows {
		n += row.Calls
	}
	return n, r.Hidden == 0
}

var (
	rowRE  = regexp.MustCompile(`^(.+?)\s+(\d{1,12}(?:\.\d{1,9})?)\s+(ns|µs|us|ms|s)\s+(\d{1,3}(?:\.\d{1,9})?)%\s+(\d{1,12})$`)
	timeRE = regexp.MustCompile(`^(total|GPU busy.*?)\s+(\d{1,12}(?:\.\d{1,9})?)\s+(ns|µs|us|ms|s)$`)
	moreRE = regexp.MustCompile(`^\.\.\. (\d{1,12}) more$`)

	unit = map[string]float64{"ns": 1, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}
)

const overlapNote = "an overlapping kernel's time includes"

// ParseReport parses a judge kernel table.
func ParseReport(text string) (Report, error) {
	var r Report
	var haveTotal, haveBusy bool
	for i, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "kernel ") {
			continue
		}
		if m := rowRE.FindStringSubmatch(line); m != nil {
			d, err := duration(m[2], m[3])
			if err != nil {
				return Report{}, fmt.Errorf("line %d: %w", i+1, err)
			}
			share, _ := strconv.ParseFloat(m[4], 64)
			calls, _ := strconv.Atoi(m[5])
			name, trunc := strings.CutSuffix(m[1], "...")
			r.Rows = append(r.Rows, Row{Kernel: name, Truncated: trunc, Time: d, Share: share, Calls: calls})
			continue
		}
		if m := timeRE.FindStringSubmatch(line); m != nil {
			d, err := duration(m[2], m[3])
			if err != nil {
				return Report{}, fmt.Errorf("line %d: %w", i+1, err)
			}
			if m[1] == "total" {
				if haveTotal {
					return Report{}, fmt.Errorf("line %d: duplicate total", i+1)
				}
				r.Total, haveTotal = d, true
			} else {
				if haveBusy {
					return Report{}, fmt.Errorf("line %d: duplicate GPU busy", i+1)
				}
				r.Busy, haveBusy, r.Overlap = d, true, true
			}
			continue
		}
		if m := moreRE.FindStringSubmatch(line); m != nil {
			r.Hidden, _ = strconv.Atoi(m[1])
			continue
		}
		if strings.HasPrefix(line, overlapNote) {
			r.Overlap = true
			continue
		}
		return Report{}, fmt.Errorf("line %d: unrecognized report line %q", i+1, line)
	}
	if !haveTotal || len(r.Rows) == 0 {
		return Report{}, errors.New("report needs a total and kernel rows")
	}
	return r, nil
}

func duration(v, u string) (time.Duration, error) {
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return 0, err
	}
	return time.Duration(math.Round(f * unit[u])), nil
}

// Kernel is one row of the per-load kernel table built by [Kernels].
type Kernel struct {
	Load      string
	Kernel    string
	Truncated bool

	// Cases is the number of profiled cases in the load whose table shows
	// the kernel; Calls is its visible launch count summed over them.
	Cases int
	Calls int

	// Weighted is the case-weighted mean time, Σ weight·time / Σ weight,
	// over every profiled case in the load, counting zero where the kernel
	// is absent or hidden. Share is Weighted as a fraction of the load's
	// weighted Total.
	Weighted time.Duration
	Share    float64

	// Merged reports that some table printed the name on more than one row:
	// truncation made different kernels look alike, and their times were
	// added.
	Merged bool
}

// Table is the per-kernel table for a profile.
type Table struct {
	Kernels []Kernel // grouped by load in first-appearance order, then by Weighted descending

	// Hidden is the number of kernel groups hidden across all reports. When
	// it is nonzero, Calls are lower bounds and Shares do not sum to 1.
	Hidden int

	// Overlap reports that some table flagged overlapping kernels, so
	// Shares can sum to more than 1.
	Overlap bool
}

// Kernels parses every profile report in f and sums them into one weighted
// table per load.
func Kernels(f *File) (Table, error) {
	type key struct{ load, kernel string }
	var (
		t       Table
		loads   []string
		weights = map[string]float64{}
		totals  = map[string]float64{}
		sums    = map[key]float64{}
		rows    = map[key]*Kernel{}
	)
	for i, p := range f.Profiles {
		rep, err := ParseReport(p.Report)
		if err != nil {
			return Table{}, fmt.Errorf("profile %d: %w", i, err)
		}
		w := p.Case.Weight
		if !(w > 0) || math.IsInf(w, 0) {
			return Table{}, fmt.Errorf("profile %d: weight %v, want finite and positive", i, w)
		}
		load := p.Case.Load
		if _, ok := weights[load]; !ok {
			loads = append(loads, load)
		}
		weights[load] += w
		totals[load] += w * float64(rep.Total)
		t.Hidden += rep.Hidden
		t.Overlap = t.Overlap || rep.Overlap
		seen := map[string]bool{}
		for _, row := range rep.Rows {
			k := key{load, row.Kernel}
			kr := rows[k]
			if kr == nil {
				kr = &Kernel{Load: load, Kernel: row.Kernel}
				rows[k] = kr
			}
			kr.Truncated = kr.Truncated || row.Truncated
			kr.Calls += row.Calls
			if seen[row.Kernel] {
				kr.Merged = true
			} else {
				kr.Cases++
			}
			seen[row.Kernel] = true
			sums[k] += w * float64(row.Time)
		}
	}
	order := map[string]int{}
	for i, l := range loads {
		order[l] = i
	}
	for k, kr := range rows {
		kr.Weighted = time.Duration(math.Round(sums[k] / weights[k.load]))
		kr.Share = sums[k] / totals[k.load]
		t.Kernels = append(t.Kernels, *kr)
	}
	sort.Slice(t.Kernels, func(i, j int) bool {
		a, b := t.Kernels[i], t.Kernels[j]
		if a.Load != b.Load {
			return order[a.Load] < order[b.Load]
		}
		if a.Weighted != b.Weighted {
			return a.Weighted > b.Weighted
		}
		return a.Kernel < b.Kernel
	})
	return t, nil
}
