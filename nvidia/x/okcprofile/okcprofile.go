package okcprofile

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

// File is a judge profile. Fields the package does not use are not decoded.
type File struct {
	Problem    string      `json:"problem"`
	JobID      string      `json:"job_id"`
	Summary    Summary     `json:"summary"`
	Benchmarks []Benchmark `json:"benchmarks"`
	Profiles   []Profile   `json:"profiles"`
}

// Summary is the judge's verdict on the whole submission.
type Summary struct {
	Speedup   float64    `json:"speedup"`
	SpeedupCI [2]float64 `json:"speedup_ci"`
	Passed    bool       `json:"passed"`
}

// Case identifies one benchmark shape. Load names the group a score is taken
// over, and Weight the shape's share of it. Shape parameters vary by problem
// and are kept undecoded in Params.
type Case struct {
	Load   string
	Weight float64
	Params map[string]json.RawMessage
}

// UnmarshalJSON decodes load and weight and keeps every field in Params.
func (c *Case) UnmarshalJSON(b []byte) error {
	var known struct {
		Load   string  `json:"load"`
		Weight float64 `json:"weight"`
	}
	if err := json.Unmarshal(b, &known); err != nil {
		return err
	}
	var params map[string]json.RawMessage
	if err := json.Unmarshal(b, &params); err != nil {
		return err
	}
	*c = Case{Load: known.Load, Weight: known.Weight, Params: params}
	return nil
}

// Benchmark is one timed shape: baseline and candidate samples in the order
// they ran, and the replay overhead the judge subtracts from both.
type Benchmark struct {
	Case           Case      `json:"case"`
	BaselineRunsNS []float64 `json:"baseline_runs_ns"`
	RunsNS         []float64 `json:"runs_ns"`
	ReplayOverhead struct {
		OverheadNS float64 `json:"overhead_ns"`
	} `json:"replay_overhead"`
}

// Profile is one profiled case and its kernel table, unparsed.
type Profile struct {
	Case   Case   `json:"case"`
	Event  string `json:"event"`
	Index  int    `json:"index"`
	Report string `json:"report"`
}

// Decode reads a judge profile.
func Decode(r io.Reader) (*File, error) {
	var f File
	if err := json.NewDecoder(r).Decode(&f); err != nil {
		return nil, fmt.Errorf("decode judge profile: %w", err)
	}
	return &f, nil
}

// Series returns one series per benchmark, named by its index. Samples r and
// r+1 of each side form the block A1=baseline[r], B1=candidate[r],
// B2=candidate[r+1], A2=baseline[r+1].
func (f *File) Series() ([]pairedtiming.Series, error) {
	var out []pairedtiming.Series
	for i, b := range f.Benchmarks {
		a, c := b.BaselineRunsNS, b.RunsNS
		if len(a) == 0 || len(a) != len(c) || len(a)%2 != 0 {
			return nil, fmt.Errorf("benchmark %d: %d baseline and %d candidate samples, want equal even counts", i, len(a), len(c))
		}
		s := pairedtiming.Series{
			Name:      fmt.Sprint(i),
			Load:      b.Case.Load,
			Weight:    b.Case.Weight,
			Intercept: b.ReplayOverhead.OverheadNS,
		}
		for r := 0; r < len(a); r += 2 {
			s.Blocks = append(s.Blocks, pairedtiming.Block{A1: a[r], B1: c[r], B2: c[r+1], A2: a[r+1]})
		}
		out = append(out, s)
	}
	return out, nil
}
