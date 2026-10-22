package dsv4bench

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

// Range summarizes one measured value across benchmark rows.
type Range struct {
	Min    float64 `json:"min"`
	Median float64 `json:"median"`
	Max    float64 `json:"max"`
}

// Summary is the model-free summary of a strict 14-column benchmark CSV.
// The dimensions and memory columns remain available in the input rows; this
// summary deliberately reports only the performance measurements.
type Summary struct {
	Samples      int   `json:"samples"`
	PrefillTPS   Range `json:"prefill_tps"`
	GenTPS       Range `json:"gen_tps"`
	GenFirstMS   Range `json:"gen_first_ms"`
	GenSteadyTPS Range `json:"gen_steady_tps"`
}

// Summarize validates rows and computes a median, minimum, and maximum for
// each performance column. It does not reorder or mutate rows.
func Summarize(rows []Row) (Summary, error) {
	if len(rows) == 0 {
		return Summary{}, fmt.Errorf("summarize benchmark: no rows")
	}
	values := make([][]float64, 4)
	for i, row := range rows {
		if err := validate(row); err != nil {
			return Summary{}, fmt.Errorf("summarize benchmark row %d: %w", i, err)
		}
		values[0] = append(values[0], row.PrefillTPS)
		values[1] = append(values[1], row.GenTPS)
		values[2] = append(values[2], row.GenFirstMS)
		values[3] = append(values[3], row.GenSteadyTPS)
	}
	return Summary{
		Samples:      len(rows),
		PrefillTPS:   summarizeValues(values[0]),
		GenTPS:       summarizeValues(values[1]),
		GenFirstMS:   summarizeValues(values[2]),
		GenSteadyTPS: summarizeValues(values[3]),
	}, nil
}

// ReadSummary reads and strictly validates the canonical 14-column CSV,
// then returns its model-free performance summary.
func ReadSummary(r io.Reader) (Summary, error) {
	rows, err := ReadCSV(r)
	if err != nil {
		return Summary{}, err
	}
	return Summarize(rows)
}

// WriteSummaryJSON writes the stable JSON representation of a Summary.
func WriteSummaryJSON(w io.Writer, summary Summary) error {
	if w == nil {
		return fmt.Errorf("write summary: nil writer")
	}
	if err := validateSummary(summary); err != nil {
		return fmt.Errorf("write summary: %w", err)
	}
	enc := json.NewEncoder(w)
	if err := enc.Encode(summary); err != nil {
		return fmt.Errorf("write summary: %w", err)
	}
	return nil
}

func validateSummary(s Summary) error {
	if s.Samples <= 0 {
		return fmt.Errorf("samples must be positive")
	}
	for name, r := range map[string]Range{
		"prefill_tps":    s.PrefillTPS,
		"gen_tps":        s.GenTPS,
		"gen_first_ms":   s.GenFirstMS,
		"gen_steady_tps": s.GenSteadyTPS,
	} {
		if !finite(r.Min) || !finite(r.Median) || !finite(r.Max) {
			return fmt.Errorf("%s contains a non-finite value", name)
		}
		if r.Min < 0 || r.Median < 0 || r.Max < 0 {
			return fmt.Errorf("%s contains a negative value", name)
		}
		if r.Min > r.Median || r.Median > r.Max {
			return fmt.Errorf("%s is not ordered", name)
		}
	}
	return nil
}

func summarizeValues(values []float64) Range {
	ordered := append([]float64(nil), values...)
	sort.Float64s(ordered)
	median := ordered[len(ordered)/2]
	if len(ordered)%2 == 0 {
		median = (ordered[len(ordered)/2-1] + ordered[len(ordered)/2]) / 2
	}
	return Range{Min: ordered[0], Median: median, Max: ordered[len(ordered)-1]}
}
