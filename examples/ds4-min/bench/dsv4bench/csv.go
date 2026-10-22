// Package dsv4bench defines the portable ds4-bench CSV result schema.
package dsv4bench

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"strconv"
)

var header = []string{
	"ctx_tokens", "prefill_tokens", "prefill_tps", "gen_tokens", "gen_tps",
	"gen_first_ms", "gen_steady_tokens", "gen_steady_tps", "kvcache_bytes",
	"context_total_bytes", "context_raw_bytes", "context_compressed_bytes",
	"context_scratch_bytes", "peak_rss_bytes",
}

// Row is one frontier measurement. Counts and byte sizes must be
// non-negative. Rates and latency must be finite and non-negative.
type Row struct {
	CtxTokens              int64
	PrefillTokens          int64
	PrefillTPS             float64
	GenTokens              int64
	GenTPS                 float64
	GenFirstMS             float64
	GenSteadyTokens        int64
	GenSteadyTPS           float64
	KVCacheBytes           int64
	ContextTotalBytes      int64
	ContextRawBytes        int64
	ContextCompressedBytes int64
	ContextScratchBytes    int64
	PeakRSSBytes           int64
}

// WriteCSV writes the canonical 14-column ds4-bench CSV, including its
// header. It does not append to an existing document: callers that combine
// runs should write each run through a separate writer or add rows directly.
func WriteCSV(w io.Writer, rows []Row) error {
	if w == nil {
		return fmt.Errorf("write csv: nil writer")
	}
	c := csv.NewWriter(w)
	if err := c.Write(header); err != nil {
		return fmt.Errorf("write csv header: %w", err)
	}
	for i, row := range rows {
		if err := validate(row); err != nil {
			return fmt.Errorf("write csv row %d: %w", i, err)
		}
		if err := c.Write(rowFields(row)); err != nil {
			return fmt.Errorf("write csv row %d: %w", i, err)
		}
	}
	c.Flush()
	if err := c.Error(); err != nil {
		return fmt.Errorf("flush csv: %w", err)
	}
	return nil
}

// ReadCSV reads the canonical 14-column ds4-bench CSV. The header must match
// exactly and every data row is validated before it is returned.
func ReadCSV(r io.Reader) ([]Row, error) {
	if r == nil {
		return nil, fmt.Errorf("read csv: nil reader")
	}
	c := csv.NewReader(r)
	c.FieldsPerRecord = len(header)
	record, err := c.Read()
	if err == io.EOF {
		return nil, fmt.Errorf("read csv: missing header")
	}
	if err != nil {
		return nil, fmt.Errorf("read csv header: %w", err)
	}
	if !sameFields(record, header) {
		return nil, fmt.Errorf("read csv: unexpected header")
	}
	rows := make([]Row, 0)
	for n := 2; ; n++ {
		record, err := c.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read csv row %d: %w", n, err)
		}
		row, err := parseRow(record)
		if err != nil {
			return nil, fmt.Errorf("read csv row %d: %w", n, err)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// ValidateFrontier checks the ordering required by a context-frontier sweep.
// It deliberately does not sort rows: an unordered or duplicate measurement
// is evidence of a bad input, not a formatting problem to hide.
func ValidateFrontier(rows []Row) error {
	if len(rows) == 0 {
		return fmt.Errorf("frontier is empty")
	}
	var previous int64
	for i, row := range rows {
		if row.CtxTokens <= 0 {
			return fmt.Errorf("frontier row %d: ctx_tokens must be positive", i)
		}
		if i > 0 && row.CtxTokens <= previous {
			return fmt.Errorf("frontier row %d: ctx_tokens is not strictly increasing", i)
		}
		previous = row.CtxTokens
	}
	return nil
}

func sameFields(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func parseRow(fields []string) (Row, error) {
	if len(fields) != len(header) {
		return Row{}, fmt.Errorf("want %d columns, got %d", len(header), len(fields))
	}
	var row Row
	counts := []*int64{
		&row.CtxTokens, &row.PrefillTokens, nil, &row.GenTokens, nil, nil,
		&row.GenSteadyTokens, nil, &row.KVCacheBytes, &row.ContextTotalBytes,
		&row.ContextRawBytes, &row.ContextCompressedBytes, &row.ContextScratchBytes,
		&row.PeakRSSBytes,
	}
	rates := []*float64{nil, nil, &row.PrefillTPS, nil, &row.GenTPS, &row.GenFirstMS, nil, &row.GenSteadyTPS, nil, nil, nil, nil, nil, nil}
	for i, dst := range counts {
		if dst == nil {
			continue
		}
		value, err := strconv.ParseInt(fields[i], 10, 64)
		if err != nil {
			return Row{}, fmt.Errorf("%s: invalid integer %q", header[i], fields[i])
		}
		*dst = value
	}
	for i, dst := range rates {
		if dst == nil {
			continue
		}
		value, err := strconv.ParseFloat(fields[i], 64)
		if err != nil {
			return Row{}, fmt.Errorf("%s: invalid number %q", header[i], fields[i])
		}
		*dst = value
	}
	if err := validate(row); err != nil {
		return Row{}, err
	}
	return row, nil
}

func validate(r Row) error {
	counts := []struct {
		name  string
		value int64
	}{
		{"ctx_tokens", r.CtxTokens}, {"prefill_tokens", r.PrefillTokens}, {"gen_tokens", r.GenTokens}, {"gen_steady_tokens", r.GenSteadyTokens},
		{"kvcache_bytes", r.KVCacheBytes}, {"context_total_bytes", r.ContextTotalBytes}, {"context_raw_bytes", r.ContextRawBytes}, {"context_compressed_bytes", r.ContextCompressedBytes}, {"context_scratch_bytes", r.ContextScratchBytes}, {"peak_rss_bytes", r.PeakRSSBytes},
	}
	for _, f := range counts {
		if f.value < 0 {
			return fmt.Errorf("%s is negative", f.name)
		}
	}
	rates := []struct {
		name  string
		value float64
	}{{"prefill_tps", r.PrefillTPS}, {"gen_tps", r.GenTPS}, {"gen_first_ms", r.GenFirstMS}, {"gen_steady_tps", r.GenSteadyTPS}}
	for _, f := range rates {
		if math.IsNaN(f.value) || math.IsInf(f.value, 0) || f.value < 0 {
			return fmt.Errorf("%s is not a finite non-negative number", f.name)
		}
	}
	return nil
}

func rowFields(r Row) []string {
	return []string{
		strconv.FormatInt(r.CtxTokens, 10), strconv.FormatInt(r.PrefillTokens, 10), strconv.FormatFloat(r.PrefillTPS, 'g', -1, 64),
		strconv.FormatInt(r.GenTokens, 10), strconv.FormatFloat(r.GenTPS, 'g', -1, 64), strconv.FormatFloat(r.GenFirstMS, 'g', -1, 64),
		strconv.FormatInt(r.GenSteadyTokens, 10), strconv.FormatFloat(r.GenSteadyTPS, 'g', -1, 64), strconv.FormatInt(r.KVCacheBytes, 10),
		strconv.FormatInt(r.ContextTotalBytes, 10), strconv.FormatInt(r.ContextRawBytes, 10), strconv.FormatInt(r.ContextCompressedBytes, 10), strconv.FormatInt(r.ContextScratchBytes, 10), strconv.FormatInt(r.PeakRSSBytes, 10),
	}
}
