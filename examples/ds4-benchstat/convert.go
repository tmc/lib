package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// A row is one ds4-bench CSV record: one context length, measured once.
// Fields absent from the CSV header stay zero and their has* flag stays false.
type row struct {
	ctxTokens     int
	prefillTokens int
	prefillTPS    float64
	genTokens     int
	genTPS        float64

	genFirstMS      float64
	genSteadyTokens int
	genSteadyTPS    float64
	hasFirst        bool
	hasSteady       bool

	kvcacheBytes uint64
}

// config is one benchstat configuration line, "key: value". Order is preserved
// so a converted file reads the way the receipt did.
type config struct {
	key   string
	value string
}

// parseCSV reads ds4-bench CSV. Columns are matched by header name, because the
// oracle's published speed-bench files predate three of the columns the current
// ds4_bench.c writes.
func parseCSV(r io.Reader) ([]row, error) {
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = -1
	recs, err := cr.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read csv: %w", err)
	}
	if len(recs) == 0 {
		return nil, fmt.Errorf("empty csv")
	}
	index := make(map[string]int, len(recs[0]))
	for i, name := range recs[0] {
		// The oracle's m4_max.csv and its siblings begin with a UTF-8 BOM,
		// which would otherwise hide the ctx_tokens column.
		index[strings.TrimPrefix(strings.TrimSpace(name), "\ufeff")] = i
	}
	if _, ok := index["ctx_tokens"]; !ok {
		return nil, fmt.Errorf("csv has no ctx_tokens column; got %q", strings.Join(recs[0], ","))
	}
	var rows []row
	for n, rec := range recs[1:] {
		fields := make(map[string]string, len(index))
		for name, i := range index {
			if i < len(rec) {
				fields[name] = strings.TrimSpace(rec[i])
			}
		}
		got, err := rowFromFields(fields)
		if err != nil {
			return nil, fmt.Errorf("csv line %d: %w", n+2, err)
		}
		rows = append(rows, got)
	}
	return rows, nil
}

// rowFromFields builds a row from named CSV fields. It is shared with the
// receipt reader, whose csv object has exactly these names.
func rowFromFields(f map[string]string) (row, error) {
	var r row
	var err error
	req := func(name string, dst *int) {
		if err != nil {
			return
		}
		s, ok := f[name]
		if !ok {
			err = fmt.Errorf("missing %s", name)
			return
		}
		var v int
		if v, err = strconv.Atoi(s); err != nil {
			err = fmt.Errorf("%s: %w", name, err)
			return
		}
		*dst = v
	}
	reqFloat := func(name string, dst *float64) {
		if err != nil {
			return
		}
		s, ok := f[name]
		if !ok {
			err = fmt.Errorf("missing %s", name)
			return
		}
		var v float64
		if v, err = strconv.ParseFloat(s, 64); err != nil {
			err = fmt.Errorf("%s: %w", name, err)
			return
		}
		*dst = v
	}
	req("ctx_tokens", &r.ctxTokens)
	req("prefill_tokens", &r.prefillTokens)
	reqFloat("prefill_tps", &r.prefillTPS)
	req("gen_tokens", &r.genTokens)
	reqFloat("gen_tps", &r.genTPS)
	if err != nil {
		return row{}, err
	}
	if s, ok := f["gen_first_ms"]; ok && s != "" {
		v, e := strconv.ParseFloat(s, 64)
		if e != nil {
			return row{}, fmt.Errorf("gen_first_ms: %w", e)
		}
		r.genFirstMS, r.hasFirst = v, true
	}
	if s, ok := f["gen_steady_tps"]; ok && s != "" {
		v, e := strconv.ParseFloat(s, 64)
		if e != nil {
			return row{}, fmt.Errorf("gen_steady_tps: %w", e)
		}
		r.genSteadyTPS, r.hasSteady = v, true
		if t, ok := f["gen_steady_tokens"]; ok && t != "" {
			n, e := strconv.Atoi(t)
			if e != nil {
				return row{}, fmt.Errorf("gen_steady_tokens: %w", e)
			}
			r.genSteadyTokens = n
		}
	}
	if s, ok := f["kvcache_bytes"]; ok && s != "" {
		v, e := strconv.ParseUint(s, 10, 64)
		if e != nil {
			return row{}, fmt.Errorf("kvcache_bytes: %w", e)
		}
		r.kvcacheBytes = v
	}
	return r, nil
}

// receipt is the subset of an oracle RESULT.json this command reads.
type receipt struct {
	Result  string            `json:"result"`
	Backend string            `json:"backend"`
	Host    string            `json:"host"`
	Oracle  string            `json:"oracle"`
	CSV     map[string]string `json:"csv"`
	Model   string            `json:"model_sha256"`
	Source  string            `json:"ds4_c_sha256"`
	Binary  string            `json:"binary_sha256"`
}

// parseReceipt reads one qualified oracle run receipt. A receipt that did not
// qualify is refused: its rate is not admissible, and converting it would put an
// inadmissible number into a comparison that no longer shows where it came from.
func parseReceipt(data []byte) ([]config, []row, error) {
	var rc receipt
	if err := json.Unmarshal(data, &rc); err != nil {
		return nil, nil, fmt.Errorf("parse receipt: %w", err)
	}
	if !strings.HasPrefix(rc.Result, "QUALIFIED") {
		return nil, nil, fmt.Errorf("receipt result is %q, not qualified; refusing to convert it into a rate", rc.Result)
	}
	if len(rc.CSV) == 0 {
		return nil, nil, fmt.Errorf("receipt has no csv object")
	}
	r, err := rowFromFields(rc.CSV)
	if err != nil {
		return nil, nil, fmt.Errorf("receipt csv: %w", err)
	}
	value := map[string]string{
		"result":        rc.Result,
		"backend":       rc.Backend,
		"host":          rc.Host,
		"oracle":        rc.Oracle,
		"model_sha256":  rc.Model,
		"ds4_c_sha256":  rc.Source,
		"binary_sha256": rc.Binary,
	}
	cfg := []config{{"pkg", "ds4.c"}}
	for _, k := range provenanceKeys {
		if v := value[k]; v != "" {
			cfg = append(cfg, config{k, v})
		}
	}
	return cfg, []row{r}, nil
}

// provenanceKeys are the configuration keys that identify a run rather than name
// a comparison dimension, in the order they are written. benchstat groups by
// configuration, so leaving these in place puts each file in a table of its own;
// ignoreFlag folds the tables back together while the keys stay in the file for
// whoever reads it later.
var provenanceKeys = []string{
	"result", "backend", "host", "oracle",
	"model_sha256", "ds4_c_sha256", "binary_sha256",
}

// ignoreFlag is the benchstat flag that makes converted files comparable to each
// other. It is derived from provenanceKeys so it cannot drift from what is
// written.
func ignoreFlag() string { return "-ignore=" + strings.Join(provenanceKeys, ",") }

// units declared for the whole file. benchstat assumes lower-is-better, which is
// wrong for a rate, so tok/s says otherwise explicitly.
var units = []string{
	"Unit sec/op better=lower",
	"Unit tok/s better=higher",
	"Unit sec/first-token better=lower",
	"Unit B/op better=lower",
}

// write emits rows in Go benchmark format. It returns the number of benchmark
// lines written and the number of measurements skipped as unreported.
func write(w io.Writer, cfg []config, rows []row) (written, skipped int, err error) {
	bw := &errWriter{w: w}
	for _, c := range cfg {
		bw.printf("%s: %s\n", c.key, c.value)
	}
	for _, u := range units {
		bw.printf("%s\n", u)
	}
	sorted := append([]row(nil), rows...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].ctxTokens < sorted[j].ctxTokens })
	for _, r := range sorted {
		name := func(bench string) string {
			return fmt.Sprintf("BenchmarkDS4%s/ctx=%d", bench, r.ctxTokens)
		}
		// A rate of zero is ds4-bench's sentinel for a phase it did not
		// measure, not a measurement of zero, so it yields no line.
		if r.prefillTPS > 0 {
			bw.printf("%s %d %s sec/op %s tok/s\n",
				name("Prefill"), r.prefillTokens, num(1/r.prefillTPS), num(r.prefillTPS))
			written++
		} else {
			skipped++
		}
		if r.genTPS > 0 {
			line := fmt.Sprintf("%s %d %s sec/op %s tok/s",
				name("Generate"), r.genTokens, num(1/r.genTPS), num(r.genTPS))
			if r.hasFirst && r.genFirstMS > 0 {
				line += fmt.Sprintf(" %s sec/first-token", num(r.genFirstMS/1000))
			}
			bw.printf("%s\n", line)
			written++
		} else {
			skipped++
		}
		if r.hasSteady && r.genSteadyTPS > 0 && r.genSteadyTokens > 0 {
			bw.printf("%s %d %s sec/op %s tok/s\n",
				name("GenerateSteady"), r.genSteadyTokens, num(1/r.genSteadyTPS), num(r.genSteadyTPS))
			written++
		}
		// kvcache_bytes is 0 in receipts whose own logs report a cache of
		// gigabytes, so 0 means unreported and must not be published as a size.
		if r.kvcacheBytes > 0 {
			bw.printf("%s 1 %d B/op\n", name("KVCache"), r.kvcacheBytes)
			written++
		} else {
			skipped++
		}
	}
	return written, skipped, bw.err
}

// num formats a value with enough digits to reconstruct the rate it came from.
func num(v float64) string { return strconv.FormatFloat(v, 'g', 9, 64) }

type errWriter struct {
	w   io.Writer
	err error
}

func (e *errWriter) printf(format string, args ...any) {
	if e.err != nil {
		return
	}
	_, e.err = fmt.Fprintf(e.w, format, args...)
}
