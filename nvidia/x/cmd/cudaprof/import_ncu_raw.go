package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/nsightcompute"
)

// ncuRawLaunch is one launch of a raw-page export with the rates derived
// from it. A rate is absent when a metric it needs is absent, not a number,
// or in a unit this command does not convert.
type ncuRawLaunch struct {
	ID           string   `json:"id"`
	Kernel       string   `json:"kernel"`
	Grid         string   `json:"grid"`
	Block        string   `json:"block"`
	DurationNS   *float64 `json:"duration_ns,omitempty"`
	DRAMBytes    *float64 `json:"dram_bytes,omitempty"`
	DRAMGBps     *float64 `json:"dram_gbps,omitempty"`
	CeilingFrac  *float64 `json:"ceiling_fraction,omitempty"`
	WarpsActive  *float64 `json:"warps_active_pct,omitempty"`
	CTAsLaunched *float64 `json:"ctas_launched,omitempty"`
}

var ncuDurationNS = map[string]float64{
	"ns": 1, "nsecond": 1, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "s": 1e9, "second": 1e9,
}

// ncuRawLaunches derives per-launch rates. DRAM bytes are read plus write;
// GB/s is decimal. Occupancy is taken only from a "%" unit, and a CTA count
// only from "block" or "cta", because the same metric name in another unit
// is a different quantity.
func ncuRawLaunches(t *nsightcompute.RawTable, ceilingGBps float64) []ncuRawLaunch {
	value := func(l *nsightcompute.RawLaunch, metric string, units ...string) *float64 {
		v, unit, err := t.Value(l, metric)
		if err != nil {
			return nil
		}
		for _, u := range units {
			if u == unit {
				return &v
			}
		}
		return nil
	}
	var out []ncuRawLaunch
	for i := range t.Launches {
		l := &t.Launches[i]
		r := ncuRawLaunch{ID: l.ID, Kernel: l.Kernel, Grid: l.Grid, Block: l.Block}
		if v, unit, err := t.Value(l, "gpu__time_duration.sum"); err == nil {
			if scale, ok := ncuDurationNS[unit]; ok && v > 0 {
				ns := v * scale
				r.DurationNS = &ns
			}
		}
		rd := value(l, "dram__bytes_read.sum", "byte")
		wr := value(l, "dram__bytes_write.sum", "byte")
		if rd != nil && wr != nil {
			b := *rd + *wr
			r.DRAMBytes = &b
			if r.DurationNS != nil {
				gbps := b / *r.DurationNS
				r.DRAMGBps = &gbps
				if ceilingGBps > 0 {
					f := gbps / ceilingGBps
					r.CeilingFrac = &f
				}
			}
		}
		if v := value(l, "sm__warps_active.avg.pct_of_peak_sustained_active", "%"); v != nil && *v >= 0 && *v <= 100 {
			r.WarpsActive = v
		}
		r.CTAsLaunched = value(l, "sm__ctas_launched.sum", "block", "cta")
		out = append(out, r)
	}
	return out
}

func runImportNCURaw(data []byte, maxBytes int, ceilingGBps float64, jsonOutput bool, stdout io.Writer) error {
	t, err := nsightcompute.ReadRaw(bytes.NewReader(data), maxBytes)
	if err != nil {
		return fmt.Errorf("import-ncu: %w", err)
	}
	launches := ncuRawLaunches(t, ceilingGBps)
	if jsonOutput {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(struct {
			Metrics  int            `json:"metrics"`
			Ceiling  float64        `json:"ceiling_gbps,omitempty"`
			Launches []ncuRawLaunch `json:"launches"`
		}{len(t.Metrics), ceilingGBps, launches})
	}
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "id\tduration ns\tDRAM bytes\tGB/s\tof ceiling\twarps active %\tCTAs\tkernel")
	f := func(p *float64, format string) string {
		if p == nil {
			return "-"
		}
		return fmt.Sprintf(format, *p)
	}
	for _, r := range launches {
		k := r.Kernel
		if len(k) > 60 {
			k = k[:57] + "..."
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", r.ID, f(r.DurationNS, "%.0f"), f(r.DRAMBytes, "%.0f"),
			f(r.DRAMGBps, "%.1f"), f(r.CeilingFrac, "%.3f"), f(r.WarpsActive, "%.2f"), f(r.CTAsLaunched, "%.0f"), k)
	}
	return tw.Flush()
}
