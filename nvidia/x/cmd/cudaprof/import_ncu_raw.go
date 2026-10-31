package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/nsightcompute"
)

// ncuCeiling is the bandwidth the raw import divides by, and where it came
// from: a -ceiling number (Device empty) or a -device table lookup.
type ncuCeiling struct {
	GBps       float64 `json:"ceiling_gbps,omitempty"`
	Device     string  `json:"ceiling_device,omitempty"`
	Provenance string  `json:"ceiling_provenance,omitempty"`
}

// ncuDisplayName is the metric ncu records the device's product name in.
const ncuDisplayName = "device__attribute_display_name"

// resolveNCUCeiling returns the ceiling for the export. With device "auto"
// the product name comes from the export, which must name exactly one. With
// any other device name, every launch that records a name must agree with
// it: a ceiling measured on another part is the wrong denominator.
func resolveNCUCeiling(t *nsightcompute.RawTable, gbps float64, device string) (ncuCeiling, error) {
	if gbps < 0 || math.IsNaN(gbps) || math.IsInf(gbps, 0) {
		return ncuCeiling{}, fmt.Errorf("invalid DRAM bandwidth ceiling %g", gbps)
	}
	if device == "" {
		return ncuCeiling{GBps: gbps}, nil
	}
	norm := func(s string) string { return strings.ToLower(strings.Join(strings.Fields(s), " ")) }
	var names []string
	seen := map[string]bool{}
	if t.Column(ncuDisplayName) >= 0 {
		for i := range t.Launches {
			v := strings.TrimSpace(t.Launches[i].Raw[t.Column(ncuDisplayName)])
			if v != "" && !seen[norm(v)] {
				seen[norm(v)] = true
				names = append(names, v)
			}
		}
	}
	if device == "auto" {
		if len(names) != 1 {
			return ncuCeiling{}, fmt.Errorf("-device auto: export names %d devices %q in %s, want 1", len(names), names, ncuDisplayName)
		}
		device = names[0]
	}
	for _, n := range names {
		if norm(n) != norm(device) {
			return ncuCeiling{}, fmt.Errorf("-device %q: export was recorded on %q", device, n)
		}
	}
	c, err := cudaroofline.LookupDeviceCeiling(device, cudaroofline.DeviceResident)
	if err != nil {
		return ncuCeiling{}, err
	}
	return ncuCeiling{GBps: c.PeakMemoryGBps, Device: c.Device, Provenance: string(c.MemoryProvenance)}, nil
}

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
// is a different quantity. Invalid counts and nonfinite derived values are absent.
func ncuRawLaunches(t *nsightcompute.RawTable, ceilingGBps float64) []ncuRawLaunch {
	finite := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
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
				if ns > 0 && finite(ns) {
					r.DurationNS = &ns
				}
			}
		}
		rd := value(l, "dram__bytes_read.sum", "byte")
		wr := value(l, "dram__bytes_write.sum", "byte")
		if rd != nil && wr != nil && *rd >= 0 && *wr >= 0 {
			b := *rd + *wr
			if finite(b) {
				r.DRAMBytes = &b
			}
			if r.DRAMBytes != nil && r.DurationNS != nil {
				gbps := b / *r.DurationNS
				if finite(gbps) {
					r.DRAMGBps = &gbps
				}
				if r.DRAMGBps != nil && ceilingGBps > 0 && finite(ceilingGBps) {
					f := gbps / ceilingGBps
					if finite(f) {
						r.CeilingFrac = &f
					}
				}
			}
		}
		if v := value(l, "sm__warps_active.avg.pct_of_peak_sustained_active", "%"); v != nil && *v >= 0 && *v <= 100 {
			r.WarpsActive = v
		}
		if v := value(l, "sm__ctas_launched.sum", "block", "cta"); v != nil && *v >= 0 {
			r.CTAsLaunched = v
		}
		out = append(out, r)
	}
	return out
}

func runImportNCURaw(data []byte, maxBytes int, ceilingGBps float64, device string, jsonOutput bool, stdout io.Writer) error {
	t, err := nsightcompute.ReadRaw(bytes.NewReader(data), maxBytes)
	if err != nil {
		return fmt.Errorf("import-ncu: %w", err)
	}
	c, err := resolveNCUCeiling(t, ceilingGBps, device)
	if err != nil {
		return fmt.Errorf("import-ncu: %w", err)
	}
	launches := ncuRawLaunches(t, c.GBps)
	if jsonOutput {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(struct {
			Metrics int `json:"metrics"`
			ncuCeiling
			Launches []ncuRawLaunch `json:"launches"`
		}{len(t.Metrics), c, launches})
	}
	if c.Device != "" {
		fmt.Fprintf(stdout, "ceiling %s: %.2f GB/s (%s)\n", c.Device, c.GBps, c.Provenance)
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
