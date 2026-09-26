package main

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
)

const compareGEMMsUsage = "usage: cudaprof compare -gemms [-phase i] [-warmup-until=RE | -warmup-ms=N] [-max-records=N] [-journal-bytes=N] [--capture-file-bytes=N] [--capture-total-bytes=N] [-json] base.pprof target.pprof"

// A GEMMComparison joins two captures' cuBLASLt calls by shape, operand
// order and types.
type GEMMComparison struct {
	Base   *GEMMCensus      `json:"base"`
	Target *GEMMCensus      `json:"target"`
	Shapes []GEMMShapeDelta `json:"shapes"`
}

// A GEMMShapeDelta is one shape as each side ran it. A side that has no
// call of the shape has a nil entry.
type GEMMShapeDelta struct {
	Shape  string          `json:"shape"`
	Op     string          `json:"op"`
	Types  string          `json:"types"`
	Base   *GEMMShapeTotal `json:"base,omitempty"`
	Target *GEMMShapeTotal `json:"target,omitempty"`
}

// A GEMMShapeTotal is one side's calls of one shape, over every algorithm,
// beta and epilogue it used.
type GEMMShapeTotal struct {
	Calls       int   `json:"calls"`
	Joined      int   `json:"joined"`
	KernelNanos int64 `json:"kernel_ns"`
	// Rows are the census rows the total sums.
	Rows []GEMMRow `json:"rows"`
}

func (t *GEMMShapeTotal) perCall() int64 {
	if t == nil || t.Joined == 0 {
		return 0
	}
	return t.KernelNanos / int64(t.Joined)
}

func (t *GEMMShapeTotal) nanos() int64 {
	if t == nil {
		return 0
	}
	return t.KernelNanos
}

// compareWantsGEMMs reports whether compare's arguments ask for -gemms.
func compareWantsGEMMs(args []string) bool {
	for _, a := range args {
		switch a {
		case "-gemms", "--gemms", "-gemms=true", "--gemms=true":
			return true
		}
	}
	return false
}

func runCompareGEMMs(args []string, stdout io.Writer) error {
	opts, paths, err := parseGEMMsArgs(args, stdout, "compare", 2)
	if err != nil || paths == nil {
		return err
	}
	var sides [2]*GEMMCensus
	for i, path := range paths {
		sides[i], err = readGEMMCensus(path, opts)
		if err != nil {
			return fmt.Errorf("compare: %w", err)
		}
	}
	c := compareGEMMs(sides[0], sides[1])
	if opts.json {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(c)
	}
	return c.write(stdout)
}

func compareGEMMs(base, target *GEMMCensus) *GEMMComparison {
	c := &GEMMComparison{Base: base, Target: target}
	shapes := make(map[string]*GEMMShapeDelta)
	add := func(census *GEMMCensus, side func(*GEMMShapeDelta) **GEMMShapeTotal) {
		for _, r := range census.Rows {
			key := r.shapeKey()
			d, ok := shapes[key]
			if !ok {
				d = &GEMMShapeDelta{Shape: r.Shape(), Op: r.Op, Types: r.Types}
				shapes[key] = d
			}
			t := side(d)
			if *t == nil {
				*t = new(GEMMShapeTotal)
			}
			(*t).Calls += r.Calls
			(*t).Joined += r.Joined
			(*t).KernelNanos += r.KernelNanos
			(*t).Rows = append((*t).Rows, r)
		}
	}
	add(base, func(d *GEMMShapeDelta) **GEMMShapeTotal { return &d.Base })
	add(target, func(d *GEMMShapeDelta) **GEMMShapeTotal { return &d.Target })
	for _, d := range shapes {
		c.Shapes = append(c.Shapes, *d)
	}
	// Largest change first: that is the row a reader is looking for.
	sort.Slice(c.Shapes, func(i, j int) bool {
		a, b := &c.Shapes[i], &c.Shapes[j]
		da := abs64(a.Target.nanos() - a.Base.nanos())
		db := abs64(b.Target.nanos() - b.Base.nanos())
		if da != db {
			return da > db
		}
		return a.Shape+a.Op+a.Types < b.Shape+b.Op+b.Types
	})
	return c
}

func (c *GEMMComparison) write(w io.Writer) error {
	for _, s := range []struct {
		name string
		c    *GEMMCensus
	}{{"base", c.Base}, {"target", c.Target}} {
		var total int64
		for i := range s.c.Rows {
			total += s.c.Rows[i].KernelNanos
		}
		fmt.Fprintf(w, "%s: %s\n  %s\n  %d calls, %d joined, %d unparsed; kernel time %s ms\n", s.name, s.c.Capture, s.c.Side.windowLine(), s.c.Calls, s.c.Joined, s.c.Unparsed, gemmMS(total))
	}
	fmt.Fprintln(w)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "shape MxNxK\top\ttypes\tbase calls\ttarget calls\tbase us/call\ttarget us/call\tdelta us/call\tbase ms\ttarget ms\tdelta ms\t")
	for i := range c.Shapes {
		d := &c.Shapes[i]
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n", d.Shape, d.Op, d.Types,
			gemmCalls(d.Base), gemmCalls(d.Target),
			gemmPerCall(d.Base), gemmPerCall(d.Target), gemmDelta(d.Base, d.Target, (*GEMMShapeTotal).perCall, gemmUS),
			gemmTotal(d.Base), gemmTotal(d.Target), gemmDelta(d.Base, d.Target, (*GEMMShapeTotal).nanos, gemmMS))
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	fmt.Fprintln(w, "\nhow each side ran it (beta, epilogue, algo: kernel):")
	for i := range c.Shapes {
		d := &c.Shapes[i]
		fmt.Fprintf(w, "%s %s %s\n", d.Shape, d.Op, d.Types)
		for _, s := range []struct {
			name string
			t    *GEMMShapeTotal
		}{{"base", d.Base}, {"target", d.Target}} {
			if s.t == nil {
				fmt.Fprintf(w, "  %-6s  -\n", s.name)
				continue
			}
			for _, r := range s.t.Rows {
				fmt.Fprintf(w, "  %-6s  %dx beta=%s %s algo %s: %s\n", s.name, r.Calls, r.Beta, r.Epilogue, r.Algo, r.Kernel())
			}
		}
	}
	return nil
}

func gemmCalls(t *GEMMShapeTotal) string {
	if t == nil {
		return "-"
	}
	return strconv.Itoa(t.Calls)
}

func gemmPerCall(t *GEMMShapeTotal) string {
	if t == nil || t.Joined == 0 {
		return "-"
	}
	return gemmUS(t.perCall())
}

func gemmTotal(t *GEMMShapeTotal) string {
	if t == nil {
		return "-"
	}
	return gemmMS(t.KernelNanos)
}

// gemmDelta formats target minus base, or "-" when either side lacks the
// shape or ran it with no joined kernel.
func gemmDelta(b, t *GEMMShapeTotal, f func(*GEMMShapeTotal) int64, format func(int64) string) string {
	if b == nil || t == nil || b.Joined == 0 || t.Joined == 0 {
		return "-"
	}
	d := f(t) - f(b)
	s := format(d)
	if d >= 0 {
		s = "+" + s
	}
	return strings.TrimSpace(s)
}
