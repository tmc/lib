package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
	"unicode/utf8"

	"golang.org/x/perf/benchproc"
	"golang.org/x/perf/benchunit"
)

// A grid is rows of cells, written with aligned columns.
// Columns in sep are separators, drawn only in header rows.
type grid struct {
	rows [][]string
	sep  map[int]bool
}

func (g *grid) add(cells ...string) { g.rows = append(g.rows, cells) }

func (g *grid) writeText(w io.Writer) error {
	var width []int
	for _, row := range g.rows {
		for i, c := range row {
			if i >= len(width) {
				width = append(width, 0)
			}
			width[i] = max(width[i], utf8.RuneCountInString(c))
		}
	}
	var b strings.Builder
	for _, row := range g.rows {
		var line strings.Builder
		for i, c := range row {
			if i > 0 {
				line.WriteString("  ")
			}
			line.WriteString(c)
			line.WriteString(strings.Repeat(" ", width[i]-utf8.RuneCountInString(c)))
		}
		b.WriteString(strings.TrimRight(line.String(), " "))
		b.WriteByte('\n')
	}
	_, err := io.WriteString(w, b.String())
	return err
}

func (g *grid) writeCSV(cw *csv.Writer) {
	for _, row := range g.rows {
		var rec []string
		for i, c := range row {
			if !g.sep[i] {
				rec = append(rec, c)
			}
		}
		cw.Write(rec)
	}
}

// grid returns the grid of a block: a row label column, then for each
// column its summary, and for columns after the first, the difference
// from the first, its interval, the test, and whether it is worse.
func (a *analysis) grid(b *block) *grid {
	g := &grid{sep: make(map[int]bool)}
	head1 := []string{b.unit}
	head2 := []string{""}
	for j, col := range a.cols {
		g.sep[len(head1)] = true
		head1 = append(head1, "│", col.StringValues())
		head2 = append(head2, "│", a.label(b, col))
		if j > 0 {
			var ci string
			if a.normal(b.unit) && a.haveCI(b, col) {
				ci = fmt.Sprintf("%g%% CI of Δ", a.st.confidence*100)
			}
			head1 = append(head1, "", "", "", "")
			head2 = append(head2, "Δ", ci, "", "")
		}
	}
	g.sep[len(head1)] = true
	g.add(append(head1, "│")...)
	g.add(append(head2, "│")...)

	add := func(label string, cells [][]string) {
		line := []string{label}
		for _, c := range cells {
			line = append(line, "")
			line = append(line, c...)
		}
		g.add(line...)
	}
	for _, row := range b.rows {
		add(row.StringValues(), a.formatRow(b, row))
	}
	if cells := a.geomean(b); cells != nil {
		add("geomean", cells)
	}
	return g
}

// haveCI reports whether any row of b has an interval for the difference
// between col and the first column.
func (a *analysis) haveCI(b *block, col benchproc.Key) bool {
	for _, row := range b.rows {
		if c, ok := a.compare(b, row, col); ok && !math.IsNaN(c.lo) {
			return true
		}
	}
	return false
}

// geomean returns the cells of a geometric mean row for a block of a
// relative unit (not delta=abs) with more than one row and column, over the rows
// present in every column, or nil if there is none or a center is not
// positive.
func (a *analysis) geomean(b *block) [][]string {
	if a.d.absDelta(b.unit) || len(b.rows) < 2 || len(a.cols) < 2 {
		return nil
	}
	logs := make([]float64, len(a.cols))
	var n int
rows:
	for _, row := range b.rows {
		centers := make([]float64, len(a.cols))
		for j, col := range a.cols {
			s := b.cell(row, col)
			if s == nil {
				continue rows
			}
			centers[j] = a.summarize(b, s).center
			if !(centers[j] > 0) {
				return nil
			}
		}
		for j, c := range centers {
			logs[j] += math.Log(c)
		}
		n++
	}
	if n < 2 {
		return nil
	}
	gms := make([]float64, len(a.cols))
	for j := range gms {
		gms[j] = math.Exp(logs[j] / float64(n))
	}
	scaler := benchunit.CommonScale(gms, benchunit.ClassOf(b.unit))
	cells := [][]string{{scaler.Format(gms[0])}}
	for _, gm := range gms[1:] {
		cells = append(cells, []string{scaler.Format(gm), fmt.Sprintf("%+.2f%%", 100*(gm/gms[0]-1)), "", "", ""})
	}
	return cells
}

// label returns a column's summary label for a block.
func (a *analysis) label(b *block, col benchproc.Key) string {
	allOne := true
	for _, row := range b.rows {
		if s := b.cells[[2]benchproc.Key{row, col}]; s != nil && len(s.xs) > 1 {
			allOne = false
		}
	}
	switch {
	case allOne:
		return "n=1"
	case a.normal(b.unit):
		return "mean"
	}
	return "median"
}

// cell returns the sample of a block cell, or nil.
func (b *block) cell(row, col benchproc.Key) *sample {
	s := b.cells[[2]benchproc.Key{row, col}]
	if s == nil || s.bad != "" || len(s.xs) == 0 {
		return nil
	}
	return s
}

// summarize returns the summary of a block cell.
func (a *analysis) summarize(b *block, s *sample) summary {
	return a.st.summarize(s, a.normal(b.unit), a.assumption(b.unit))
}

// compare compares a block's cell in column col with the first column.
func (a *analysis) compare(b *block, row, col benchproc.Key) (comparison, bool) {
	base, s := b.cell(row, a.cols[0]), b.cell(row, col)
	if base == nil || s == nil {
		return comparison{}, false
	}
	return a.st.compare(base, s, a.normal(b.unit), true, a.assumption(b.unit)), true
}

// formatRow formats one row of a block: per column, its cells.
func (a *analysis) formatRow(b *block, row benchproc.Key) [][]string {
	normal := a.normal(b.unit)
	sums := make([]summary, len(a.cols))
	cmps := make([]comparison, len(a.cols))
	have := make([]bool, len(a.cols))
	haveCmp := make([]bool, len(a.cols))
	dec, pctDec := -1, -1
	var centers []float64
	for j, col := range a.cols {
		s := b.cell(row, col)
		if s == nil {
			continue
		}
		have[j] = true
		sums[j] = a.summarize(b, s)
		centers = append(centers, sums[j].center)
		dec = max(dec, decimals((sums[j].hi-sums[j].lo)/2))
		if j > 0 {
			cmps[j], haveCmp[j] = a.compare(b, row, col)
			if haveCmp[j] {
				c := cmps[j]
				dec = max(dec, decimals((c.hi-c.lo)/2))
				pctDec = max(pctDec, decimals(50*(c.hi-c.lo)/math.Abs(sums[0].center)))
			}
		}
	}
	if dec < 0 && len(centers) > 0 {
		// No interval: show four significant digits of the largest value.
		var big float64
		for _, c := range centers {
			big = max(big, math.Abs(c))
		}
		dec = max(decimals(big)+2, 0)
	}
	scaler := benchunit.CommonScale(centers, benchunit.ClassOf(b.unit))

	var out [][]string
	for j, col := range a.cols {
		var cells []string
		switch s := b.cells[[2]benchproc.Key{row, col}]; {
		case s != nil && s.bad != "":
			cells = append(cells, s.bad)
		case !have[j]:
			cells = append(cells, "")
		case normal && sums[j].n > 1:
			cells = append(cells, format(sums[j].center, dec, false)+" ± "+format((sums[j].hi-sums[j].lo)/2, dec, false))
		case normal:
			cells = append(cells, format(sums[j].center, dec, false))
		case sums[j].n == 1:
			cells = append(cells, scaler.Format(sums[j].center))
		default:
			cells = append(cells, scaler.Format(sums[j].center)+" ± "+sums[j].pct)
		}
		if j == 0 {
			out = append(out, cells)
			continue
		}
		if !haveCmp[j] {
			out = append(out, append(cells, "", "", "", ""))
			continue
		}
		c := cmps[j]
		var delta, ci string
		switch {
		case c.method == "U" || c.method == "exact":
			delta = c.fmtDelta
		case sums[0].center != 0 && (!normal || !a.d.absDelta(b.unit)):
			delta, ci = a.pct(c, sums[0].center, pctDec)
		default:
			delta = format(c.delta, dec, true)
			if !math.IsNaN(c.lo) {
				ci = "[" + format(c.lo, dec, true) + ", " + format(c.hi, dec, true) + "]"
			}
		}
		if c.method != "point" && !c.significant(a.st.alpha) && delta != "~" {
			delta = "~"
		}
		info := c.info()
		if c.note != "" {
			info += " (" + c.note + ")"
		}
		var worse string
		if c.significant(a.st.alpha) && float64(a.d.better(b.unit))*c.delta < 0 {
			worse = "(worse)"
		}
		out = append(out, append(cells, delta, ci, info, worse))
	}
	return out
}

// pct formats a comparison as percentages of the base center.
func (a *analysis) pct(c comparison, base float64, dec int) (delta, ci string) {
	f := func(v float64) string {
		if dec < 0 {
			return fmt.Sprintf("%+.2f%%", v)
		}
		return fmt.Sprintf("%+.*f%%", dec, v)
	}
	delta = f(100 * c.delta / base)
	if !math.IsNaN(c.lo) {
		ci = "[" + f(100*c.lo/base) + ", " + f(100*c.hi/base) + "]"
	}
	return delta, ci
}

// writeText writes the summary.
func (a *analysis) writeText(w io.Writer) error {
	var b strings.Builder
	fmt.Fprintf(&b, "columns: %s\n", a.colDesc)
	if len(a.reps) > 0 {
		fmt.Fprintf(&b, "replicates: %s (%s)\n", a.rep, strings.Join(a.reps, " "))
	} else {
		fmt.Fprintf(&b, "replicates: none (no %s); results with equal configuration must be identical\n", a.rep)
	}
	if a.done+a.stopped > 0 {
		fmt.Fprintf(&b, "runs: %d done, %d stopped\n", a.done, a.stopped)
	}
	if a.valFrac != "" {
		fmt.Fprintf(&b, "val-frac %s: each seed has its own validation set; Eval, Final and Best spread includes eval-set sampling\n", a.valFrac)
	}
	if _, err := io.WriteString(w, b.String()); err != nil {
		return err
	}
	var differ, tests int
	for _, t := range a.tables {
		var hdr strings.Builder
		for _, f := range t.key.Projection().FlattenedFields() {
			if v := t.key.Get(f); v != "" {
				fmt.Fprintf(&hdr, "\n%s: %s", f.Name, v)
			}
		}
		if hdr.Len() > 0 {
			hdr.WriteString("\n")
		}
		if _, err := io.WriteString(w, hdr.String()); err != nil {
			return err
		}
		for _, blk := range t.blocks {
			if _, err := io.WriteString(w, "\n"); err != nil {
				return err
			}
			if err := a.grid(blk).writeText(w); err != nil {
				return err
			}
			for _, row := range blk.rows {
				for _, col := range a.cols[1:] {
					if c, ok := a.compare(blk, row, col); ok && !math.IsNaN(c.p) {
						tests++
						if c.significant(a.st.alpha) {
							differ++
						}
					}
				}
			}
		}
	}
	if err := a.writeStepTimes(w); err != nil {
		return err
	}
	b.Reset()
	if tests > 0 {
		fmt.Fprintf(&b, "\n%d of %d comparisons differ at α=%g; ~%.1f expected by chance if nothing changed\n", differ, tests, a.st.alpha, a.st.alpha*float64(tests))
	}
	if b.Len() == 0 && len(a.notes) > 0 {
		b.WriteString("\n")
	}
	for _, n := range a.notes {
		fmt.Fprintf(&b, "note: %s\n", n)
	}
	_, err := io.WriteString(w, b.String())
	return err
}

// writeStepTimes writes percentiles of Train sec/step per column: each
// run's percentile, then the median over the column's runs.
func (a *analysis) writeStepTimes(w io.Writer) error {
	if len(a.steps) == 0 {
		return nil
	}
	g := &grid{}
	g.add("sec/step", "runs", "p50", "p90", "p99", "max")
	for _, col := range a.cols {
		var per [4][]float64
		for _, rs := range a.steps {
			if rs.col != col {
				continue
			}
			xs := slices.Sorted(slices.Values(rs.sec))
			for i, p := range []float64{0.5, 0.9, 0.99, 1} {
				per[i] = append(per[i], quantile(xs, p))
			}
		}
		if len(per[0]) == 0 {
			continue
		}
		line := []string{col.StringValues(), fmt.Sprint(len(per[0]))}
		for _, xs := range per {
			slices.Sort(xs)
			line = append(line, fmt.Sprintf("%.4g", quantile(xs, 0.5)))
		}
		g.add(line...)
	}
	if _, err := io.WriteString(w, "\nTrain step times after -skip, per run; median over runs\n"); err != nil {
		return err
	}
	return g.writeText(w)
}

// quantile returns the p-quantile of sorted xs by the nearest-rank method.
func quantile(xs []float64, p float64) float64 {
	i := int(math.Ceil(p*float64(len(xs)))) - 1
	return xs[max(i, 0)]
}

// writeCSV writes the summary's blocks as CSV.
func (a *analysis) writeCSV(w io.Writer) error {
	cw := csv.NewWriter(w)
	for _, t := range a.tables {
		for _, blk := range t.blocks {
			a.grid(blk).writeCSV(cw)
			cw.Write(nil)
		}
	}
	cw.Flush()
	return cw.Error()
}
