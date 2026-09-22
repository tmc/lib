package cudalineinfo

import (
	"bufio"
	"fmt"
	"io"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// A Rollup is the attribution of one program frame together with everything
// built beneath it. Flat counts the SASS instructions attributed to the
// frame's own site. Cum adds the instructions of every site whose location
// lists the frame among its callers, so a helper's instructions are charged
// to each line that called the helper as well as to the helper.
type Rollup struct {
	Site     Site
	Function string
	Flat     int
	Cum      int
}

// Rollup charges each site's instructions to the site and to every frame
// above it, and returns one entry per frame in file and line order. Sites
// without a recorded location roll up only to themselves.
func (a *Attribution) Rollup() []Rollup {
	by := map[Site]*Rollup{}
	frame := func(s Site, function string) *Rollup {
		r := by[s]
		if r == nil {
			r = &Rollup{Site: s, Function: function}
			by[s] = r
		}
		if r.Function == "" {
			r.Function = function
		}
		return r
	}
	for _, s := range a.Sites() {
		n := len(a.SASS[s])
		loc, _ := a.Unit.Location(s)
		self := frame(s, loc.Function)
		self.Flat += n
		self.Cum += n
		seen := map[Site]bool{s: true}
		for _, c := range loc.Callers {
			cs := Site{File: c.File, Line: c.Line}
			if seen[cs] {
				continue
			}
			seen[cs] = true
			frame(cs, c.Function).Cum += n
		}
	}
	var out []Rollup
	for _, r := range by {
		out = append(out, *r)
	}
	sortRollups(out)
	return out
}

// Callers returns the frames above a site, nearest first, or nil when the
// unit recorded none.
func (a *Attribution) Callers(s Site) []cudair.Frame {
	loc, ok := a.Unit.Location(s)
	if !ok {
		return nil
	}
	return loc.Callers
}

// WriteRollup writes one row per frame with its flat and cumulative
// instruction counts. Frames that only call into other frames have a flat
// count of zero; a table with no such row means no location recorded any
// callers.
func WriteRollup(w io.Writer, a *Attribution) error {
	src := newSourceReader(a.Unit)
	rollup := a.Rollup()
	sites := make([]Site, len(rollup))
	for i, r := range rollup {
		sites[i] = r.Site
	}
	names := SiteNames(sites)
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%s: SASS instructions by frame, cumulative\n", a.Unit.Name)
	tw := tabwriter.NewWriter(bw, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  SITE\tFUNCTION\tSOURCE\tFLAT\tCUM")
	for _, r := range rollup {
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%d\t%d\n", names[r.Site], shortFunction(r.Function), src.line(r.Site), r.Flat, r.Cum)
	}
	tw.Flush()
	return bw.Flush()
}

func sortRollups(rs []Rollup) {
	sites := make([]Site, len(rs))
	by := make(map[Site]Rollup, len(rs))
	for i, r := range rs {
		sites[i] = r.Site
		by[r.Site] = r
	}
	sortSites(sites)
	for i, s := range sites {
		rs[i] = by[s]
	}
}

// shortFunction drops the package path from a symbol, keeping the package
// name: "github.com/x/y.(*T).M" becomes "y.(*T).M".
func shortFunction(name string) string {
	for i := len(name) - 1; i >= 0; i-- {
		if name[i] == '/' {
			return name[i+1:]
		}
	}
	return name
}
