package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/perf/benchfmt"
	"golang.org/x/perf/benchmath"
	"golang.org/x/perf/benchproc"
)

// options are the flags common to the summary and the gate.
type options struct {
	col, row, table   string
	filter, ignore    string
	replicate         string
	alpha, confidence float64
	skip              int
}

func (o *options) register(fs *flag.FlagSet) {
	fs.StringVar(&o.col, "col", "", "split columns by `projection`; default: the configuration keys that differ between inputs, else .file")
	fs.StringVar(&o.row, "row", ".fullname", "split rows by `projection`")
	fs.StringVar(&o.table, "table", ".config", "split tables by `projection`")
	fs.StringVar(&o.filter, "filter", "*", "use only results matching `query`")
	fs.StringVar(&o.ignore, "ignore", "", "ignore variations in `keys`")
	fs.StringVar(&o.replicate, "replicate", "", "replicate `key`; default: seed, or /seed if no result has seed configuration")
	fs.Float64Var(&o.alpha, "alpha", 0.05, "significance level")
	fs.Float64Var(&o.confidence, "confidence", 0.95, "confidence level of intervals")
	fs.IntVar(&o.skip, "skip", 20, "exclude the first `n` steps from throughput")
}

// An analysis is the inputs grouped into tables, units, rows and columns.
type analysis struct {
	d       *data
	st      *stat
	rep     string // replicate projection
	reps    []string
	colDesc string // the column projection, for the header
	cols    []benchproc.Key
	tables  []*table
	results []*benchfmt.Result // derived and filtered
	done    int                // runs with done 1
	stopped int                // runs with done 0
	valFrac string             // val-frac configuration, if any
	steps   []runSteps         // per-run step times
	notes   []string
}

// A table is the blocks of one table key.
type table struct {
	key    benchproc.Key
	blocks []*block
}

// A block is one unit of a table.
type block struct {
	unit  string
	rows  []benchproc.Key
	cells map[[2]benchproc.Key]*sample // by row and column
}

// runSteps is one run's step times, for percentiles.
type runSteps struct {
	run string
	col benchproc.Key
	sec []float64
}

func analyze(d *data, o *options) (*analysis, error) {
	a := &analysis{d: d, st: newStat(o.confidence, o.alpha)}
	results := derive(d, o.skip)

	// Run status, before filtering changes Run results.
	status := make(map[string]float64)
	var runs []string
	for _, r := range results {
		if string(r.Name.Base()) != "Run" {
			continue
		}
		if v, ok := r.Value("done"); ok {
			run := r.GetConfig(".run")
			if _, ok := status[run]; !ok {
				runs = append(runs, run)
			}
			status[run] = v
		}
	}

	// The default replicate is configuration seed or else name key /seed,
	// so that attributes and artifacts of one test agree.
	a.rep = o.replicate
	repQuery := o.replicate
	if a.rep == "" {
		a.rep, repQuery = "seed", "seed,/seed"
	}
	ignore := []string{"run", "val_sha256"}
	for _, r := range results {
		for _, c := range r.Config {
			if c.File && strings.HasPrefix(c.Key, "out-") && !slices.Contains(ignore, c.Key) {
				ignore = append(ignore, c.Key)
			}
		}
		if v := r.GetConfig("val-frac"); v != "" && v != "0" && r.GetConfig("val") == "" {
			a.valFrac = v
		}
	}

	filter, err := benchproc.NewFilter(o.filter)
	if err != nil {
		return nil, fmt.Errorf("parsing -filter: %w", err)
	}
	var parser benchproc.ProjectionParser
	repBy, err := parser.Parse(repQuery, filter)
	if err != nil {
		return nil, fmt.Errorf("parsing -replicate: %w", err)
	}
	replicate := func(r *benchfmt.Result) string {
		if o.replicate != "" {
			return repBy.Project(r).StringValues()
		}
		if v := r.GetConfig("seed"); v != "" {
			return v
		}
		return nameKey(r, "seed")
	}
	if o.ignore != "" {
		ignore = append(ignore, o.ignore)
	}
	if _, err := parser.Parse(strings.Join(ignore, ","), filter); err != nil {
		return nil, fmt.Errorf("parsing -ignore: %w", err)
	}
	a.colDesc = o.col
	if a.colDesc == "" {
		skip := append(slices.Clone(ignore), a.rep, "seed")
		if o.ignore != "" {
			skip = append(skip, strings.Split(o.ignore, ",")...)
		}
		cols := autoColumns(results, skip)
		fillColumns(results, cols)
		a.colDesc = strings.Join(cols, ",")
		if a.colDesc == "" {
			a.colDesc = ".file"
		}
	}
	colBy, err := parser.Parse(a.colDesc, filter)
	if err != nil {
		return nil, fmt.Errorf("parsing -col: %w", err)
	}
	tableBy, err := parser.Parse(o.table, filter)
	if err != nil {
		return nil, fmt.Errorf("parsing -table: %w", err)
	}
	rowBy, err := parser.Parse(o.row, filter)
	if err != nil {
		return nil, fmt.Errorf("parsing -row: %w", err)
	}

	type group struct {
		vals, reps, hashes []string // parallel
		xs                 []float64
	}
	groups := make(map[*block]map[[2]benchproc.Key]*group)
	tables := make(map[benchproc.Key]*table)
	var tableKeys, colKeys []benchproc.Key
	seenCol := make(map[benchproc.Key]bool)
	seenRep := make(map[string]bool)
	selected := make(map[string]bool) // runs with a result that matches the filter
	var train []*benchfmt.Result
	for _, r := range results {
		switch string(r.Name.Base()) {
		case "Train":
			train = append(train, r)
			continue
		case "Eval":
			continue
		}
		if ok, err := filter.Apply(r); err != nil {
			return nil, err
		} else if !ok {
			continue
		}
		a.results = append(a.results, r)
		selected[r.GetConfig(".run")] = true
		tk, rk, ck := tableBy.Project(r), rowBy.Project(r), colBy.Project(r)
		rep := replicate(r)
		if rep != "" && !seenRep[rep] {
			seenRep[rep] = true
			a.reps = append(a.reps, rep)
		}
		if !seenCol[ck] {
			seenCol[ck] = true
			colKeys = append(colKeys, ck)
		}
		t := tables[tk]
		if t == nil {
			t = &table{key: tk}
			tables[tk] = t
			tableKeys = append(tableKeys, tk)
		}
		for _, v := range r.Values {
			var b *block
			for _, x := range t.blocks {
				if x.unit == v.Unit {
					b = x
				}
			}
			if b == nil {
				b = &block{unit: v.Unit, cells: make(map[[2]benchproc.Key]*sample)}
				t.blocks = append(t.blocks, b)
				groups[b] = make(map[[2]benchproc.Key]*group)
			}
			k := [2]benchproc.Key{rk, ck}
			g := groups[b][k]
			if g == nil {
				g = new(group)
				groups[b][k] = g
				if !slices.Contains(b.rows, rk) {
					b.rows = append(b.rows, rk)
				}
			}
			g.xs = append(g.xs, v.Value)
			g.reps = append(g.reps, rep)
			g.hashes = append(g.hashes, r.GetConfig("val_sha256"))
		}
	}
	if len(a.results) == 0 {
		return nil, fmt.Errorf("no results match")
	}
	for _, run := range runs {
		switch {
		case !selected[run]:
		case status[run] == 1:
			a.done++
		default:
			a.stopped++
		}
	}
	for _, r := range train {
		if sec, ok := r.Value("sec/step"); ok && stepOf(r) > o.skip && selected[r.GetConfig(".run")] {
			a.addStep(r, colBy.Project(r), sec)
		}
	}
	sortReps(a.reps)
	a.cols = colKeys
	benchproc.SortKeys(a.cols)
	benchproc.SortKeys(tableKeys)
	for _, tk := range tableKeys {
		t := tables[tk]
		a.tables = append(a.tables, t)
		for _, b := range t.blocks {
			benchproc.SortKeys(b.rows)
			for k, g := range groups[b] {
				s, note := a.merge(b.unit, g.xs, g.reps, g.hashes)
				if note != "" {
					a.notes = append(a.notes, fmt.Sprintf("%s %s %s: %s", k[0].StringValues(), b.unit, k[1].StringValues(), note))
				}
				b.cells[k] = s
			}
		}
	}
	sort.Strings(a.notes)
	return a, nil
}

// merge makes a cell's sample. Every value of a time unit is a sample
// value. Values of other units are merged by replicate: results with the
// same replicate must be bit-identical, and give one value.
func (a *analysis) merge(unit string, xs []float64, reps, hashes []string) (*sample, string) {
	if a.d.isTime(unit) {
		return &sample{xs: xs}, ""
	}
	order := slices.Clone(reps)
	sortReps(order)
	order = slices.Compact(order)
	s := &sample{reps: order}
	for _, rep := range order {
		first := -1
		for i := range xs {
			if reps[i] != rep {
				continue
			}
			if first < 0 {
				first = i
				continue
			}
			if math.Float64bits(xs[i]) != math.Float64bits(xs[first]) {
				s.bad = "not reproducible"
				note := fmt.Sprintf("results differ: %v and %v", xs[first], xs[i])
				if rep != "" {
					note = fmt.Sprintf("replicate %s: %s", rep, note)
				}
				return s, note
			}
		}
		s.xs = append(s.xs, xs[first])
		s.vals = append(s.vals, hashes[first])
	}
	return s, ""
}

// addStep records a Train step time for the step-time percentiles.
func (a *analysis) addStep(r *benchfmt.Result, col benchproc.Key, sec float64) {
	run := r.GetConfig(".run")
	for i := range a.steps {
		if a.steps[i].run == run && a.steps[i].col == col {
			a.steps[i].sec = append(a.steps[i].sec, sec)
			return
		}
	}
	a.steps = append(a.steps, runSteps{run: run, col: col, sec: []float64{sec}})
}

// sortReps sorts replicate values, numerically when they are numbers.
func sortReps(reps []string) {
	slices.SortStableFunc(reps, func(x, y string) int {
		fx, errx := strconv.ParseFloat(x, 64)
		fy, erry := strconv.ParseFloat(y, 64)
		if errx == nil && erry == nil && fx != fy {
			if fx < fy {
				return -1
			}
			return 1
		}
		return strings.Compare(x, y)
	})
}

// fillColumns sets each key in cols on the results that lack it, when the
// other results of their input agree on its value. Benchmark lines in go
// test output then join the column of the tests beside them.
func fillColumns(results []*benchfmt.Result, cols []string) {
	for _, k := range cols {
		vals := make(map[string]string) // by input; "\x00" if more than one
		for _, r := range results {
			f := r.GetConfig(".file")
			if v := r.GetConfig(k); v != "" {
				if old, ok := vals[f]; ok && old != v {
					v = "\x00"
				}
				vals[f] = v
			}
		}
		for _, r := range results {
			v := vals[r.GetConfig(".file")]
			if r.GetConfig(k) == "" && v != "" && v != "\x00" {
				setFileConfig(r, k, v)
			}
		}
	}
}

// autoColumns returns the file configuration keys, other than skip, whose
// values differ between the inputs that set them, or, if only one input
// sets a key, that has more than one value within it.
func autoColumns(results []*benchfmt.Result, skip []string) []string {
	var keys []string
	for _, r := range results {
		for _, c := range r.Config {
			if c.File && !slices.Contains(keys, c.Key) && !slices.Contains(skip, c.Key) {
				keys = append(keys, c.Key)
			}
		}
	}
	var cols []string
	for _, k := range keys {
		sets := make(map[string][]string) // by input
		for _, r := range results {
			f := r.GetConfig(".file")
			if v := r.GetConfig(k); v != "" && !slices.Contains(sets[f], v) {
				sets[f] = append(sets[f], v)
			}
		}
		var strs []string
		for _, set := range sets {
			sort.Strings(set)
			strs = append(strs, strings.Join(set, "\x00"))
		}
		differ := len(sets) == 1 && strings.Contains(strs[0], "\x00")
		for _, s := range strs {
			differ = differ || s != strs[0]
		}
		if differ {
			cols = append(cols, k)
		}
	}
	return cols
}

// derive adds each run's Best and Throughput results, unless the run
// already has them.
//
// Best/keys holds, for each unit with a better direction, the best value
// over the run's Eval/keys/step=N results. Throughput holds the mean
// sec/step, and tokens/s, over Train steps after the first skip, excluding
// each step that follows an evaluation; its iteration count is the number
// of steps.
func derive(d *data, skip int) []*benchfmt.Result {
	type run struct {
		have       map[string]bool // base names present
		best       map[string]*benchfmt.Result
		bestOrder  []string
		evalSteps  map[int]bool
		train      []*benchfmt.Result
		lastResult *benchfmt.Result
	}
	runs := make(map[string]*run)
	var order []string
	for _, r := range d.results {
		id := r.GetConfig(".run")
		rn := runs[id]
		if rn == nil {
			rn = &run{have: make(map[string]bool), best: make(map[string]*benchfmt.Result), evalSteps: make(map[int]bool)}
			runs[id] = rn
			order = append(order, id)
		}
		base := string(r.Name.Base())
		rn.have[base] = true
		switch base {
		case "Train":
			rn.train = append(rn.train, r)
		case "Eval":
			rn.evalSteps[stepOf(r)] = true
			name := "Best" + string(withoutKey(r.Name, "step"))
			b := rn.best[name]
			if b == nil {
				b = r.Clone()
				b.Name = benchfmt.Name(name)
				b.Values = nil
				rn.best[name] = b
				rn.bestOrder = append(rn.bestOrder, name)
			}
			for _, v := range r.Values {
				better := d.better(v.Unit)
				if better == 0 || math.IsNaN(v.Value) {
					continue
				}
				i := slices.IndexFunc(b.Values, func(x benchfmt.Value) bool { return x.Unit == v.Unit })
				if i < 0 {
					b.Values = append(b.Values, v)
				} else if float64(better)*(v.Value-b.Values[i].Value) > 0 {
					b.Values[i] = v
				}
			}
		}
	}

	results := slices.Clone(d.results)
	for _, id := range order {
		rn := runs[id]
		if !rn.have["Best"] {
			for _, name := range rn.bestOrder {
				if b := rn.best[name]; len(b.Values) > 0 {
					results = append(results, b)
				}
			}
		}
		if rn.have["Throughput"] {
			continue
		}
		var sec, tokens float64
		var n int
		var haveTokens bool
		var last *benchfmt.Result
		for _, r := range rn.train {
			step := stepOf(r)
			s, ok := r.Value("sec/step")
			if !ok || step <= skip || rn.evalSteps[step-1] {
				continue
			}
			sec += s
			n++
			if t, ok := r.Value("tokens"); ok {
				tokens += t
				haveTokens = true
			}
			last = r
		}
		if n == 0 {
			continue
		}
		t := last.Clone()
		t.Name = benchfmt.Name("Throughput")
		t.Iters = n
		t.Values = []benchfmt.Value{{Value: sec / float64(n), Unit: "sec/step"}}
		if haveTokens {
			t.Values = append(t.Values, benchfmt.Value{Value: tokens / sec, Unit: "tokens/s"})
		}
		results = append(results, t)
	}
	return results
}

// stepOf returns the value of r's /step name key, or 0.
func stepOf(r *benchfmt.Result) int {
	n, _ := strconv.Atoi(nameKey(r, "step"))
	return n
}

// withoutKey returns the name keys of n, other than key, as /k=v....
func withoutKey(n benchfmt.Name, key string) string {
	_, parts := n.Parts()
	var b strings.Builder
	for _, p := range parts {
		if !strings.HasPrefix(string(p), "/"+key+"=") {
			b.Write(p)
		}
	}
	return b.String()
}

// nameKey returns the value of r's /key name key, or "".
func nameKey(r *benchfmt.Result, key string) string {
	_, parts := r.Name.Parts()
	for _, p := range parts {
		if v, ok := strings.CutPrefix(string(p), "/"+key+"="); ok {
			return v
		}
	}
	return ""
}

// assumption returns the benchmath assumption for a unit that is not
// assume=normal.
func (a *analysis) assumption(unit string) benchmath.Assumption {
	if a.d.meta(unit, "assume") == "exact" {
		return benchmath.AssumeExact
	}
	return benchmath.AssumeNothing
}

// normal reports whether unit is summarized by a mean and t interval.
func (a *analysis) normal(unit string) bool {
	return a.d.normal(unit) && !a.d.isTime(unit)
}

func runSummary(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("trainstat", flag.ContinueOnError)
	var o options
	o.register(fs)
	format := fs.String("format", "text", "output `format`: text, csv or benchfmt")
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "usage: trainstat [flags] inputs...\n       trainstat jsonl|repro|gate ...\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return usageError(err)
	}
	if fs.NArg() == 0 {
		fs.Usage()
		return usageError(fmt.Errorf("no inputs"))
	}
	if *format != "text" && *format != "csv" && *format != "benchfmt" {
		return usageError(fmt.Errorf("unknown -format %q", *format))
	}
	d, err := read(fs.Args())
	if err != nil {
		return err
	}
	if *format == "benchfmt" {
		return writeBenchfmt(stdout, d, derive(d, o.skip), o.filter)
	}
	a, err := analyze(d, &o)
	if err != nil {
		return err
	}
	if *format == "csv" {
		return a.writeCSV(stdout)
	}
	return a.writeText(stdout)
}

// writeBenchfmt writes the results that match filter, with the metadata
// of their units.
func writeBenchfmt(w io.Writer, d *data, results []*benchfmt.Result, query string) error {
	filter, err := benchproc.NewFilter(query)
	if err != nil {
		return fmt.Errorf("parsing -filter: %w", err)
	}
	var keep []*benchfmt.Result
	var units []string
	for _, r := range results {
		if ok, err := filter.Apply(r); err != nil {
			return err
		} else if !ok {
			continue
		}
		keep = append(keep, r)
		for _, v := range r.Values {
			if !slices.Contains(units, v.Unit) {
				units = append(units, v.Unit)
			}
		}
	}
	var b strings.Builder
	for _, u := range units {
		var meta []string
		for _, k := range []string{"better", "assume", "delta", "kind"} {
			if v := d.meta(u, k); v != "" {
				meta = append(meta, k+"="+v)
			}
		}
		if len(meta) > 0 {
			fmt.Fprintf(&b, "Unit %s %s\n", u, strings.Join(meta, " "))
		}
	}
	if _, err := io.WriteString(w, b.String()); err != nil {
		return err
	}
	bw := benchfmt.NewWriter(w)
	for _, r := range keep {
		if err := bw.Write(r); err != nil {
			return err
		}
	}
	return nil
}
