package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"

	"golang.org/x/perf/benchfmt"
)

// pinning lists the configuration keys that must be equal for two runs to
// be expected bit-identical: the default -pin list.
var pinning = []string{"commit", "lib-nvidia", "nvrtc", "cublas", "driver", "device", "exact", "tf32", "mlx-c", "libmlx"}

func runRepro(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("repro", flag.ContinueOnError)
	pin := fs.String("pin", strings.Join(pinning, ","), "comma-separated pinning `keys`")
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "usage: trainstat repro [-pin keys] a b\n")
		fs.PrintDefaults()
	}
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		fs.Usage()
		return usageError(errors.New("repro takes two inputs"))
	}
	var runs [2]*reproRun
	for i, arg := range fs.Args() {
		d, err := read([]string{arg})
		if err != nil {
			return err
		}
		if runs[i], err = newReproRun(arg, d); err != nil {
			return err
		}
	}
	if differ := pinDiffs(runs[0], runs[1], strings.Split(*pin, ",")); len(differ) > 0 {
		fmt.Fprintf(stdout, "refused: pinning keys differ: %s\n", strings.Join(differ, ", "))
		return &exitCodeError{Code: 1, Err: errors.New("refused: pinning keys differ")}
	}
	ok, err := repro(stdout, runs[0], runs[1])
	if err != nil {
		return err
	}
	if !ok {
		return &exitCodeError{Code: 1, Err: errors.New("not bit-identical")}
	}
	return nil
}

// pinDiffs returns each pinning key whose value differs between a and b,
// with the two values, such as "exact (true vs false)". A key set in only
// one run differs, as does one whose value is "unknown", which a trainer
// records when it cannot tell, as for a binary built from a checkout that
// has since changed.
func pinDiffs(a, b *reproRun, pins []string) []string {
	var differ []string
	for _, k := range unionKeys(a, b) {
		va, oka := a.config[k]
		vb, okb := b.config[k]
		if slices.Contains(pins, k) && (va != vb || oka != okb || va == "unknown") {
			differ = append(differ, fmt.Sprintf("%s (%s vs %s)", k, show(va, oka), show(vb, okb)))
		}
	}
	return differ
}

func show(v string, ok bool) string {
	if !ok {
		return "unset"
	}
	return v
}

// unionKeys returns the configuration keys of a and then those only in b.
func unionKeys(a, b *reproRun) []string {
	keys := slices.Clone(a.keys)
	for _, k := range b.keys {
		if !slices.Contains(keys, k) {
			keys = append(keys, k)
		}
	}
	return keys
}

// A reproRun is one run's configuration and values, as written.
type reproRun struct {
	d      *data
	keys   []string
	config map[string]string
	names  []string
	values map[string][]benchfmt.Value // by full name
}

func newReproRun(name string, d *data) (*reproRun, error) {
	r := &reproRun{d: d, config: make(map[string]string), values: make(map[string][]benchfmt.Value)}
	run := d.results[0].GetConfig(".run")
	for _, res := range d.results {
		if res.GetConfig(".run") != run {
			return nil, fmt.Errorf("%s: more than one run", name)
		}
		for _, c := range res.Config {
			if !c.File {
				continue
			}
			if _, ok := r.config[c.Key]; !ok {
				r.keys = append(r.keys, c.Key)
			}
			r.config[c.Key] = string(c.Value)
		}
		full := string(res.Name.Full())
		if _, ok := r.values[full]; ok {
			return nil, fmt.Errorf("%s: more than one %s result", name, full)
		}
		r.names = append(r.names, full)
		r.values[full] = res.Values
	}
	return r, nil
}

// repro compares two runs bit for bit and reports whether they are
// identical: every result of a unit that is not a time, and every out-
// configuration key. Its caller has checked the pinning keys.
func repro(w io.Writer, a, b *reproRun) (bool, error) {
	var out strings.Builder
	identical := true

	var outKeys, differ []string
	for _, k := range unionKeys(a, b) {
		switch {
		case strings.HasPrefix(k, "out-"):
			outKeys = append(outKeys, k)
		case k != "run" && k != "host" && a.config[k] != b.config[k]:
			differ = append(differ, fmt.Sprintf("%s %q != %q", k, a.config[k], b.config[k]))
		}
	}
	if len(differ) == 0 {
		out.WriteString("config: equal (ignored: run, host)\n")
	} else {
		fmt.Fprintf(&out, "config: differs: %s\n", strings.Join(differ, "; "))
	}

	type count struct{ compared, differ int }
	counts := make(map[string]*count)
	var bases, units, skipped []string
	var first string
	var values int // differing values
	names := slices.Clone(a.names)
	for _, n := range b.names {
		if _, ok := a.values[n]; !ok {
			names = append(names, n)
		}
	}
	for _, name := range names {
		base := string(benchfmt.Name(name).Base())
		c := counts[base]
		if c == nil {
			c = new(count)
			counts[base] = c
			bases = append(bases, base)
		}
		va, oka := a.values[name]
		vb, okb := b.values[name]
		if !oka || !okb {
			c.compared++
			c.differ++
			values++
			identical = false
			if first == "" {
				missing := "b"
				if !oka {
					missing = "a"
				}
				first = fmt.Sprintf("%s missing in %s", name, missing)
			}
			continue
		}
		compared, diff := false, false
		for _, x := range va {
			if a.d.isTime(x.Unit) || b.d.isTime(x.Unit) {
				if !slices.Contains(skipped, x.Unit) {
					skipped = append(skipped, x.Unit)
				}
				continue
			}
			compared = true
			if !slices.Contains(units, x.Unit) {
				units = append(units, x.Unit)
			}
			xa, unit := orig(x)
			i := slices.IndexFunc(vb, func(y benchfmt.Value) bool { return y.Unit == x.Unit })
			if i < 0 {
				diff = true
				values++
				if first == "" {
					first = fmt.Sprintf("%s %s missing in b", name, unit)
				}
				continue
			}
			xb, _ := orig(vb[i])
			if math.Float64bits(xa) != math.Float64bits(xb) {
				diff = true
				values++
				if first == "" {
					first = fmt.Sprintf("%s %s %v != %v (%s)", name, unit, xa, xb, ulps(xa, xb))
				}
			}
		}
		for _, y := range vb {
			if !b.d.isTime(y.Unit) && !slices.ContainsFunc(va, func(x benchfmt.Value) bool { return x.Unit == y.Unit }) {
				compared, diff = true, true
				values++
				if first == "" {
					first = fmt.Sprintf("%s %s missing in a", name, y.Unit)
				}
			}
		}
		if compared {
			c.compared++
		}
		if diff {
			c.differ++
			identical = false
		}
	}

	var parts []string
	for _, base := range bases {
		if c := counts[base]; c.compared > 0 {
			parts = append(parts, fmt.Sprintf("%d %s", c.compared, base))
		}
	}
	fmt.Fprintf(&out, "compared bitwise: %s; units %s\n", strings.Join(parts, ", "), strings.Join(units, " "))
	if len(skipped) > 0 {
		fmt.Fprintf(&out, "skipped kind=time: %s\n", strings.Join(skipped, " "))
	}
	var outLines []string
	for _, k := range outKeys {
		va, oka := a.config[k]
		vb, okb := b.config[k]
		if oka && okb && va == vb {
			outLines = append(outLines, k+": equal")
		} else {
			outLines = append(outLines, fmt.Sprintf("%s: differs (%s vs %s)", k, show(va, oka), show(vb, okb)))
			identical = false
		}
	}
	if !identical && len(differ) == 0 {
		// The caller has checked the pinning keys, and the rest of the
		// configuration is equal: the runs should have agreed.
		out.WriteString("nondeterministic: same configuration, different results\n")
	}
	if first != "" {
		fmt.Fprintf(&out, "first difference: %s\n", first)
		parts = parts[:0]
		for _, base := range bases {
			if c := counts[base]; c.differ > 0 {
				parts = append(parts, fmt.Sprintf("%s %d of %d", base, c.differ, c.compared))
			}
		}
		verb := "values differ"
		if values == 1 {
			verb = "value differs"
		}
		fmt.Fprintf(&out, "%d %s (results: %s)\n", values, verb, strings.Join(parts, ", "))
	}
	for _, line := range outLines {
		fmt.Fprintln(&out, line)
	}
	if identical {
		out.WriteString("ok: bit-identical\n")
	} else {
		out.WriteString("FAIL: not bit-identical\n")
	}
	_, err := io.WriteString(w, out.String())
	return identical, err
}

// orig returns a value and unit as written, before tidying.
func orig(v benchfmt.Value) (float64, string) {
	if v.OrigUnit != "" {
		return v.OrigValue, v.OrigUnit
	}
	return v.Value, v.Unit
}

// ulps describes the distance between x and y in float32 units in the
// last place, and in float64 units when they round to one float32.
func ulps(x, y float64) string {
	if math.IsNaN(x) || math.IsNaN(y) {
		return "NaN"
	}
	d := dist(ordered32(float32(x)), ordered32(float32(y)))
	if d == 0 && x != y {
		return fmt.Sprintf("0 float32 ulp, %d float64 ulp", dist(ordered64(x), ordered64(y)))
	}
	return fmt.Sprintf("%d float32 ulp", d)
}

// ordered64 maps float64 bits to integers in the order of the values.
func ordered64(x float64) int64 {
	b := int64(math.Float64bits(x))
	if b < 0 {
		b = math.MinInt64 - b
	}
	return b
}

func ordered32(x float32) int64 {
	b := int64(int32(math.Float32bits(x)))
	if b < 0 {
		b = math.MinInt32 - b
	}
	return b
}

func dist(a, b int64) uint64 {
	if a > b {
		return uint64(a) - uint64(b)
	}
	return uint64(b) - uint64(a)
}
