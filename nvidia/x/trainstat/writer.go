package trainstat

import (
	"bytes"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

// A Writer writes one or more runs to an io.Writer.
// Configuration set before a result applies to it and to later results.
//
// Each result is one Write to the underlying writer, preceded in the same
// Write by any configuration that changed and by the metadata of units
// not yet written. A unit's metadata is what [Writer.Unit] set, or else its
// entry in [Units].
//
// The zero Writer is not usable; use [NewWriter].
type Writer struct {
	w       io.Writer
	keys    []string          // configuration keys in the order first set
	config  map[string]string // configuration for the next result
	written map[string]string // configuration as last written
	units   map[string][]string
	used    map[string]bool // units whose metadata is written
	buf     bytes.Buffer
	err     error
}

// A Value is one measurement and its unit.
type Value struct {
	V    float64
	Unit string
}

// A Run is the status and totals of a run, written by [Writer.Done].
type Run struct {
	Steps, Seqs, Tokens int64
	TrainSec            float64 // sum of step times, excluding evaluation
	Done                bool    // the run finished
	Nonfinite           bool    // the run stopped on a non-finite value
}

// NewWriter returns a Writer that writes to w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{
		w:       w,
		config:  make(map[string]string),
		written: make(map[string]string),
		units:   make(map[string][]string),
		used:    make(map[string]bool),
	}
}

// Config sets a configuration key for the results that follow.
// An empty value removes the key.
// An invalid key or value is reported by the next write and by [Writer.Err].
func (w *Writer) Config(key, value string) {
	if !validKey(key) {
		w.fail(fmt.Errorf("trainstat: invalid configuration key %q", key))
		return
	}
	if strings.ContainsAny(value, "\r\n") || strings.TrimLeft(value, " \t") != value {
		w.fail(fmt.Errorf("trainstat: invalid configuration value %q for %s", value, key))
		return
	}
	if !slices.Contains(w.keys, key) {
		w.keys = append(w.keys, key)
	}
	if value == "" {
		delete(w.config, key)
		return
	}
	w.config[key] = value
}

// Unit records metadata for a unit, such as "better=higher" or "delta=abs".
// It replaces the unit's entry in [Units] and must be called before the
// first result that uses the unit.
func (w *Writer) Unit(unit string, meta ...string) {
	if !validUnit(unit) {
		w.fail(fmt.Errorf("trainstat: invalid unit %q", unit))
		return
	}
	if w.used[unit] {
		w.fail(fmt.Errorf("trainstat: metadata for unit %s after its first use", unit))
		return
	}
	for _, m := range meta {
		k, v, ok := strings.Cut(m, "=")
		if !ok || k == "" || v == "" || strings.ContainsFunc(m, unicode.IsSpace) {
			w.fail(fmt.Errorf("trainstat: invalid metadata %q for unit %s", m, unit))
			return
		}
	}
	w.units[unit] = meta
}

// Result writes one result line: name (without "Benchmark"), iterations, values.
func (w *Writer) Result(name string, n int, vs ...Value) error {
	if w.err != nil {
		return w.err
	}
	if name == "" || strings.ContainsFunc(name, unicode.IsSpace) {
		return w.fail(fmt.Errorf("trainstat: invalid result name %q", name))
	}
	if n < 1 {
		return w.fail(fmt.Errorf("trainstat: %s: iteration count %d < 1", name, n))
	}
	if len(vs) == 0 {
		return w.fail(fmt.Errorf("trainstat: %s: no values", name))
	}
	for _, v := range vs {
		if !validUnit(v.Unit) {
			return w.fail(fmt.Errorf("trainstat: %s: invalid unit %q", name, v.Unit))
		}
	}

	w.buf.Reset()
	for _, k := range w.keys {
		v := w.config[k]
		if v != w.written[k] {
			fmt.Fprintf(&w.buf, "%s: %s\n", k, v)
			if v == "" {
				delete(w.written, k)
			} else {
				w.written[k] = v
			}
		}
	}
	std := Units()
	for _, v := range vs {
		if w.used[v.Unit] {
			continue
		}
		w.used[v.Unit] = true
		meta, ok := w.units[v.Unit]
		if !ok {
			meta = std[v.Unit]
		}
		if len(meta) > 0 {
			fmt.Fprintf(&w.buf, "Unit %s %s\n", v.Unit, strings.Join(meta, " "))
		}
	}
	fmt.Fprintf(&w.buf, "Benchmark%s %d", name, n)
	for _, v := range vs {
		fmt.Fprintf(&w.buf, " %s %s", strconv.FormatFloat(v.V, 'g', -1, 64), v.Unit)
	}
	w.buf.WriteByte('\n')
	if _, err := w.w.Write(w.buf.Bytes()); err != nil {
		return w.fail(fmt.Errorf("trainstat: %w", err))
	}
	return nil
}

// Step writes Train/step=step, one update of a training curve.
func (w *Writer) Step(step int, vs ...Value) error {
	return w.Result("Train/step="+strconv.Itoa(step), 1, vs...)
}

// Eval writes Eval/keys/step=step, one evaluation.
// Keys are name keys such as "split=val/group=all/cal=raw".
func (w *Writer) Eval(step int, keys string, vs ...Value) error {
	return w.Result(join("Eval", keys)+"/step="+strconv.Itoa(step), 1, vs...)
}

// Final writes Final/keys, an end-of-run result.
func (w *Writer) Final(keys string, vs ...Value) error {
	return w.Result(join("Final", keys), 1, vs...)
}

// Done writes the Run line: the run's totals and status.
func (w *Writer) Done(r Run) error {
	return w.Result("Run", 1,
		Value{float64(r.Steps), "steps"},
		Value{float64(r.Seqs), "seqs"},
		Value{float64(r.Tokens), "tokens"},
		Value{r.TrainSec, "train-sec"},
		Value{b2f(r.Done), "done"},
		Value{b2f(r.Nonfinite), "nonfinite"})
}

// Err reports the first error, from an invalid argument or a failed write.
func (w *Writer) Err() error { return w.err }

func (w *Writer) fail(err error) error {
	if w.err == nil {
		w.err = err
	}
	return w.err
}

func join(name, keys string) string {
	if keys == "" {
		return name
	}
	return name + "/" + keys
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// validKey reports whether key is a valid configuration key: a lowercase
// first letter and no spaces, upper-case letters or colons.
func validKey(key string) bool {
	for i, r := range key {
		if i == 0 && !unicode.IsLower(r) || unicode.IsSpace(r) || unicode.IsUpper(r) || r == ':' {
			return false
		}
	}
	return key != ""
}

func validUnit(unit string) bool {
	return unit != "" && !strings.ContainsFunc(unit, unicode.IsSpace)
}
