package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/trainstat"
	"golang.org/x/perf/benchfmt"
	"golang.org/x/perf/benchunit"
)

// data is every input read as benchfmt results.
//
// Each result carries two internal configuration keys: .file, the input's
// label, and .run, the identity of the run it belongs to.
type data struct {
	results []*benchfmt.Result
	units   map[benchfmt.UnitMetadataKey]string
	std     map[benchfmt.UnitMetadataKey]string // trainstat.Units, tidied
}

func newData() *data {
	d := &data{
		units: make(map[benchfmt.UnitMetadataKey]string),
		std:   make(map[benchfmt.UnitMetadataKey]string),
	}
	for unit, meta := range trainstat.Units() {
		_, unit = benchunit.Tidy(1, unit)
		for _, m := range meta {
			k, v, _ := strings.Cut(m, "=")
			d.std[benchfmt.UnitMetadataKey{Unit: unit, Key: k}] = v
		}
	}
	return d
}

// read reads the inputs named by args. An argument label=path names the
// input label rather than path; "-" is standard input.
func read(args []string) (*data, error) {
	d := newData()
	for _, arg := range args {
		label, path := arg, arg
		if l, p, ok := strings.Cut(arg, "="); ok {
			if _, err := os.Stat(arg); err != nil {
				label, path = l, p
			}
		}
		if err := d.readPath(label, path); err != nil {
			return nil, err
		}
	}
	if len(d.results) == 0 {
		return nil, errors.New("no results")
	}
	return d, nil
}

func (d *data) readPath(label, path string) error {
	if path == "-" {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		return d.readBytes(label, "stdin", b)
	}
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		return d.readDir(label, path)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return d.readBytes(label, path, b)
}

// readDir reads a run directory: DIR/trainstat.txt, or else
// DIR/progress.jsonl converted with DIR/train/state.json as configuration.
func (d *data) readDir(label, dir string) error {
	name := filepath.Join(dir, "trainstat.txt")
	b, err := os.ReadFile(name)
	if err == nil {
		_, err := d.readBench(label, name, b, nil, "")
		return err
	}
	if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	name = filepath.Join(dir, "progress.jsonl")
	progress, err := os.ReadFile(name)
	if err != nil {
		return fmt.Errorf("%s: no trainstat.txt or progress.jsonl", dir)
	}
	c := newConverter()
	state := filepath.Join(dir, "train", "state.json")
	if b, err := os.ReadFile(state); err == nil {
		if err := c.addConfig(state, b); err != nil {
			return err
		}
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	var out bytes.Buffer
	if err := c.convert(name, progress, &out); err != nil {
		return err
	}
	_, err = d.readBench(label, name, out.Bytes(), nil, "")
	return err
}

// readBytes reads one input by content: go test -json, go test -v text,
// a JSONL event log, or benchfmt text.
func (d *data) readBytes(label, name string, b []byte) error {
	switch trimmed := bytes.TrimLeft(b, " \t\r\n"); {
	case len(trimmed) > 0 && trimmed[0] == '{':
		var first struct{ Action *string }
		if err := json.NewDecoder(bytes.NewReader(trimmed)).Decode(&first); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		if first.Action != nil {
			return d.readTestJSON(label, name, b)
		}
		var out bytes.Buffer
		if err := newConverter().convert(name, b, &out); err != nil {
			return err
		}
		_, err := d.readBench(label, name, out.Bytes(), nil, "")
		return err
	case isTestText(b):
		return d.readTestText(label, name, b)
	}
	_, err := d.readBench(label, name, b, nil, "")
	return err
}

// isTestText reports whether b looks like go test -v output.
func isTestText(b []byte) bool {
	sc := bufio.NewScanner(bytes.NewReader(b))
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "=== RUN ") || strings.HasPrefix(line, "=== ATTR ") {
			return true
		}
	}
	return false
}

// readBench reads benchfmt text, applying config on top of the file's own
// configuration. Each result is labeled and assigned to a run: the named
// run, or else the one its configuration identifies (every key except
// per-run outputs).
func (d *data) readBench(label, name string, b []byte, config [][2]string, run string) ([]*benchfmt.Result, error) {
	var rs []*benchfmt.Result
	rd := benchfmt.NewReader(bytes.NewReader(b), name)
	for rd.Scan() {
		switch rec := rd.Result().(type) {
		case *benchfmt.SyntaxError:
			log.Print(rec)
		case *benchfmt.UnitMetadata:
			if err := d.setUnit(rec.Unit, rec.Key, rec.Value, name); err != nil {
				return nil, err
			}
		case *benchfmt.Result:
			r := rec.Clone()
			for _, kv := range config {
				setFileConfig(r, kv[0], kv[1])
			}
			r.SetConfig(".file", label)
			if run == "" {
				r.SetConfig(".run", label+"\x00"+runConfig(r))
			} else {
				r.SetConfig(".run", label+"\x00"+run)
			}
			rs = append(rs, r)
		}
	}
	if err := rd.Err(); err != nil {
		return nil, err
	}
	d.results = append(d.results, rs...)
	return rs, nil
}

// runConfig returns the configuration that identifies r's run.
func runConfig(r *benchfmt.Result) string {
	var b strings.Builder
	for _, c := range r.Config {
		if c.File && !strings.HasPrefix(c.Key, "out-") {
			fmt.Fprintf(&b, "%s: %s\n", c.Key, c.Value)
		}
	}
	return b.String()
}

// setFileConfig sets a file configuration key of r.
func setFileConfig(r *benchfmt.Result, key, value string) {
	r.SetConfig(key, value)
	if i, ok := r.ConfigIndex(key); ok {
		r.Config[i].File = true
	}
}

// setUnit records unit metadata. Metadata that conflicts with earlier
// metadata for the same unit is an error, as it is within one benchfmt file.
func (d *data) setUnit(unit, key, value, where string) error {
	_, unit = benchunit.Tidy(1, unit)
	k := benchfmt.UnitMetadataKey{Unit: unit, Key: key}
	if have, ok := d.units[k]; ok && have != value {
		return fmt.Errorf("%s: unit %s: %s=%s, but earlier %s=%s", where, unit, key, value, key, have)
	}
	d.units[k] = value
	return nil
}

// meta returns the metadata value for a tidied unit, falling back to the
// standard units.
func (d *data) meta(unit, key string) string {
	k := benchfmt.UnitMetadataKey{Unit: unit, Key: key}
	if v, ok := d.units[k]; ok {
		return v
	}
	return d.std[k]
}

// isTime reports whether a unit measures time: kind=time, or, without kind
// metadata, a unit that benchmarks report, such as sec/op, B/s or allocs/op.
// Every result of a time unit is a sample; results of other units are
// merged by replicate.
func (d *data) isTime(unit string) bool {
	switch d.meta(unit, "kind") {
	case "time":
		return true
	case "":
		return strings.Contains(unit, "sec") || strings.HasSuffix(unit, "/s") || strings.HasSuffix(unit, "/op")
	}
	return false
}

// better returns 1 if higher is better, -1 if lower is, and 0 if unknown.
func (d *data) better(unit string) int {
	switch d.meta(unit, "better") {
	case "higher":
		return 1
	case "lower":
		return -1
	}
	switch unit {
	case "sec/op", "B/op", "allocs/op":
		return -1
	case "B/s":
		return 1
	}
	return 0
}

func (d *data) normal(unit string) bool   { return d.meta(unit, "assume") == "normal" }
func (d *data) absDelta(unit string) bool { return d.meta(unit, "delta") == "abs" }
