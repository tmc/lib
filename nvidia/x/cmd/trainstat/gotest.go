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
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/trainstat"
)

// A goTest is one test's trainstat attributes and artifacts from go test
// output. Each becomes one run.
type goTest struct {
	pkg, name string
	keys      []string // attribute keys in order
	attrs     map[string]string
	artifacts string // artifact directory, or ""
	action    string // pass, fail or skip
}

// A testReader collects tests and benchmark output from go test output.
type testReader struct {
	d           *data
	label, name string
	tests       []*goTest
	index       map[string]*goTest // by package and name
	bench       map[string]*bytes.Buffer
	pkgs        []string // bench keys in order
}

func newTestReader(d *data, label, name string) *testReader {
	return &testReader{
		d:     d,
		label: label,
		name:  name,
		index: make(map[string]*goTest),
		bench: make(map[string]*bytes.Buffer),
	}
}

func (t *testReader) test(pkg, name string) *goTest {
	k := pkg + "\x00" + name
	g := t.index[k]
	if g == nil {
		g = &goTest{pkg: pkg, name: name, attrs: make(map[string]string)}
		t.index[k] = g
		t.tests = append(t.tests, g)
	}
	return g
}

func (t *testReader) attr(pkg, test, key, value string) error {
	if !strings.HasPrefix(key, "trainstat") {
		return nil
	}
	g := t.test(pkg, test)
	if have, ok := g.attrs[key]; ok {
		if have != value {
			return fmt.Errorf("%s: %s: attribute %s set to %q and %q", t.name, test, key, have, value)
		}
		return nil
	}
	g.keys = append(g.keys, key)
	g.attrs[key] = value
	return nil
}

func (t *testReader) action(pkg, test, action string) {
	if g := t.index[pkg+"\x00"+test]; g != nil {
		g.action = action
	}
}

func (t *testReader) output(pkg, text string) {
	b := t.bench[pkg]
	if b == nil {
		b = new(bytes.Buffer)
		t.bench[pkg] = b
		t.pkgs = append(t.pkgs, pkg)
	}
	b.WriteString(text)
}

// readTestJSON reads a go test -json stream.
func (d *data) readTestJSON(label, name string, b []byte) error {
	t := newTestReader(d, label, name)
	dec := json.NewDecoder(bytes.NewReader(b))
	for {
		var e struct {
			Action, Package, Test, Output, Key, Value, Path string
		}
		if err := dec.Decode(&e); err == io.EOF {
			break
		} else if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		switch e.Action {
		case "attr":
			if err := t.attr(e.Package, e.Test, e.Key, e.Value); err != nil {
				return err
			}
		case "artifacts":
			t.test(e.Package, e.Test).artifacts = e.Path
		case "pass", "fail", "skip":
			t.action(e.Package, e.Test, e.Action)
		case "output":
			if e.Test == "" || strings.HasPrefix(e.Test, "Benchmark") {
				t.output(e.Package, e.Output)
			}
		}
	}
	return t.finish()
}

// readTestText reads go test -v output.
func (d *data) readTestText(label, name string, b []byte) error {
	t := newTestReader(d, label, name)
	sc := bufio.NewScanner(bytes.NewReader(b))
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if rest, ok := strings.CutPrefix(line, "=== ATTR "); ok {
			test, kv, _ := strings.Cut(strings.TrimLeft(rest, " "), " ")
			key, value, _ := strings.Cut(kv, " ")
			if err := t.attr("", test, key, value); err != nil {
				return err
			}
			continue
		}
		if rest, ok := strings.CutPrefix(line, "=== ARTIFACTS "); ok {
			test, path, _ := strings.Cut(strings.TrimLeft(rest, " "), " ")
			t.test("", test).artifacts = path
			continue
		}
		if rest, ok := strings.CutPrefix(strings.TrimLeft(line, " "), "--- "); ok {
			action, rest, _ := strings.Cut(rest, ": ")
			test, _, _ := strings.Cut(rest, " ")
			switch action {
			case "PASS", "FAIL", "SKIP":
				t.action("", test, strings.ToLower(action))
			}
			continue
		}
		// A package's summary line ends its tests.
		if f := strings.Fields(line); len(f) >= 2 && (f[0] == "ok" || f[0] == "FAIL" || f[0] == "?") {
			for _, g := range t.tests {
				if g.pkg == "" {
					delete(t.index, "\x00"+g.name)
					g.pkg = f[1]
				}
			}
			continue
		}
		t.output("", line+"\n")
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return t.finish()
}

// finish converts the collected tests and benchmark output to results.
func (t *testReader) finish() error {
	for _, pkg := range t.pkgs {
		if _, err := t.d.readBench(t.label, t.name, t.bench[pkg].Bytes(), nil, ""); err != nil {
			return err
		}
	}
	for _, g := range t.tests {
		if err := t.finishTest(g); err != nil {
			return err
		}
	}
	return nil
}

func (t *testReader) finishTest(g *goTest) error {
	if g.action == "skip" {
		log.Printf("%s: %s skipped", t.name, g.name)
		return nil
	}
	base, sub, _ := strings.Cut(g.name, "/")
	if sub != "" {
		sub = "/" + sub
	}
	own := base
	for _, p := range []string{"Test", "Benchmark", "Fuzz"} {
		own = strings.TrimPrefix(own, p)
	}
	own += sub
	isBench := strings.HasPrefix(base, "Benchmark")

	var config [][2]string
	if g.pkg != "" {
		config = append(config, [2]string{"pkg", g.pkg})
	}
	config = append(config, [2]string{"test", base})
	var text bytes.Buffer
	w := trainstat.NewWriter(&text)
	for _, kv := range config {
		w.Config(kv[0], kv[1])
	}
	var names []string
	rows := make(map[string][]trainstat.Value)
	for _, key := range g.keys {
		value := g.attrs[key]
		if k, ok := strings.CutPrefix(key, "trainstat.config:"); ok {
			w.Config(k, value)
			config = append(config, [2]string{k, value})
			continue
		}
		if unit, ok := strings.CutPrefix(key, "trainstat.unit:"); ok {
			w.Unit(unit, strings.Fields(value)...)
			continue
		}
		rest, ok := strings.CutPrefix(key, "trainstat:")
		if !ok {
			continue
		}
		name, unit := own, rest
		if i := strings.LastIndexByte(rest, ':'); i >= 0 {
			name, unit = rest[:i]+sub, rest[i+1:]
		} else if isBench {
			continue // the benchmark line carries the test's own values
		}
		v, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return fmt.Errorf("%s: %s: attribute %s: %w", t.name, g.name, key, err)
		}
		if _, ok := rows[name]; !ok {
			names = append(names, name)
		}
		rows[name] = append(rows[name], trainstat.Value{V: v, Unit: unit})
	}

	run := g.pkg + "\x00" + g.name
	done := b2f(g.action == "pass")
	var haveRun bool
	if g.artifacts != "" {
		name := filepath.Join(g.artifacts, "trainstat.txt")
		b, err := os.ReadFile(name)
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return err
		}
		rs, err := t.d.readBench(t.label, name, b, config, run)
		if err != nil {
			return err
		}
		for _, r := range rs {
			r.Name = append(r.Name, sub...)
			if string(r.Name.Base()) != "Run" {
				continue
			}
			haveRun = true
			for i, v := range r.Values {
				if v.Unit == "done" {
					r.Values[i].Value = math.Min(v.Value, done)
				}
			}
		}
	} else if len(names) == 0 {
		return nil
	}
	for _, name := range names {
		w.Result(name, 1, rows[name]...)
	}
	if !haveRun {
		w.Result("Run"+sub, 1, trainstat.Value{V: done, Unit: "done"})
	}
	if err := w.Err(); err != nil {
		return fmt.Errorf("%s: %s: %w", t.name, g.name, err)
	}
	_, err := t.d.readBench(t.label, t.name, text.Bytes(), nil, run)
	return err
}
