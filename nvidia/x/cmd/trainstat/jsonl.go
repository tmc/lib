package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/trainstat"
)

func runJSONL(args []string) error {
	fs := flag.NewFlagSet("jsonl", flag.ContinueOnError)
	c := newConverter()
	fs.Func("config", "lift the scalar leaves of JSON `file` into configuration (repeatable)", func(name string) error {
		b, err := os.ReadFile(name)
		if err != nil {
			return err
		}
		return c.addConfig(name, b)
	})
	fs.StringVar(&c.x, "x", c.x, "step events' x `field`, written as the /step= name key")
	fs.StringVar(&c.time, "time", c.time, "step events' duration `field` in milliseconds, written as sec/step")
	fs.StringVar(&c.groupKey, "group-key", c.groupKey, "name `key` for evaluation metric groups")
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "usage: trainstat jsonl [flags] events.jsonl\n")
		fs.PrintDefaults()
	}
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return usageError(errors.New("jsonl takes one input"))
	}
	name := fs.Arg(0)
	var b []byte
	var err error
	if name == "-" {
		b, err = io.ReadAll(os.Stdin)
	} else {
		b, err = os.ReadFile(name)
	}
	if err != nil {
		return err
	}
	return c.convert(name, b, os.Stdout)
}

// A converter converts a JSONL event log, in the shape of laya's
// progress.jsonl, to benchfmt text.
type converter struct {
	x, time, groupKey string

	keys   []string          // configuration keys in the order set
	config map[string]string // configuration
	from   map[string]string // where each key came from
}

func newConverter() *converter {
	return &converter{
		x:        "step",
		time:     "ms",
		groupKey: "group",
		config:   make(map[string]string),
		from:     make(map[string]string),
	}
}

// A row is one result line.
type row struct {
	name string
	vs   []trainstat.Value
}

// A field is one member of a JSON object, in input order.
// Its val is a json.Number, string, bool, nil, []field or []any.
type field struct {
	key string
	val any
}

// addConfig lifts the scalar leaves of the JSON object in b into
// configuration, keeping leaf names.
func (c *converter) addConfig(name string, b []byte) error {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	v, err := decodeValue(dec)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	obj, ok := v.([]field)
	if !ok {
		return fmt.Errorf("%s: not a JSON object", name)
	}
	return c.liftConfig(name, obj)
}

func (c *converter) liftConfig(name string, obj []field) error {
	for _, f := range obj {
		var value string
		switch v := f.val.(type) {
		case []field:
			if err := c.liftConfig(name, v); err != nil {
				return err
			}
			continue
		case json.Number:
			value = v.String()
		case string:
			value = v
		case bool:
			value = fmt.Sprint(v)
		default:
			continue
		}
		if f.key == "event" || f.key == c.x || value == "" {
			continue
		}
		if have, ok := c.config[f.key]; ok {
			if have != value {
				return fmt.Errorf("%s: config %s=%s conflicts with %s=%s from %s", name, f.key, value, f.key, have, c.from[f.key])
			}
			continue
		}
		c.keys = append(c.keys, f.key)
		c.config[f.key] = value
		c.from[f.key] = name
	}
	return nil
}

// convert converts the events in b and writes benchfmt text to w:
// configuration, then Train, Eval, Final and Run results.
func (c *converter) convert(name string, b []byte, w io.Writer) error {
	var (
		train, evals, finals []row
		evalAt               = make(map[string]int) // eval row name → index in evals
		lastEval             string                 // x of the last eval event
		steps                string                 // x of the last step event
		seqs, tokens, sec    float64
		haveSeqs, haveTokens bool
		haveTime, done       bool
		nonfinite            bool // a stop event reports a non-finite value
		events               bool // any object has an event field
	)
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	for n := 1; ; n++ {
		v, err := decodeValue(dec)
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("%s: object %d: %w", name, n, err)
		}
		obj, ok := v.([]field)
		if !ok {
			return fmt.Errorf("%s: object %d: not a JSON object", name, n)
		}
		event, _ := get(obj, "event").(string)
		x, _ := get(obj, c.x).(json.Number)
		events = events || event != ""
		if x == "" && (event == "step" || event == "eval") {
			return fmt.Errorf("%s: object %d: %s event without %s", name, n, event, c.x)
		}
		switch event {
		case "start":
			if err := c.liftConfig(name, obj); err != nil {
				return err
			}
		case "step":
			r := row{name: "Train/step=" + x.String()}
			for _, f := range obj {
				if f.key == c.x {
					continue
				}
				if f.key == c.time {
					if ms, ok := number(f.val); ok {
						r.vs = append(r.vs, trainstat.Value{V: ms / 1e3, Unit: "sec/step"})
						sec += ms / 1e3
						haveTime = true
					}
					continue
				}
				r.vs = appendNumbers(r.vs, "", f)
				if v, ok := number(f.val); ok {
					switch f.key {
					case "seqs":
						seqs += v
						haveSeqs = true
					case "tokens":
						tokens += v
						haveTokens = true
					}
				}
			}
			train = append(train, r)
			steps = x.String()
		case "eval":
			for _, r := range c.evalRows(obj, "/step="+x.String()) {
				r.name = "Eval" + r.name
				if i, ok := evalAt[r.name]; ok {
					if !equalValues(evals[i].vs, r.vs) {
						return fmt.Errorf("%s: object %d: %s differs from an earlier evaluation at the same %s", name, n, r.name, c.x)
					}
					continue
				}
				evalAt[r.name] = len(evals)
				evals = append(evals, r)
			}
			lastEval = x.String()
		case "done":
			done, nonfinite = true, false
		case "stop":
			done = false
			nonfinite, _ = get(obj, "nonfinite").(bool)
		case "":
			for _, r := range c.evalRows(obj, "") {
				finals = append(finals, row{"Final" + r.name, r.vs})
			}
		}
	}
	for _, r := range evals {
		if base, ok := strings.CutSuffix(r.name, "/step="+lastEval); ok {
			finals = append(finals, row{"Final" + strings.TrimPrefix(base, "Eval"), r.vs})
		}
	}

	tw := trainstat.NewWriter(w)
	for _, k := range c.keys {
		tw.Config(k, c.config[k])
	}
	for _, rows := range [][]row{train, evals, finals} {
		for _, r := range rows {
			tw.Result(r.name, 1, r.vs...)
		}
	}
	var run []trainstat.Value
	if steps != "" {
		v, _ := json.Number(steps).Float64()
		run = append(run, trainstat.Value{V: v, Unit: "steps"})
	}
	if haveSeqs {
		run = append(run, trainstat.Value{V: seqs, Unit: "seqs"})
	}
	if haveTokens {
		run = append(run, trainstat.Value{V: tokens, Unit: "tokens"})
	}
	if haveTime {
		run = append(run, trainstat.Value{V: sec, Unit: "train-sec"})
	}
	// Objects without events are one evaluation, not a run.
	if events {
		run = append(run, trainstat.Value{V: b2f(done), Unit: "done"},
			trainstat.Value{V: b2f(nonfinite), Unit: "nonfinite"})
		tw.Result("Run", 1, run...)
	}
	if err := tw.Err(); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}

// evalRows returns the rows of an evaluation object, named
// /split=S/group=G/cal=C+suffix: one per group of "metrics" (cal=raw) and
// of "calibrated" (cal=temp).
func (c *converter) evalRows(obj []field, suffix string) []row {
	var prefix string
	if split, ok := get(obj, "split").(string); ok && split != "" {
		prefix = "/split=" + split
	}
	var rows []row
	for _, sec := range []struct{ key, cal string }{{"metrics", "raw"}, {"calibrated", "temp"}} {
		groups, _ := get(obj, sec.key).([]field)
		for _, g := range groups {
			vals, ok := g.val.([]field)
			if !ok {
				continue
			}
			r := row{name: prefix + "/" + c.groupKey + "=" + g.key + "/cal=" + sec.cal + suffix}
			for _, f := range vals {
				r.vs = appendNumbers(r.vs, "", f)
			}
			if len(r.vs) > 0 {
				rows = append(rows, r)
			}
		}
	}
	return rows
}

// appendNumbers appends the numeric leaves of f as values, named by the
// field path joined with "-" and with "_" replaced by "-".
func appendNumbers(vs []trainstat.Value, prefix string, f field) []trainstat.Value {
	unit := prefix + strings.ReplaceAll(f.key, "_", "-")
	if v, ok := number(f.val); ok {
		return append(vs, trainstat.Value{V: v, Unit: unit})
	}
	if obj, ok := f.val.([]field); ok {
		for _, g := range obj {
			vs = appendNumbers(vs, unit+"-", g)
		}
	}
	return vs
}

func number(v any) (float64, bool) {
	n, ok := v.(json.Number)
	if !ok {
		return 0, false
	}
	f, err := n.Float64()
	return f, err == nil
}

func get(obj []field, key string) any {
	for _, f := range obj {
		if f.key == key {
			return f.val
		}
	}
	return nil
}

func equalValues(a, b []trainstat.Value) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Unit != b[i].Unit || math.Float64bits(a[i].V) != math.Float64bits(b[i].V) {
			return false
		}
	}
	return true
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// decodeValue decodes the next JSON value from dec, keeping the order of
// object members. dec must use numbers.
func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	d, ok := tok.(json.Delim)
	if !ok {
		return tok, nil
	}
	switch d {
	case '{':
		var obj []field
		for dec.More() {
			k, err := dec.Token()
			if err != nil {
				return nil, err
			}
			v, err := decodeValue(dec)
			if err != nil {
				return nil, err
			}
			obj = append(obj, field{k.(string), v})
		}
		if _, err := dec.Token(); err != nil {
			return nil, err
		}
		return obj, nil
	case '[':
		var arr []any
		for dec.More() {
			v, err := decodeValue(dec)
			if err != nil {
				return nil, err
			}
			arr = append(arr, v)
		}
		if _, err := dec.Token(); err != nil {
			return nil, err
		}
		return arr, nil
	}
	return nil, fmt.Errorf("unexpected %v", d)
}
