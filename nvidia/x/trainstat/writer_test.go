package trainstat

import (
	"bytes"
	"errors"
	"math"
	"strings"
	"testing"

	"golang.org/x/perf/benchfmt"
)

func TestWriterErrors(t *testing.T) {
	tests := []struct {
		name  string
		write func(w *Writer) error
		want  string
	}{
		{"upper key", func(w *Writer) error { w.Config("Seed", "1"); return w.Err() }, `invalid configuration key "Seed"`},
		{"space key", func(w *Writer) error { w.Config("a b", "1"); return w.Err() }, "invalid configuration key"},
		{"colon key", func(w *Writer) error { w.Config("a:b", "1"); return w.Err() }, "invalid configuration key"},
		{"empty key", func(w *Writer) error { w.Config("", "1"); return w.Err() }, "invalid configuration key"},
		{"newline value", func(w *Writer) error { w.Config("a", "1\n2"); return w.Err() }, "invalid configuration value"},
		{"space unit", func(w *Writer) error { return w.Step(1, Value{1, "a b"}) }, `invalid unit "a b"`},
		{"empty unit", func(w *Writer) error { return w.Step(1, Value{1, ""}) }, "invalid unit"},
		{"no values", func(w *Writer) error { return w.Step(1) }, "no values"},
		{"space name", func(w *Writer) error { return w.Result("a b", 1, Value{1, "x"}) }, "invalid result name"},
		{"zero iterations", func(w *Writer) error { return w.Result("A", 0, Value{1, "x"}) }, "iteration count 0"},
		{"bad metadata", func(w *Writer) error { w.Unit("loss", "better"); return w.Err() }, `invalid metadata "better"`},
		{"sticky", func(w *Writer) error { w.Config("A", "1"); return w.Step(1, Value{1, "loss"}) }, "invalid configuration key"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := tt.write(NewWriter(&buf))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
			if buf.Len() != 0 {
				t.Errorf("wrote %q after an error", buf.String())
			}
		})
	}
}

func TestWriterLateUnit(t *testing.T) {
	w := NewWriter(new(bytes.Buffer))
	w.Step(1, Value{1, "loss"})
	w.Unit("loss", "better=lower")
	if err := w.Err(); err == nil || !strings.Contains(err.Error(), "after its first use") {
		t.Fatalf("Err = %v, want after its first use", err)
	}
}

type errWriter struct{}

func (errWriter) Write([]byte) (int, error) { return 0, errors.New("disk full") }

func TestWriterWriteError(t *testing.T) {
	w := NewWriter(errWriter{})
	if err := w.Step(1, Value{1, "loss"}); err == nil || !strings.Contains(err.Error(), "disk full") {
		t.Fatalf("Step = %v, want disk full", err)
	}
	if w.Err() == nil {
		t.Fatal("Err = nil after a failed write")
	}
}

// countWriter counts Write calls.
type countWriter struct {
	bytes.Buffer
	n int
}

func (c *countWriter) Write(p []byte) (int, error) {
	c.n++
	return c.Buffer.Write(p)
}

func TestWriterOneWritePerResult(t *testing.T) {
	var c countWriter
	w := NewWriter(&c)
	w.Config("seed", "1")
	w.Config("mode", "lora")
	w.Step(1, Value{1, "loss"}, Value{2, "sec/step"})
	w.Config("seed", "2")
	w.Step(2, Value{1, "loss"})
	w.Step(3, Value{1, "loss"})
	if c.n != 3 {
		t.Errorf("%d writes for 3 results", c.n)
	}
	if got := strings.Count(c.String(), "Unit loss"); got != 1 {
		t.Errorf("Unit loss written %d times, want 1", got)
	}
	if got := strings.Count(c.String(), "mode: lora"); got != 1 {
		t.Errorf("mode written %d times, want 1", got)
	}
}

// TestWriterRoundTrip reads the Writer's output with x/perf's benchfmt
// reader and checks that names, configuration, units and every value's
// bits come back.
func TestWriterRoundTrip(t *testing.T) {
	type result struct {
		name   string
		config map[string]string
		vs     []Value
	}
	third := float64(float32(1.0 / 3))
	want := []result{
		{"Train/step=1", map[string]string{"seed": "1", "mode": "lora"}, []Value{{0.1 + 0.2, "loss"}, {third, "grad-norm"}, {2.412, "sec/step"}}},
		{"Train/step=2", map[string]string{"seed": "1", "mode": "lora"}, []Value{{math.NaN(), "loss"}, {math.Inf(1), "grad-norm"}, {5e-324, "sec/step"}}},
		{"Eval/split=val/group=all/cal=raw/step=2", map[string]string{"seed": "1", "mode": "lora"}, []Value{{0.7968, "accuracy"}}},
		{"Final/split=val/group=all/cal=temp", map[string]string{"seed": "1", "mode": "lora", "out-model-sha256": "7be1"}, []Value{{0.8119, "accuracy"}, {-0, "ece"}}},
		{"Run", map[string]string{"seed": "2", "mode": "lora"}, []Value{{1830, "steps"}, {1, "done"}}},
	}
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.Unit("grad-norm", "assume=normal", "better=lower")
	for i, r := range want {
		switch i {
		case 0:
			w.Config("seed", "1")
			w.Config("mode", "lora")
		case 3:
			w.Config("out-model-sha256", "7be1")
		case 4:
			w.Config("out-model-sha256", "")
			w.Config("seed", "2")
		}
		if err := w.Result(r.name, 1, r.vs...); err != nil {
			t.Fatal(err)
		}
	}

	rd := benchfmt.NewReader(&buf, "trainstat.txt")
	var got []result
	for rd.Scan() {
		switch rec := rd.Result().(type) {
		case *benchfmt.SyntaxError:
			t.Fatal(rec)
		case *benchfmt.Result:
			r := result{name: rec.Name.String(), config: make(map[string]string)}
			for _, c := range rec.Config {
				r.config[c.Key] = string(c.Value)
			}
			for _, v := range rec.Values {
				val, unit := v.Value, v.Unit
				if v.OrigUnit != "" {
					val, unit = v.OrigValue, v.OrigUnit
				}
				r.vs = append(r.vs, Value{val, unit})
			}
			got = append(got, r)
		}
	}
	if err := rd.Err(); err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("read %d results, want %d:\n%s", len(got), len(want), buf.String())
	}
	for i := range want {
		g, w := got[i], want[i]
		if g.name != w.name {
			t.Errorf("result %d: name %q, want %q", i, g.name, w.name)
		}
		if len(g.config) != len(w.config) {
			t.Errorf("%s: config %v, want %v", w.name, g.config, w.config)
		}
		for k, v := range w.config {
			if g.config[k] != v {
				t.Errorf("%s: config %s = %q, want %q", w.name, k, g.config[k], v)
			}
		}
		if len(g.vs) != len(w.vs) {
			t.Errorf("%s: values %v, want %v", w.name, g.vs, w.vs)
			continue
		}
		for j := range w.vs {
			if g.vs[j].Unit != w.vs[j].Unit || math.Float64bits(g.vs[j].V) != math.Float64bits(w.vs[j].V) {
				t.Errorf("%s: value %d = %v, want %v", w.name, j, g.vs[j], w.vs[j])
			}
		}
	}
	units := rd.Units()
	for _, tt := range []struct{ unit, key, want string }{
		{"loss", "better", "lower"},
		{"accuracy", "delta", "abs"},
		{"grad-norm", "better", "lower"},
		{"sec/step", "kind", "time"},
	} {
		m := units.Get(tt.unit, tt.key)
		if m == nil || m.Value != tt.want {
			t.Errorf("unit %s %s = %v, want %s", tt.unit, tt.key, m, tt.want)
		}
	}
}
