package cudahotswap

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestWrongAnswerJSON(t *testing.T) {
	for _, tt := range []struct {
		name      string
		got       float32
		nonfinite int
		token     string
	}{
		{"finite versus zero", 1, 0, `"rel_err":"+Inf"`},
		{"nan prefill no write", float32(math.NaN()), 1, `"got":"NaN"`},
		{"positive infinity", float32(math.Inf(1)), 1, `"got":"+Inf"`},
		{"negative infinity", float32(math.Inf(-1)), 1, `"got":"-Inf"`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c, err := Compare([]float32{tt.got}, []float32{0}, Tolerance{})
			if err != nil || c.OK() || c.Mismatches != 1 || c.NonFinite != tt.nonfinite {
				t.Fatalf("comparison = %+v/%v", c, err)
			}
			r := Report{Outcome: OutcomeWrongAnswer, Correctness: &CorrectnessReport{Trials: []TrialReport{{Comparison: c}}}}
			b, err := json.Marshal(r)
			if err != nil || !json.Valid(b) || !strings.Contains(string(b), tt.token) {
				t.Fatalf("JSON = %s/%v", b, err)
			}
			var decoded Report
			if err := json.Unmarshal(b, &decoded); err != nil {
				t.Fatal(err)
			}
			d := decoded.Correctness.Trials[0].Comparison
			if decoded.Outcome != OutcomeWrongAnswer || decoded.Outcome.ExitCode() != 1 || d.OK() || d.Mismatches != 1 || d.NonFinite != tt.nonfinite || d.Worst == nil {
				t.Fatalf("failure lost: %+v", decoded)
			}
			if math.IsNaN(float64(tt.got)) {
				if !math.IsNaN(d.Worst.Got) {
					t.Fatal("NaN replaced")
				}
			} else if d.Worst.Got != float64(tt.got) {
				t.Fatal("output replaced")
			}
			if !math.IsInf(d.Worst.RelErr, 1) {
				t.Fatal("undefined relative error replaced")
			}
		})
	}
}

func TestMismatchJSONFiniteNamesAndValues(t *testing.T) {
	m := Mismatch{Index: 2, Got: 3, Want: 1, AbsErr: 2, RelErr: 2}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"index":2,"got":3,"want":1,"abs_err":2,"rel_err":2}`
	if string(b) != want {
		t.Fatalf("JSON = %s, want %s", b, want)
	}
	var got Mismatch
	if err := json.Unmarshal(b, &got); err != nil || got != m {
		t.Fatalf("round trip = %+v/%v", got, err)
	}
}

func TestMismatchJSONInvalidLeavesDestination(t *testing.T) {
	for _, value := range []string{`null`, `"unknown"`, `"1"`, `[]`, `{}`, `1e999`} {
		d := Mismatch{Index: 7, Got: 8, Want: 9, AbsErr: 10, RelErr: 11}
		want := d
		b := []byte(`{"index":1,"got":` + value + `,"want":0,"abs_err":1,"rel_err":1}`)
		if err := json.Unmarshal(b, &d); err == nil || d != want {
			t.Fatalf("invalid value %s admitted or changed destination: %+v/%v", value, d, err)
		}
	}
	for _, b := range []string{`null`, `{}`, `{"index":0,"got":1,"want":0,"abs_err":1}`} {
		m := Mismatch{Index: 2, Got: 3}
		want := m
		if err := json.Unmarshal([]byte(b), &m); err == nil || m != want {
			t.Fatalf("incomplete comparison admitted: %s/%v", b, err)
		}
	}
}

func FuzzMismatchJSON(f *testing.F) {
	f.Add(uint64(0), uint64(0), uint64(0), uint64(0))
	f.Add(math.Float64bits(math.NaN()), math.Float64bits(math.Inf(1)), math.Float64bits(math.Inf(-1)), uint64(1<<63))
	f.Fuzz(func(t *testing.T, g, w, a, r uint64) {
		m := Mismatch{Index: 1, Got: math.Float64frombits(g), Want: math.Float64frombits(w), AbsErr: math.Float64frombits(a), RelErr: math.Float64frombits(r)}
		b, err := json.Marshal(m)
		if err != nil || !json.Valid(b) {
			t.Fatalf("cannot encode failure: %s/%v", b, err)
		}
		var d Mismatch
		if err := json.Unmarshal(b, &d); err != nil {
			t.Fatal(err)
		}
		left, right := []float64{m.Got, m.Want, m.AbsErr, m.RelErr}, []float64{d.Got, d.Want, d.AbsErr, d.RelErr}
		for i, v := range left {
			if math.IsNaN(v) {
				if !math.IsNaN(right[i]) {
					t.Fatal("NaN lost")
				}
			} else if math.Float64bits(v) != math.Float64bits(right[i]) {
				t.Fatalf("field %d changed: %v/%v", i, v, right[i])
			}
		}
	})
}
