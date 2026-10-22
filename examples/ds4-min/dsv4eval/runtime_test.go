package dsv4eval

import (
	"bytes"
	"fmt"
	"math"
	"strings"
	"testing"
)

type smokeRuntime struct {
	step  int
	bad   bool
	evals int
}

type thinkingRuntime struct {
	smokeRuntime
	closed   bool
	answered bool
}

func (m *thinkingRuntime) Sync([]int) error {
	m.step = 0
	m.closed = false
	m.answered = false
	return nil
}
func (m *thinkingRuntime) Eval(id int) error {
	m.step++
	m.evals++
	if id == 1 {
		m.closed = true
	}
	if id == 2 {
		m.answered = true
	}
	return nil
}
func (m *thinkingRuntime) Logits() []float32 {
	if m.closed {
		if !m.answered {
			return []float32{0, 0, 10, 0}
		}
		return []float32{0, 0, 0, 10}
	}
	return []float32{10, 9, 0, 0}
}
func (m *thinkingRuntime) Text(id int) (string, error) {
	if id == 0 {
		return ".", nil
	}
	return m.smokeRuntime.Text(id)
}

func TestSoftRuleMutation(t *testing.T) {
	for _, tc := range []struct {
		soft int
		want string
	}{{3, "soft_forced"}, {2, "hard_forced"}} {
		var b bytes.Buffer
		err := RunSmoke(&b, new(thinkingRuntime), runtimeCases(), RunOptions{Build: "fixture", ModelSHA256: strings.Repeat("a", 64), Context: 32, Soft: tc.soft, Hard: 2, Rank: 3})
		if err != nil {
			t.Fatal(err)
		}
		trace, err := ReadTrace(&b)
		if err != nil {
			t.Fatal(err)
		}
		if trace.Cases[0].Fields["think_close"] != tc.want {
			t.Fatalf("soft=%d close=%s", tc.soft, trace.Cases[0].Fields["think_close"])
		}
	}
}

func (m *smokeRuntime) Render(_, _ string) ([]int, error) { return []int{0}, nil }
func (m *smokeRuntime) Sync([]int) error                  { m.step = 0; return nil }
func (m *smokeRuntime) Eval(int) error                    { m.step++; m.evals++; return nil }
func (m *smokeRuntime) Logits() []float32 {
	if m.bad {
		return []float32{float32(math.NaN())}
	}
	l := make([]float32, 4)
	i := m.step + 1
	if i > 3 {
		i = 3
	}
	l[i] = 10
	return l
}
func (m *smokeRuntime) Text(i int) (string, error) {
	switch i {
	case 1:
		return "</think>", nil
	case 2:
		return "\nAnswer: A", nil
	}
	return "", fmt.Errorf("bad text")
}
func (m *smokeRuntime) IsStop(i int) bool { return i == 3 }
func (m *smokeRuntime) ThinkClose() []int { return []int{1} }

func runtimeCases() []Case {
	cs, _ := SelectN(Cases(), Filter{Suite: HardSmoke}, 0)
	for i := range cs {
		cs[i].Question = "fixture"
		cs[i].Answer = "A"
		cs[i].Kind = Choice
		cs[i].Choices = []string{"first", "second"}
		cs[i].MaxTokens = 8
	}
	return cs
}

func TestRunSmokeExecutesAndFailsClosed(t *testing.T) {
	o := RunOptions{Build: "fixture", ModelSHA256: strings.Repeat("a", 64), Context: 32, Soft: 3, Hard: 2, Rank: 3}
	var b bytes.Buffer
	m := new(smokeRuntime)
	if err := RunSmoke(&b, m, runtimeCases(), o); err != nil {
		t.Fatal(err)
	}
	trace, err := ReadTrace(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if !trace.SummaryPresent || m.evals != 24 || len(trace.Cases) != 12 {
		t.Fatalf("missing execution: evals=%d trace=%+v", m.evals, trace)
	}
	for _, c := range trace.Cases {
		if c.Fields["status"] != "PASSED" || c.Fields["think_close"] != "natural" {
			t.Fatalf("unexpected result %+v", c)
		}
	}
	for _, mutation := range []string{"missing question", "duplicate case", "nonfinite logits", "short context"} {
		t.Run(mutation, func(t *testing.T) {
			cs := runtimeCases()
			opt := o
			model := new(smokeRuntime)
			switch mutation {
			case "missing question":
				cs[0].Question = ""
			case "duplicate case":
				cs[1] = cs[0]
			case "nonfinite logits":
				model.bad = true
			case "short context":
				opt.Context = 2
			}
			var out bytes.Buffer
			if err := RunSmoke(&out, model, cs, opt); err == nil {
				t.Fatal("mutation accepted")
			}
			if strings.Contains(out.String(), "===== SUMMARY =====") {
				t.Fatal("failed run emitted complete summary")
			}
		})
	}
}

func TestSmokeCorpusRequiresExternalAdmission(t *testing.T) {
	if _, err := LoadSmokeCorpus(t.TempDir()); err == nil {
		t.Fatal("missing external manifest accepted")
	}
}

func TestGreedyFiniteAndRank(t *testing.T) {
	if got, err := greedyFinite([]float32{1, 1, 0}); err != nil || got != 0 {
		t.Fatalf("tie=%d,%v", got, err)
	}
	if tokenRank([]float32{3, 2, 1}, 2) != 3 {
		t.Fatal("rank")
	}
	if tokenRank([]float32{1, 1, 1, 1}, 3) != 4 {
		t.Fatal("tied rank must preserve upstream token ID order")
	}
	for _, v := range []float32{float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
		if _, err := greedyFinite([]float32{v}); err == nil {
			t.Fatal("nonfinite accepted")
		}
	}
}
