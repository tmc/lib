package dsv4eval

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type liveLogitsRuntime struct {
	smokeRuntime
	live []float32
}

func (m *liveLogitsRuntime) Logits() []float32 {
	if m.live == nil {
		m.live = make([]float32, 4)
	}
	copy(m.live, m.smokeRuntime.Logits())
	return m.live
}
func (m *liveLogitsRuntime) Eval(id int) error {
	for i := range m.live {
		m.live[i] = -999
	}
	return m.smokeRuntime.Eval(id)
}

func TestSmokeArtifactsLiveSelection(t *testing.T) {
	var out bytes.Buffer
	var captures []CaseArtifacts
	model := new(liveLogitsRuntime)
	opt := RunOptions{Build: "fixture", ModelSHA256: strings.Repeat("a", 64), Context: 32, Soft: 3, Hard: 2, Rank: 3, Capture: func(c CaseArtifacts) error { captures = append(captures, c); return nil }}
	if err := RunSmoke(&out, model, runtimeCases(), opt); err != nil {
		t.Fatal(err)
	}
	if len(captures) != 12 {
		t.Fatalf("captures=%d", len(captures))
	}
	c := captures[0]
	if !reflect.DeepEqual(c.History, []int{0, 1, 2}) || c.StopDisposition != "selected_stop_not_consumed" {
		t.Fatalf("history/disposition=%v/%s", c.History, c.StopDisposition)
	}
	if c.Decisions[0].LogitsSHA256 != logitsHash([]float32{0, 10, 0, 0}) {
		t.Fatal("hashed logits after Eval mutation")
	}
	stop := c.Decisions[len(c.Decisions)-1]
	if stop.Chosen != 3 || !stop.Stop || stop.Accepted || len(c.Generated) != 2 {
		t.Fatal("stop token treated as consumed")
	}
	if stop.BeforeHistorySHA256 != c.HistorySHA256 {
		t.Fatal("stop changed history")
	}
	if got := logitsHash(c.TerminalLogits); got != logitsHash([]float32{0, 0, 0, 10}) {
		t.Fatal("terminal vector does not describe accepted history")
	}
}

func TestSmokeArtifactsWriteFailure(t *testing.T) {
	var out bytes.Buffer
	want := errors.New("write failure")
	err := RunSmoke(&out, new(smokeRuntime), runtimeCases(), RunOptions{Build: "fixture", ModelSHA256: strings.Repeat("a", 64), Context: 32, Soft: 3, Hard: 2, Rank: 3, Capture: func(CaseArtifacts) error { return want }})
	if !errors.Is(err, want) || strings.Contains(out.String(), "===== SUMMARY") || strings.Contains(out.String(), "===== CASE") {
		t.Fatalf("write failure qualified: %v", err)
	}
}

func TestSmokeArtifactsSoftChoice(t *testing.T) {
	var chosen [][]int
	for _, soft := range []int{3, 2} {
		var out bytes.Buffer
		var first CaseArtifacts
		err := RunSmoke(&out, new(thinkingRuntime), runtimeCases(), RunOptions{Build: "fixture", ModelSHA256: strings.Repeat("a", 64), Context: 32, Soft: soft, Hard: 2, Rank: 3, Capture: func(c CaseArtifacts) error {
			if c.Index == 1 {
				first = c
			}
			return nil
		}})
		if err != nil {
			t.Fatal(err)
		}
		if soft == 3 {
			fired := false
			for _, d := range first.Decisions {
				if d.Forced == "soft_forced" {
					fired = true
					if d.Chosen != 1 || d.Greedy != 0 || !d.Accepted {
						t.Fatal("soft site did not choose and consume close token")
					}
				}
			}
			if !fired {
				t.Fatal("soft site unexercised")
			}
		}
		chosen = append(chosen, first.Generated)
	}
	if reflect.DeepEqual(chosen[0], chosen[1]) {
		t.Fatal("soft-rule change never affected an accepted token")
	}
}

func TestArtifactRecorderExclusive(t *testing.T) {
	r := ArtifactRecorder{Dir: t.TempDir()}
	c := artifactFixture(t)
	c.Status = "INCOMPLETE"
	if err := r.Record(c); err != nil {
		t.Fatal(err)
	}
	if err := r.Record(c); err == nil {
		t.Fatal("overwrote case")
	}
	b, err := os.ReadFile(filepath.Join(r.Dir, "case-01", "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b, []byte("INCOMPLETE")) {
		t.Fatal("incomplete result erased")
	}
}

func artifactFixture(t *testing.T) CaseArtifacts {
	t.Helper()
	var c CaseArtifacts
	err := RunSmoke(new(bytes.Buffer), new(smokeRuntime), runtimeCases(), RunOptions{Build: "fixture", ModelSHA256: strings.Repeat("a", 64), Context: 32, Soft: 3, Hard: 2, Rank: 3, Capture: func(got CaseArtifacts) error {
		if got.Index == 1 {
			c = got
		}
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestArtifactRecorderRejectsBeforeOutput(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*CaseArtifacts)
	}{
		{"nonfinite", func(c *CaseArtifacts) { c.TerminalLogits[0] = float32(math.NaN()) }},
		{"history", func(c *CaseArtifacts) { c.History[0] = 3 }},
		{"hash", func(c *CaseArtifacts) { c.HistorySHA256 = strings.Repeat("0", 64) }},
		{"accepted-stop", func(c *CaseArtifacts) { c.Decisions[len(c.Decisions)-1].Accepted = true }},
		{"generated", func(c *CaseArtifacts) { c.Generated = c.Generated[:1] }},
		{"model", func(c *CaseArtifacts) { c.ModelSHA256 = "invalid" }},
		{"source", func(c *CaseArtifacts) { c.Source = "unknown" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			c := artifactFixture(t)
			tt.mutate(&c)
			if err := (ArtifactRecorder{Dir: dir}).Record(c); err == nil {
				t.Fatal("malformed case accepted")
			}
			entries, err := os.ReadDir(dir)
			if err != nil || len(entries) != 0 {
				t.Fatalf("created output: %v %v", entries, err)
			}
		})
	}
}

func TestArtifactRecorderPartialDirectory(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "case-01"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := (ArtifactRecorder{Dir: dir}).Record(artifactFixture(t)); err == nil {
		t.Fatal("partial directory reused")
	}
}

func ExampleArtifactRecorder_Record() {
	dir, err := os.MkdirTemp("", "eval-artifacts-")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(dir)
	c := CaseArtifacts{Index: 1, Source: "MMLU-Pro", ID: "2826", Build: "example", ModelSHA256: strings.Repeat("a", 64), Context: 2, Budget: 1, Prompt: []int{0}, History: []int{0}, Status: "INCOMPLETE", ThinkClose: "none", StopDisposition: "selected_stop_not_consumed", TerminalLogits: []float32{1}}
	h := sha256.New()
	tokenHashWrite(h, 0)
	c.HistorySHA256 = fmt.Sprintf("%x", h.Sum(nil))
	c.Decisions = []DecisionArtifact{{Step: 1, Greedy: 0, Chosen: 0, Stop: true, Forced: "none", BeforeHistorySHA256: c.HistorySHA256, LogitsSHA256: logitsHash(c.TerminalLogits)}}
	err = (ArtifactRecorder{Dir: dir}).Record(c)
	fmt.Println(err)
	// Output: <nil>
}
