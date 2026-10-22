package main

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeLogits(t *testing.T, path string, values ...float32) {
	t.Helper()
	data := make([]byte, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestReadCorpusGroupsDirectorySteps(t *testing.T) {
	dir := t.TempDir()
	writeLogits(t, filepath.Join(dir, "case.step1.f32"), 1, 2)
	writeLogits(t, filepath.Join(dir, "case.step0.f32"), 3, 4)
	steps, err := readCorpus(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 2 || steps[0].name != "case" || steps[0].step != 0 || steps[1].step != 1 {
		t.Fatalf("steps = %+v", steps)
	}
}

func TestReplayReportsMovementAndKeepsTemperatureZeroState(t *testing.T) {
	dir := t.TempDir()
	writeLogits(t, filepath.Join(dir, "case.step0.f32"), 1, 2, 3)
	writeLogits(t, filepath.Join(dir, "case.step1.f32"), 3, 2, 1)
	steps, err := readCorpus(dir)
	if err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	report, err := replay(steps, []gridRow{{temperature: 0}, {temperature: 1, topP: 1}}, &out)
	if err != nil {
		t.Fatal(err)
	}
	if report.rows != 2*257*2 || report.positive != 2*257 || report.changed != report.positive {
		t.Fatalf("report = %+v", report)
	}
	if got := strings.Count(out.String(), "case 0 0 2 0000000000000000"); got != 1 {
		t.Fatalf("temperature-zero row = %d, output=%q", got, out.String()[:min(len(out.String()), 200)])
	}
}

func TestManifestResolvesPathsRelativeToManifest(t *testing.T) {
	dir := t.TempDir()
	writeLogits(t, filepath.Join(dir, "step.f32"), 1, 2)
	manifest := filepath.Join(dir, "corpus.tsv")
	if err := os.WriteFile(manifest, []byte("case 0 step.f32\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	steps, err := readCorpus(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 1 || steps[0].path != filepath.Join(dir, "step.f32") {
		t.Fatalf("steps = %+v", steps)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func TestReplayRejectsIncompleteCorpus(t *testing.T) {
	for _, tc := range []struct {
		name  string
		steps []corpusStep
	}{
		{"missing first", []corpusStep{{name: "a", step: 1, logits: []float32{1}}}},
		{"duplicate", []corpusStep{{name: "a", logits: []float32{1}}, {name: "a", logits: []float32{1}}}},
		{"gap", []corpusStep{{name: "a", logits: []float32{1}}, {name: "a", step: 2, logits: []float32{1}}}},
		{"nonfinite", []corpusStep{{name: "a", logits: []float32{float32(math.NaN())}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out strings.Builder
			if _, err := replay(tc.steps, []gridRow{{temperature: 1, topP: 1}}, &out); err == nil || out.Len() != 0 {
				t.Fatalf("error=%v output=%q", err, out.String())
			}
		})
	}
}

func TestReplayResetsRNGForEachCase(t *testing.T) {
	steps := []corpusStep{{name: "a", logits: []float32{1, 2}}, {name: "b", logits: []float32{1, 2}}}
	var out strings.Builder
	var report replayReport
	if err := replayCase(steps, gridRow{temperature: 1, topP: 1}, 12345, &out, &report); err != nil {
		t.Fatal(err)
	}
	rows := strings.Split(strings.TrimSpace(out.String()), "\n")
	if strings.TrimPrefix(rows[0], "a ") != strings.TrimPrefix(rows[1], "b ") {
		t.Fatalf("independent cases consumed different RNG streams: %q", rows)
	}
}

func TestDirectorySortsNumericSteps(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"0", "1", "2", "10"} {
		writeLogits(t, filepath.Join(dir, "case.step"+n+".f32"), 1)
	}
	steps, err := readCorpus(dir)
	if err != nil || steps[2].step != 2 || steps[3].step != 10 {
		t.Fatalf("steps=%v error=%v", steps, err)
	}
}

func TestReplayRejectsReappearingCase(t *testing.T) {
	steps := []corpusStep{{name: "a", logits: []float32{1}}, {name: "b", logits: []float32{1}}, {name: "a", logits: []float32{1}}}
	var out strings.Builder
	if _, err := replay(steps, []gridRow{{temperature: 1}}, &out); err == nil {
		t.Fatal("reappearing case accepted")
	}
}
