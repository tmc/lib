package main

import (
	"bytes"
	"encoding/json"
	"github.com/tmc/lib/examples/ds4-min/sampler"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSampledDecisionRecordsActualResult(t *testing.T) {
	dir := t.TempDir()
	if err := writeSampledDecision(dir, 0, 0, 123, 456, true, []int{7, 8}, []float32{1, 9}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "step-0000.json"))
	if err != nil {
		t.Fatal(err)
	}
	var got sampledDecision
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.Token != 0 || !got.Stop || got.RNGBefore != 123 || got.RNGAfter != 456 || len(got.LogitsSHA256) != 64 || len(got.HistorySHA256) != 64 {
		t.Fatalf("record=%+v", got)
	}
	if err := writeSampledDecision(dir, 0, 0, 123, 456, true, []int{7, 8}, []float32{1, 9}); err == nil {
		t.Fatal("stale decision overwritten")
	}
}

func TestSampledTraceNonFiniteExtent(t *testing.T) {
	for _, tc := range []struct {
		values []float32
		want   string
	}{
		{[]float32{float32(math.NaN()), 1, 2}, "0-0/1/3"},
		{[]float32{float32(math.NaN()), float32(math.Inf(1)), 2}, "0-1/2/3"},
	} {
		dir := t.TempDir()
		err := writeSampledDecision(dir, 0, 0, 1, 2, false, []int{7}, tc.values)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("error=%v, want %s", err, tc.want)
		}
		entries, readErr := os.ReadDir(dir)
		if readErr != nil || len(entries) != 0 {
			t.Fatalf("failed admission wrote artifacts: %v %v", entries, readErr)
		}
	}
}

func TestSampledTraceRejectsArgmaxObservation(t *testing.T) {
	_, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-dump-sampled-trace-dir", "out", "-dump-logprobs", "log.json"}, strings.NewReader(""), new(bytes.Buffer))
	if err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("error=%v", err)
	}
}

func TestSampledTraceDoesNotConsumeRNG(t *testing.T) {
	logits := []float32{1, 2, 3}
	a, b := sampler.RNG{State: 12345}, sampler.RNG{State: 12345}
	s := sampler.New(3)
	before := a.State
	id := s.Sample(logits, sampler.Default, &a)
	if err := writeSampledDecision(t.TempDir(), 0, id, before, a.State, false, []int{7}, logits); err != nil {
		t.Fatal(err)
	}
	want := s.Sample(logits, sampler.Default, &b)
	if id != want || a.State != b.State {
		t.Fatalf("trace perturbed sample: %d/%d RNG %d/%d", id, want, a.State, b.State)
	}
}
