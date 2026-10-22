package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

func TestSampleTranscriptObserverInvariant(t *testing.T) {
	logits := []float32{1, 2, 3}
	original := append([]float32(nil), logits...)
	a, b := sampler.RNG{State: 12345}, sampler.RNG{State: 12345}
	s := sampler.New(len(logits))
	x, err := newSampleTranscript(filepath.Join(t.TempDir(), "decisions.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	for step := 0; step < 3; step++ {
		before := a.State
		token := s.Sample(logits, sampler.Default, &a)
		if err := x.Decision(step, token, before, a.State, step == 2); err != nil {
			t.Fatal(err)
		}
		if err := writeSampledDecision(t.TempDir(), step, token, before, a.State, step == 2, []int{7}, logits); err != nil {
			t.Fatal(err)
		}
		want := s.Sample(logits, sampler.Default, &b)
		if token != want || a.State != b.State {
			t.Fatalf("step %d: observation changed sampler", step)
		}
		for i := range logits {
			if logits[i] != original[i] {
				t.Fatal("observation changed logits")
			}
		}
	}
}

func TestSampleTranscriptAdmission(t *testing.T) {
	for _, extra := range [][]string{{"-dump-logprobs", "log"}, {"-interactive"}} {
		args := append([]string{"-model", "m", "-backend", "cpu", "-dump-sampled-transcript", "out"}, extra...)
		if _, err := parseConfig(args, strings.NewReader(""), new(bytes.Buffer)); err == nil {
			t.Fatalf("accepted conflicting options %v", extra)
		}
	}
	cfg, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-prompt", "hello", "-dump-sampled-transcript", "out"}, strings.NewReader(""), new(bytes.Buffer))
	if err != nil || cfg.sampledTranscript != "out" {
		t.Fatalf("transcript flag: %q %v", cfg.sampledTranscript, err)
	}
}

func TestSampleTranscriptDisabled(t *testing.T) {
	x, err := newSampleTranscript("")
	if err != nil || x != nil {
		t.Fatalf("disabled: %v %v", x, err)
	}
	if err := x.Decision(-1, -1, 0, 0, false); err != nil {
		t.Fatal(err)
	}
	if err := x.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSampleTranscriptDecisions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "decisions.jsonl")
	x, err := newSampleTranscript(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := newSampleTranscript(path); err == nil {
		t.Fatal("accepted stale transcript")
	}
	if err := x.Decision(1, 7, 12345, 99, false); err == nil {
		t.Fatal("accepted reordered row")
	}
	if err := x.Decision(0, 7, 12345, 99, false); err != nil {
		t.Fatal(err)
	}
	if err := x.Decision(1, 8, 99, 101, true); err != nil {
		t.Fatal(err)
	}
	if err := x.Decision(2, 9, 101, 102, false); err == nil {
		t.Fatal("accepted decision after stop")
	}
	if err := x.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	want := "{\"step\":0,\"token\":7,\"rng_before\":12345,\"rng_after\":99,\"stop\":false}\n{\"step\":1,\"token\":8,\"rng_before\":99,\"rng_after\":101,\"stop\":true}\n"
	if string(data) != want {
		t.Fatalf("transcript = %s, want %s", data, want)
	}
}

func TestSampleTranscriptWriteFailure(t *testing.T) {
	x, err := newSampleTranscript(filepath.Join(t.TempDir(), "decisions.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if err := x.file.Close(); err != nil {
		t.Fatal(err)
	}
	if err := x.Decision(0, 7, 12345, 99, false); err == nil {
		t.Fatal("ignored write failure")
	}
	if x.next != 0 || x.stopped {
		t.Fatal("advanced failed observation")
	}
}
