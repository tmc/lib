package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4eval"
)

type evalArtifactRuntime struct{ step int }

func (m *evalArtifactRuntime) Render(string, string) ([]int, error) { return []int{0}, nil }
func (m *evalArtifactRuntime) Sync([]int) error                     { m.step = 0; return nil }
func (m *evalArtifactRuntime) Eval(int) error                       { m.step++; return nil }
func (m *evalArtifactRuntime) Logits() []float32 {
	v := make([]float32, 4)
	i := m.step + 1
	if i > 3 {
		i = 3
	}
	v[i] = 10
	return v
}
func (*evalArtifactRuntime) IsStop(id int) bool { return id == 3 }
func (*evalArtifactRuntime) ThinkClose() []int  { return []int{1} }
func (*evalArtifactRuntime) Text(id int) (string, error) {
	if id == 1 {
		return "</think>", nil
	}
	if id == 2 {
		return "Answer: A", nil
	}
	return "", errors.New("unexpected token")
}

func evalArtifactCases(t *testing.T) []dsv4eval.Case {
	t.Helper()
	cs, err := dsv4eval.SelectN(dsv4eval.Cases(), dsv4eval.Filter{Suite: dsv4eval.HardSmoke}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := range cs {
		cs[i].Question = "unit fixture"
		cs[i].Answer = "A"
		cs[i].Kind = dsv4eval.Choice
		cs[i].Choices = []string{"one", "two"}
		cs[i].MaxTokens = 8
	}
	return cs
}

func evalArtifactsArgs(dir string) []string {
	return []string{"-model", "missing-model", "-eval-suite", "hard-smoke", "-eval-trace", "trace", "-eval-corpus", "external-corpus", "-temp", "0", "-ctx", "32", "-eval-artifacts-dir", dir}
}

func TestEvalArtifactsFlagToCapture(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "new")
	cfg, err := parseConfig(evalArtifactsArgs(dir), strings.NewReader(""), io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if err := admitEvalArtifacts(cfg); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := runEvalCases(&out, new(evalArtifactRuntime), evalArtifactCases(t), dsv4eval.RunOptions{Build: "unit", ModelSHA256: strings.Repeat("a", 64), Context: 32, Soft: 3, Hard: 2, Rank: 3}, cfg); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "case-01", "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		History []int  `json:"history_ids"`
		Stop    string `json:"stop_disposition"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if len(manifest.History) != 3 || manifest.Stop != "selected_stop_not_consumed" {
		t.Fatalf("manifest=%+v", manifest)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 12 {
		t.Fatalf("cases=%d err=%v", len(entries), err)
	}
}

func TestEvalArtifactsAdmissionBeforeModel(t *testing.T) {
	dir := t.TempDir()
	err := run(evalArtifactsArgs(dir), strings.NewReader(""), io.Discard, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "admit new artifacts directory") || strings.Contains(err.Error(), "open model") {
		t.Fatalf("model opened before exclusive admission: %v", err)
	}
}

func TestEvalTraceAdmissionBeforeModel(t *testing.T) {
	root := t.TempDir()
	trace := filepath.Join(root, "trace")
	if err := os.WriteFile(trace, []byte("retained"), 0600); err != nil {
		t.Fatal(err)
	}
	args := evalArtifactsArgs(filepath.Join(root, "new"))
	args = append(args, "-eval-trace", trace)
	err := run(args, strings.NewReader(""), io.Discard, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "admit new trace") || strings.Contains(err.Error(), "open model") {
		t.Fatalf("trace collision reached model: %v", err)
	}
	b, readErr := os.ReadFile(trace)
	if readErr != nil || string(b) != "retained" {
		t.Fatal("retained trace overwritten")
	}
}

func TestEvalArtifactsFlagConflicts(t *testing.T) {
	for _, extra := range [][]string{{"-prompt", "x"}, {"-interactive"}, {"-dump-sampled-trace-dir", "x"}, {"-temp", "1"}, {"-eval-suite", "core"}} {
		args := append(evalArtifactsArgs(filepath.Join(t.TempDir(), "new")), extra...)
		if _, err := parseConfig(args, strings.NewReader(""), io.Discard); err == nil {
			t.Fatalf("accepted %v", extra)
		}
	}
	if _, err := parseConfig([]string{"-eval-artifacts-dir", "x"}, strings.NewReader(""), io.Discard); err == nil {
		t.Fatal("accepted artifacts without suite")
	}
}

func TestEvalArtifactsPartialCaseStopsTrace(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "case-01"), 0700); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	err := runEvalCases(&out, new(evalArtifactRuntime), evalArtifactCases(t), dsv4eval.RunOptions{Build: "unit", ModelSHA256: strings.Repeat("a", 64), Context: 32, Soft: 3, Hard: 2, Rank: 3}, config{evalArtifactsDir: dir})
	if err == nil || strings.Contains(out.String(), "===== SUMMARY") {
		t.Fatalf("partial artifacts qualified: %v", err)
	}
}
