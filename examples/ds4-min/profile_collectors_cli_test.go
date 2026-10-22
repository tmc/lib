package main

import (
	"errors"
	"io"
	"reflect"
	"testing"
)

type collectorNoRead struct{}

func (collectorNoRead) Read([]byte) (int, error) { return 0, errors.New("stdin must not be read") }

func TestCalibrationDoesNotReadOrdinaryStdin(t *testing.T) {
	cfg, err := parseConfig([]string{"-model", "model", "-backend", "cpu", "-imatrix-dataset", "dataset", "-imatrix-out", "output"}, collectorNoRead{}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.collectors.dataset != "dataset" || cfg.collectors.imatrix != "output" {
		t.Fatal("collector flags not retained")
	}
	if _, err := parseConfig([]string{"-model", "model", "-backend", "cpu", "-imatrix-dataset", "dataset", "-imatrix-out", "output", "-prompt", "conflict"}, collectorNoRead{}, io.Discard); err == nil {
		t.Fatal("calibration ignored conflicting prompt")
	}
	for _, flags := range [][]string{{"-tokens", "5"}, {"-trace", "ignored"}, {"-raw"}, {"-dump-logits", "ignored"}} {
		args := append([]string{"-model", "model", "-backend", "cpu", "-imatrix-dataset", "dataset", "-imatrix-out", "output"}, flags...)
		if _, err := parseConfig(args, collectorNoRead{}, io.Discard); err == nil {
			t.Fatalf("ignored flags admitted: %q", flags)
		}
	}
}

func TestCalibrationPrompts(t *testing.T) {
	for _, tt := range []struct {
		name, text string
		want       []string
	}{
		{"plain", " one \n", []string{"one"}},
		{"non-ascii-space", "\u00a0one\u00a0", []string{"\u00a0one\u00a0"}},
		{"marked", "ignored\n===== DS4_IMATRIX_PROMPT 0\none\n===== DS4_IMATRIX_PROMPT 1\ntwo\n", []string{"one", "two"}},
		{"embedded", "one ===== DS4_IMATRIX_PROMPT 1\ntwo", []string{"one ===== DS4_IMATRIX_PROMPT 1\ntwo"}},
		{"terminal", "===== DS4_IMATRIX_PROMPT 0", []string{""}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := calibrationPrompts(tt.text); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %q want %q", got, tt.want)
			}
		})
	}
}

func TestCollectorModeAdmission(t *testing.T) {
	good := config{backend: "cuda", collectors: collectorOptions{dataset: "dataset", imatrix: "output", maxBytes: 1 << 30}}
	if err := validateCollectorOptions(good); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*config){
		func(c *config) { c.collectors.imatrix = "" },
		func(c *config) { c.collectors.expertProfile = "profile" },
		func(c *config) { c.backend = "cpu" },
		func(c *config) { c.evalSuite = "hard-smoke" },
		func(c *config) { c.warmPrefill = true },
		func(c *config) { c.collectors.maxBytes = 0 },
	} {
		c := good
		mutate(&c)
		if err := validateCollectorOptions(c); err == nil {
			t.Fatal("invalid mode accepted")
		}
	}
}
