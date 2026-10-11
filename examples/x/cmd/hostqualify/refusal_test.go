package main

import (
	"bytes"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestNonfiniteInputs(t *testing.T) {
	for _, flag := range []string{"-max-load", "-tolerance", "-prefill", "-gen"} {
		for _, value := range []string{"NaN", "+Inf", "-Inf"} {
			t.Run(flag+value, func(t *testing.T) {
				if _, err := parseConfig([]string{flag, value}, io.Discard); err == nil {
					t.Fatal("invalid number accepted")
				}
			})
		}
	}
	for _, value := range []string{"NaN", "+Inf", "-1"} {
		path := filepath.Join(t.TempDir(), "reference.csv")
		if err := os.WriteFile(path, []byte("ctx_tokens,prefill_tps,gen_tps\n1,"+value+",1\n"), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := readReference(path, 1); err == nil {
			t.Fatal("invalid reference accepted")
		}
	}
	for _, value := range []float64{math.NaN(), math.Inf(1), -1} {
		restore := swap(&readLoad, func() (float64, error) { return value, nil })
		err := baseline(config{maxLoad: 10, samples: 1}, io.Discard)
		restore()
		if err == nil {
			t.Fatal("invalid observed load accepted")
		}
	}
}

type refusalWriter struct{ err error }

func (w refusalWriter) Write([]byte) (int, error) { return 0, w.err }

func TestReportWriteFailure(t *testing.T) {
	defer swap(&readLoad, func() (float64, error) { return 0, nil })()
	failure := errors.New("write failed")
	for _, want := range []error{failure, io.ErrShortWrite} {
		writer := refusalWriter{err: failure}
		if want == io.ErrShortWrite {
			writer.err = nil
		}
		if err := run([]string{"-baseline", "-samples", "1"}, writer, &bytes.Buffer{}); !errors.Is(err, want) {
			t.Fatalf("error %v, want %v", err, want)
		}
	}
}
