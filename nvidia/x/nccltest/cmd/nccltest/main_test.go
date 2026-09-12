package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/nccltest"
)

func TestCycleSummary(t *testing.T) {
	failure := errors.New("cleanup failed")
	for _, tt := range []struct {
		name     string
		emitted  int
		runErr   error
		complete bool
	}{
		{"complete", 2, nil, true}, {"missing cycle", 1, nil, false}, {"cleanup failure", 2, failure, false}, {"early failure", 0, failure, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			err := writeCycles(&out, 2, func(yield func(nccltest.Result) error) error {
				for i := 0; i < tt.emitted; i++ {
					if err := yield(nccltest.Result{Comparisons: 8}); err != nil {
						return err
					}
				}
				return tt.runErr
			})
			if (err == nil) != tt.complete {
				t.Fatalf("error %v", err)
			}
			if tt.runErr != nil && !errors.Is(err, failure) {
				t.Fatal("lost cleanup error")
			}
			decoder := json.NewDecoder(&out)
			for i := 0; i < tt.emitted; i++ {
				var row struct {
					Kind  string
					Cycle int
				}
				if err := decoder.Decode(&row); err != nil || row.Kind != "cycle" || row.Cycle != i+1 {
					t.Fatalf("row %+v error %v", row, err)
				}
			}
			var summary struct {
				Kind                 string
				Requested, Completed int
				Complete             bool
				Error                string
			}
			if err := decoder.Decode(&summary); err != nil {
				t.Fatal(err)
			}
			if summary.Kind != "summary" || summary.Requested != 2 || summary.Completed != tt.emitted || summary.Complete != tt.complete || (summary.Error == "") != tt.complete {
				t.Fatalf("summary %+v", summary)
			}
		})
	}
}

func TestSweepRecords(t *testing.T) {
	for _, bad := range []bool{false, true} {
		var out bytes.Buffer
		err := writeSweep(&out, []int{64, 16}, 2, func(yield func(nccltest.Result) error) error {
			for i, n := range []int{64, 16, 64, 16} {
				if bad && i == 2 {
					n = 32
				}
				if err := yield(nccltest.Result{Case: nccltest.Case{Bytes: n}}); err != nil {
					return err
				}
			}
			return nil
		})
		if (err != nil) != bad {
			t.Fatal(err)
		}
		decoder := json.NewDecoder(&out)
		samples := 4
		if bad {
			samples = 2
		}
		for i := 0; i < samples; i++ {
			var row struct {
				Kind             string
				Cycle, SizeIndex int
				Result           nccltest.Result
			}
			if err := decoder.Decode(&row); err != nil {
				t.Fatal(err)
			}
			if row.Kind != "sample" || row.Cycle != i/2+1 || row.SizeIndex != i%2+1 {
				t.Fatalf("row %+v", row)
			}
		}
		var summary struct {
			Complete         bool
			CompletedSamples int
		}
		if err := decoder.Decode(&summary); err != nil {
			t.Fatal(err)
		}
		if summary.Complete == bad || summary.CompletedSamples != samples {
			t.Fatalf("summary %+v", summary)
		}
	}
}

func TestContinuousSweepSummary(t *testing.T) {
	for _, runErr := range []error{context.Canceled, errors.Join(context.Canceled, errors.New("cleanup failed")), nil} {
		var out bytes.Buffer
		err := writeSweep(&out, []int{16, 64}, 0, func(yield func(nccltest.Result) error) error {
			for _, size := range []int{16, 64, 16} {
				if err := yield(nccltest.Result{Case: nccltest.Case{Bytes: size}}); err != nil {
					return err
				}
			}
			return runErr
		})
		if err == nil || (runErr != nil && !errors.Is(err, runErr)) {
			t.Fatalf("error=%v, run error=%v", err, runErr)
		}
		decoder := json.NewDecoder(&out)
		for i := 0; i < 3; i++ {
			var row struct {
				Kind             string
				Cycle, SizeIndex int
			}
			if err := decoder.Decode(&row); err != nil || row.Kind != "sample" || row.Cycle != i/2+1 || row.SizeIndex != i%2+1 {
				t.Fatalf("row=%+v error=%v", row, err)
			}
		}
		var summary struct {
			Kind                                                string
			RequestedCycles, RequestedSamples, CompletedSamples int
			Complete                                            bool
			Error                                               string
		}
		if err := decoder.Decode(&summary); err != nil || summary.Kind != "summary" || summary.Complete || summary.RequestedCycles != 0 || summary.RequestedSamples != 0 || summary.CompletedSamples != 3 || summary.Error == "" {
			t.Fatalf("summary=%+v error=%v", summary, err)
		}
	}
}
