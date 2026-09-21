package main

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// activityCapture is the profile shape BuildActivityProfileIndex emits: three
// quantities in three units, with the duration marked default.
func activityCapture() *cudacapture.Capture {
	return &cudacapture.Capture{Profile: &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "kernel_time", Unit: "nanoseconds"},
			{Type: "copy_time", Unit: "nanoseconds"},
			{Type: "copy_bytes", Unit: "bytes"},
			{Type: "activity", Unit: "count"},
		},
		DefaultSampleType: "kernel_time",
	}}
}

// TestUnknownSampleIndexNamesTheUnits is 10.6(i). pprof already lists the
// names; what it cannot say is which name carries time, and activity_ns was
// someone asking for a unit rather than for a name.
func TestUnknownSampleIndexNamesTheUnits(t *testing.T) {
	var w bytes.Buffer
	err := preflightSampleIndex(&w, activityCapture(), []string{"-top", "-sample_index=activity_ns"})
	if err == nil {
		t.Fatal("accepted a sample index this capture does not measure")
	}
	for _, want := range []string{"kernel_time (nanoseconds)", "copy_bytes (bytes)", "activity (count)", "[default]"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("rejection does not state %q: %v", want, err)
		}
	}
}

// TestSampleIndexCheckedInBothFlagForms guards the way the refusal can be
// silently bypassed: pprof parses the flag with the flag package, so the
// space-separated spelling is as valid as the joined one.
func TestSampleIndexCheckedInBothFlagForms(t *testing.T) {
	var w bytes.Buffer
	for _, args := range [][]string{
		{"-sample_index", "activity_ns"},
		{"--sample_index=activity_ns"},
		{"-top", "-sample_index", "activity_ns", "cap.pprof"},
	} {
		if err := preflightSampleIndex(&w, activityCapture(), args); err == nil {
			t.Errorf("args %q were not checked", args)
		}
	}
}

// TestValidSampleIndexAccepted keeps the refusal from rejecting everything,
// and mirrors what pprof's own SampleIndexByName accepts: a type name or a
// numeric index in range.
func TestValidSampleIndexAccepted(t *testing.T) {
	var w bytes.Buffer
	for _, value := range []string{"kernel_time", "copy_time", "copy_bytes", "activity", "0", "3"} {
		if err := preflightSampleIndex(&w, activityCapture(), []string{"-sample_index=" + value}); err != nil {
			t.Errorf("refused %q, which pprof accepts: %v", value, err)
		}
	}
	if err := preflightSampleIndex(&w, activityCapture(), []string{"-sample_index=4"}); err == nil {
		t.Error("accepted an index outside the range pprof would take")
	}
	if err := preflightSampleIndex(&w, activityCapture(), nil); err != nil {
		t.Errorf("refused a command line with no sample index: %v", err)
	}
}

// TestNonTimeSampleIndexDisownsTheCoverageLine is the denominator half: the
// coverage statement printed directly above is about device time, and a table
// ranked by bytes or launches is not what it describes.
func TestNonTimeSampleIndexDisownsTheCoverageLine(t *testing.T) {
	var w bytes.Buffer
	if err := preflightSampleIndex(&w, activityCapture(), []string{"-top", "-sample_index=copy_bytes"}); err != nil {
		t.Fatal(err)
	}
	note := w.String()
	if !strings.Contains(note, "copy_bytes") || !strings.Contains(note, "bytes") {
		t.Errorf("note does not name the selected quantity and its unit: %q", note)
	}
	if !strings.Contains(note, "not device time") {
		t.Errorf("note does not say the coverage statement is not the table's denominator: %q", note)
	}

	w.Reset()
	if err := preflightSampleIndex(&w, activityCapture(), []string{"-top", "-sample_index=kernel_time"}); err != nil {
		t.Fatal(err)
	}
	if w.Len() != 0 {
		t.Errorf("device time selection printed a mismatch note: %q", w.String())
	}
}

// TestSampleIndexPreflightIsWired closes the one mutation the tests above
// cannot see: every assertion in this file calls preflightSampleIndex
// directly, so deleting its call site in run() leaves them all green and the
// check dead. The assertion is over source text because the pprof path it sits
// on ends in driver.PProf, which needs a real profile and a real terminal.
//
// Its denominator is one file, named here: if main.go cannot be read, or no
// longer contains the coverage preflight this one is ordered against, the test
// fails rather than passing on an empty search.
func TestSampleIndexPreflightIsWired(t *testing.T) {
	src, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatalf("read the file the call site lives in: %v", err)
	}
	coverage := bytes.Index(src, []byte("preflightCoverage(os.Stderr"))
	if coverage < 0 {
		t.Fatal("main.go no longer calls preflightCoverage, so this test is checking an ordering that does not exist")
	}
	index := bytes.Index(src, []byte("preflightSampleIndex(os.Stderr"))
	if index < 0 {
		t.Fatal("main.go does not call preflightSampleIndex: the check is unreachable from the command line")
	}
	if index < coverage {
		t.Error("the sample-index check runs before the coverage statement it refers to as printed above")
	}
}
