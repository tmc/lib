package main

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestFrontierCLIRequiresBuildAndProvenance(t *testing.T) {
	for _, args := range [][]string{nil, {"-model", "missing", "-prompt-tokens", "missing", "-receipt", "missing", "-revision", "invented"}} {
		var out, log bytes.Buffer
		err := runFrontierBenchmark(context.Background(), args, &out, &log)
		if err == nil || out.Len() != 0 {
			t.Fatal("incomplete or invented provenance admitted")
		}
	}
}

func TestFrontierMemoryAvailability(t *testing.T) {
	got := frontierMemoryAvailability()
	if len(got) != 6 {
		t.Fatal("incomplete memory availability")
	}
	for _, value := range got {
		if !strings.HasPrefix(value, "UNREAD:") {
			t.Fatal("unknown memory presented as measured")
		}
	}
}
