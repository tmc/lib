package cudaprof_test

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func TestProfileRejectsInferredInstructions(t *testing.T) {
	for _, existing := range []bool{false, true} {
		path := filepath.Join(t.TempDir(), "kernel.so")
		if existing {
			if err := os.WriteFile(path, []byte("retained code"), 0600); err != nil {
				t.Fatal(err)
			}
		}
		var out bytes.Buffer
		s := cudaprof.NewSession(&out)
		s.SetBinaryPath(path)
		s.RecordSample(context.Background(), "RMSNorm", "kernel.cu", 1, cudaprof.DefaultMappingStart+64, 1)
		if err := s.Close(); err == nil {
			t.Fatal("generated instructions accepted")
		}
		if out.Len() != 0 {
			t.Fatal("profile written despite unavailable code")
		}
		data, err := os.ReadFile(path)
		if existing {
			if err != nil || string(data) != "retained code" {
				t.Fatal("overwrote retained code")
			}
		} else if !os.IsNotExist(err) {
			t.Fatal("created inferred code image")
		}
		if err := cudaprof.Start(&out, cudaprof.WithBinaryPath(path)); err == nil {
			t.Fatal("global start accepted generated code")
		}
		fr := cudaprof.NewFlightRecorder(cudaprof.WithFlightRecorderBinaryPath(path))
		if err := fr.Start(); err == nil || fr.Enabled() {
			t.Fatal("flight recorder accepted generated code")
		}
		if n, err := fr.WriteTo(&out); err == nil || n != 0 {
			t.Fatal("flight recorder wrote generated code")
		}
	}
}

func TestProfileRetainsExplicitCodeMapping(t *testing.T) {
	var out bytes.Buffer
	s := cudaprof.NewSession(&out)
	s.SetMappingFile("retained.cubin", "native-image-hash")
	s.RecordSample(context.Background(), "RMSNorm", "kernel.cu", 1, cudaprof.DefaultMappingStart+64, 1)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	p, err := profile.ParseData(out.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Mapping) != 1 || p.Mapping[0].File != "retained.cubin" || p.Mapping[0].BuildID != "native-image-hash" {
		t.Fatalf("mapping=%v", p.Mapping)
	}
}
