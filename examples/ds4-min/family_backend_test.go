package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestFamilySelection(t *testing.T) {
	for _, tt := range []struct {
		name, family, backend string
		wantErr               bool
	}{
		{"cpu", "v41", "cpu", false},
		{"cuda", "v41", "cuda", true},
		{"metal", "v41", "metal", true},
		{"auto", "v41", "auto", true},
		{"flash", "flash", "metal", false},
		{"unknown", "unknown", "cpu", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := validateFamilySelection(config{family: tt.family, backend: tt.backend, context: 4})
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestFamilyCLIRefusal(t *testing.T) {
	for _, backend := range []string{"auto", "cuda", "metal"} {
		_, err := parseConfig([]string{"-model", "absent", "-family", "v41", "-backend", backend, "-prompt", "x"}, strings.NewReader(""), &bytes.Buffer{})
		if err == nil {
			t.Fatalf("%s admitted V4.1 GPU", backend)
		}
	}
	g := &gguf.File{Metadata: map[string]gguf.Value{"general.architecture": {TypeID: gguf.TypeString, Value: "deepseek41"}}}
	for _, cfg := range []config{{family: "flash", backend: "cpu"}, {family: "auto", backend: "metal"}} {
		if err := validateFamilyProfile(&cfg, g); err == nil {
			t.Fatal("family mismatch or GPU admitted")
		}
	}
}

func TestV41FileOwnerRetry(t *testing.T) {
	p := &failingV41File{closeErr: errors.New("release failed")}
	b := &v41CPUBackend{file: p}
	if b.Close() == nil || b.file == nil {
		t.Fatal("lost failed file owner")
	}
	p.closeErr = nil
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil || p.closes != 2 {
		t.Fatalf("close = %v, calls = %d", err, p.closes)
	}
}

func TestV41StreamingRefusal(t *testing.T) {
	for _, family := range []string{"auto", "v41"} {
		t.Run(family, func(t *testing.T) {
			args := []string{"-model", "must-not-open", "-backend", "cpu", "-family", family, "-ssd-streaming", "-prompt", "x"}
			err := run(args, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{})
			if err == nil || !strings.Contains(err.Error(), "streaming") || strings.Contains(err.Error(), "open model") {
				t.Fatalf("CPU streaming route = %v", err)
			}
		})
	}
	backend, closer, err := openSessionBackend(config{family: "v41", backend: "cpu", model: "must-not-open", context: 4, streaming: streamingOptions{enabled: true}})
	if backend != nil || closer != nil || err == nil || !strings.Contains(err.Error(), "V4.1 streaming") {
		t.Fatalf("V4.1 streaming open = %v, %v, %v", backend, closer, err)
	}
}
