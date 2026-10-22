//go:build linux

package main

import (
	"errors"
	"io"
	"os"
	"testing"

	"github.com/tmc/lib/gguf"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestCUDAFoundationConstructorCleanup(t *testing.T) {
	for _, stage := range []string{"open", "stat", "read", "validate", "map", "device", "describe", "stream", "warm", "probe", "probe-model"} {
		t.Run(stage, func(t *testing.T) {
			f := newFoundationCloseFixture(t)
			file, direct, dev, stream := f.r.file, f.r.direct, f.r.device, f.r.workStream
			info, err := file.Stat()
			if err != nil {
				t.Fatal(err)
			}
			cause := errors.New("construct fault")
			fault := func(s string) error {
				if s == stage {
					return cause
				}
				return nil
			}
			ops := cudaFoundationConstructorOperations{
				open:       func(string) (*os.File, error) { return file, fault("open") },
				stat:       func(*os.File) (os.FileInfo, error) { return info, fault("stat") },
				direct:     func(string, os.FileInfo) (*os.File, int) { return direct, 4096 },
				read:       func(*os.File, int64) (*gguf.File, error) { return new(gguf.File), fault("read") },
				validate:   func(*gguf.File, *os.File, int64) error { return fault("validate") },
				mapModel:   func(*os.File, int64) ([]byte, error) { return []byte{1}, fault("map") },
				device:     func() (*cuda.Device, error) { return dev, fault("device") },
				describe:   func(*cudaFoundation) error { return fault("describe") },
				stream:     func(*cuda.Device) (*cuda.Stream, error) { return stream, fault("stream") },
				warm:       func(*cudaFoundation, cudaRuntimeOptions) error { return fault("warm") },
				probe:      func(*cudaFoundation) error { return fault("probe") },
				probeModel: func(*cudaFoundation) error { return fault("probe-model") },
				close:      f.r.closeOps,
			}
			owner, err := newCUDAFoundationWithOperations("fixture", cudaRuntimeOptions{}, ops)
			if owner != nil || !errors.Is(err, cause) {
				t.Fatalf("owner/error = %v/%v", owner, err)
			}
			if f.calls["file"] != 1 {
				t.Fatalf("file releases = %d", f.calls["file"])
			}
			if stage == "stream" && f.calls["stream"] != 1 {
				t.Fatalf("partial stream releases = %d", f.calls["stream"])
			}
			if stage == "device" && f.calls["device"] != 1 {
				t.Fatalf("partial device releases = %d", f.calls["device"])
			}
		})
	}
}

func TestCUDAFoundationConstructorRetainsCleanup(t *testing.T) {
	f := newFoundationCloseFixture(t)
	file, dev, stream := f.r.file, f.r.device, f.r.workStream
	info, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	cause := errors.New("stream context restore fault")
	f.fail = "stream"
	ops := cudaFoundationConstructorOperations{
		open:     func(string) (*os.File, error) { return file, nil },
		stat:     func(*os.File) (os.FileInfo, error) { return info, nil },
		direct:   func(string, os.FileInfo) (*os.File, int) { return nil, 0 },
		read:     func(*os.File, int64) (*gguf.File, error) { return new(gguf.File), nil },
		validate: func(*gguf.File, *os.File, int64) error { return nil },
		mapModel: func(*os.File, int64) ([]byte, error) { return []byte{1}, nil },
		device:   func() (*cuda.Device, error) { return dev, nil },
		describe: func(*cudaFoundation) error { return nil },
		stream:   func(*cuda.Device) (*cuda.Stream, error) { return stream, cause },
		close:    f.r.closeOps,
	}
	_, err = newCUDAFoundationWithOperations("fixture", cudaRuntimeOptions{}, ops)
	var cleanup io.Closer
	if !errors.Is(err, cause) || !errors.Is(err, f.cause) || !errors.As(err, &cleanup) {
		t.Fatalf("missing retained errors/owner: %v", err)
	}
	if f.calls["device"] != 0 || f.calls["model"] != 0 || f.calls["file"] != 0 {
		t.Fatalf("dependent owners released: %v", f.calls)
	}
	f.fail = ""
	if err := cleanup.Close(); err != nil {
		t.Fatal(err)
	}
	if f.calls["stream"] != 2 || f.calls["device"] != 1 || f.calls["model"] != 1 || f.calls["file"] != 1 {
		t.Fatalf("retry releases: %v", f.calls)
	}
}
