package xla_test

import (
	"context"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func TestEngine(t *testing.T) {
	// Without a real plugin, we can't fully instantiate an engine.
	// But we can verify the API surface builds and error handling works.
	_, err := xla.NewEngine("/invalid/path/to/plugin.so")
	if err == nil {
		t.Errorf("expected error for invalid plugin path, got nil")
	}
}

func TestCompileOptions(t *testing.T) {
	opts := xla.DefaultCompileOptions()
	b := opts.ToProto()
	// replica_count=1, num_partitions=1 produces build options + top level replica/partition fields
	want := []byte{0x1a, 0x04, 0x20, 0x01, 0x28, 0x01, 0x20, 0x01, 0x28, 0x01}
	if string(b) != string(want) {
		t.Errorf("ToProto() = %x, want %x", b, want)
	}
}

func TestBufferWriteTo(t *testing.T) {
	// A pure interface test; in a real scenario we'd create a Buffer from an Engine.
	b := &xla.Buffer{}
	// Calling WriteTo on a zero Buffer should return an error since element type is unsupported (or ToHost fails).
	// We just ensure the method exists and has the right signature.
	_, err := b.WriteTo(nil)
	if err == nil {
		t.Errorf("expected error writing nil buffer")
	}
}

func TestExecutable(t *testing.T) {
	exec := &xla.Executable{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel context

	_, err := exec.Execute(ctx, nil)
	if err == nil {
		t.Errorf("expected error for cancelled context")
	}
}
