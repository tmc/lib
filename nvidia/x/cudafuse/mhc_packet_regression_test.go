package cudafuse

import (
	"context"
	"errors"
	"testing"
)

func TestMHCPacketWrappedUnwrittenOutput(t *testing.T) {
	e := &Execution{complete: true, n: 1, output: &executionBuffer{read: func([]float32) error { return ErrOutputUnwritten }}}
	out, err := e.Readback(context.Background())
	if out != nil || err == nil || err == ErrOutputUnwritten || !errors.Is(err, ErrOutputUnwritten) {
		t.Fatalf("wrapped publication error not preserved: out=%v err=%v", out, err)
	}
	if errors.Is(err, ErrOutputGuard) {
		t.Fatal("unwritten output confused with guard corruption")
	}
}
