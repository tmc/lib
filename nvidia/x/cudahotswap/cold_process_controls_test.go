package cudahotswap

import (
	"context"
	"reflect"
	"testing"
	"time"
)

func TestPublicColdCommand(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	command := publicColdCommand(ctx, "/owned/kernelharness", "/owned/candidate.cu")
	want := []string{"/owned/kernelharness", "-task", "axpby", "-kernel", "/owned/candidate.cu", "-json"}
	if !reflect.DeepEqual(command.Args, want) {
		t.Fatalf("args = %q, want %q", command.Args, want)
	}
	if command.SysProcAttr != nil {
		t.Fatal("cold harness must inherit the admitted native process group")
	}
	if command.Cancel == nil || command.WaitDelay != 2*time.Second {
		t.Fatal("cold harness lacks bounded cancellation")
	}
}
