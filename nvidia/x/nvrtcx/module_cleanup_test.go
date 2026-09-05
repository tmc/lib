package nvrtcx

import (
	"errors"
	"github.com/tmc/lib/nvidia/driver"
	"io"
	"testing"
)

func TestModuleLoadCleanup(t *testing.T) {
	restore := errors.New("restore context")
	unload := errors.New("unload")
	calls := 0
	mod, err := finishModuleLoad(1, restore, func() error {
		calls++
		if calls < 2 {
			return unload
		}
		return nil
	})
	if mod != 0 || !errors.Is(err, restore) || !errors.Is(err, unload) {
		t.Fatalf("module=%v err=%v", mod, err)
	}
	var closer io.Closer
	if !errors.As(err, &closer) {
		t.Fatal("lost cleanup ownership")
	}
	if err := closer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := closer.Close(); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("unload calls=%d, want 2", calls)
	}
}

func TestModuleLoadCleanupPaths(t *testing.T) {
	failure := errors.New("load or restore")
	for _, tt := range []struct {
		name               string
		module, wantModule uintptr
		err                error
		wantCalls          int
	}{
		{"success", 1, 1, nil, 0},
		{"load_failure", 0, 0, failure, 0},
		{"restore_failure", 1, 0, failure, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			got, err := finishModuleLoad(driver.Module(tt.module), tt.err, func() error { calls++; return nil })
			if uintptr(got) != tt.wantModule || !errors.Is(err, tt.err) || calls != tt.wantCalls {
				t.Fatalf("module=%v err=%v calls=%d", got, err, calls)
			}
		})
	}
}
