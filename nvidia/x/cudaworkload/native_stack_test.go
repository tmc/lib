//go:build linux || darwin

package cudaworkload

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/cupti"
)

type nativeStack struct {
	push func(uint64) int32
	pop  func() uint64
}

func (h nativeStack) Push(_ cupti.ExternalCorrelationKind, id uint64) error {
	if h.push(id) != 0 {
		return fmt.Errorf("stack overflow")
	}
	return nil
}
func (h nativeStack) Pop(_ cupti.ExternalCorrelationKind) (uint64, error) { return h.pop(), nil }

// This fixture exercises actual native TLS through the public pinned producer,
// independently of the Go test hook. It does not qualify NVIDIA CUPTI.
func TestCompiledThreadStack(t *testing.T) {
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("C compiler unavailable")
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "stack.c")
	lib := filepath.Join(dir, "stack.so")
	code := `#include <stdint.h>
static _Thread_local uint64_t ids[32];
static _Thread_local unsigned n;
int push(uint64_t id) { if(n==32)return 1;ids[n++]=id;return 0; }
uint64_t pop(void) { return n ? ids[--n] : 0; }
uint64_t top(void) { return n ? ids[n-1] : 0; }
`
	if err := os.WriteFile(src, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	args := []string{"-std=c11", "-shared", "-fPIC", "-o", lib, src}
	if runtime.GOOS == "darwin" {
		args[1] = "-dynamiclib"
	}
	if out, err := exec.Command(cc, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, out)
	}
	handle, err := purego.Dlopen(lib, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		t.Fatal(err)
	}
	defer purego.Dlclose(handle)
	var h nativeStack
	var top func() uint64
	purego.RegisterLibFunc(&h.push, handle, "push")
	purego.RegisterLibFunc(&h.pop, handle, "pop")
	purego.RegisterLibFunc(&top, handle, "top")
	a := NewAnnotator(ProcessGenID{1, 2}, "fixture", WithRunID("run"), WithCorrelationHook(h))
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, parent := a.Begin(context.Background(), "same")
			_, child := a.Begin(ctx, "same")
			_, err := parent.NativeOperation(ctx, ContextStreamID{}, "native", "fixture", func() error {
				if top() != uint64(parent.ID()) {
					return fmt.Errorf("parent stack mismatch")
				}
				runtime.Gosched()
				_, err := child.NativeOperation(ctx, ContextStreamID{}, "native", "fixture", func() error {
					if top() != uint64(child.ID()) {
						return fmt.Errorf("child stack mismatch")
					}
					return context.Canceled
				})
				if err != context.Canceled {
					return fmt.Errorf("child cancellation: %v", err)
				}
				if top() != uint64(parent.ID()) {
					return fmt.Errorf("parent not restored")
				}
				return nil
			})
			if err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if top() != 0 {
		t.Fatal("retained native stack")
	}
}
