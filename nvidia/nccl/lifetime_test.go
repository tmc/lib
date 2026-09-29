//go:build darwin || linux

package nccl

import (
	"runtime"
	"testing"
)

// stubDeferredOut mimics NCCL's group semantics: group depth is per thread,
// and inside a group the new handle is stored by the job GroupEnd runs,
// after the creating call has returned (src/init.cc ncclCommInitChildComm,
// src/dev_runtime.cc ncclCommWindowRegister).
const stubDeferredOut = `
#include <stddef.h>
#include <stdint.h>
static _Thread_local int depth;
static uintptr_t *pending[8];
static uintptr_t values[8];
static int npending;
static int defer(uintptr_t *out, uintptr_t v) {
	*out = 0;
	if (depth == 0) { *out = v; return 0; }
	if (npending == 8) return 1;
	pending[npending] = out;
	values[npending++] = v;
	return 0;
}
int ncclGroupStart(void) { depth++; return 0; }
int ncclGroupEnd(void) {
	if (depth == 0) return 5;
	if (--depth == 0) {
		for (int i = 0; i < npending; i++) *pending[i] = values[i];
		npending = 0;
	}
	return 0;
}
int ncclCommSplit(uintptr_t comm, int color, int key, uintptr_t *newcomm, void *config) {
	return defer(newcomm, comm*100 + color*10 + key);
}
int ncclCommWindowRegister(uintptr_t comm, void *buff, size_t size, uintptr_t *win, int flags) {
	return defer(win, comm + (uintptr_t)buff);
}
`

// TestDeferredOut checks that a handle NCCL stores after the creating call
// returns lands in the caller's variable, which stays valid across
// collections until GroupEnd.
func TestDeferredOut(t *testing.T) {
	bindStub(t, stubDeferredOut, map[string]*uintptr{
		"ncclGroupStart":         &symGroupStart,
		"ncclGroupEnd":           &symGroupEnd,
		"ncclCommSplit":          &symcommSplit,
		"ncclCommWindowRegister": &symcommWindowRegister,
	})
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Outside a group the handle is set on return.
	var c Comm
	if err := Comm(7).Split(1, 2, &c, nil); err != nil {
		t.Fatal(err)
	}
	if c != 712 {
		t.Fatalf("Split = %d, want 712", c)
	}

	if err := GroupStart(); err != nil {
		t.Fatal(err)
	}
	comms := make([]*Comm, 4)
	wins := make([]*Window, 4)
	for i := range comms {
		comms[i], wins[i] = new(Comm), new(Window)
		if err := Comm(i+1).Split(0, int32(i), comms[i], nil); err != nil {
			t.Fatal(err)
		}
		if err := Comm(i+1).WindowRegister(0x1000, 64, wins[i], 0); err != nil {
			t.Fatal(err)
		}
		if *comms[i] != 0 || *wins[i] != 0 {
			t.Fatalf("handle %d set before GroupEnd", i)
		}
	}
	for range 3 {
		runtime.GC()
		_ = make([]byte, 1<<20)
	}
	if err := GroupEnd(); err != nil {
		t.Fatal(err)
	}
	for i := range comms {
		if want := Comm((i+1)*100 + i); *comms[i] != want {
			t.Errorf("comm %d = %d, want %d", i, *comms[i], want)
		}
		if want := Window(i + 1 + 0x1000); *wins[i] != want {
			t.Errorf("window %d = %#x, want %#x", i, *wins[i], want)
		}
	}
}
