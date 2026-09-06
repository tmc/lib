//go:build linux && cufile_physical

package cufile_test

import (
	"bytes"
	"fmt"
	"github.com/tmc/lib/nvidia/cufile"
	"github.com/tmc/lib/nvidia/driver"
	"os"
	"runtime"
	"testing"
	"unsafe"
)

// Set CUFILE_PHYSICAL_PERTURB=expected to run the negative control, which
// corrupts one byte of the expected buffer so that a test that is genuinely
// comparing content fails and one that is not does not.
func perturb() bool { return os.Getenv("CUFILE_PHYSICAL_PERTURB") == "expected" }

// TestPhysicalTransfer checks bytes through device memory, not the storage transport.
func TestPhysicalTransfer(t *testing.T) {
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	cleanup := func(f func() error) {
		t.Cleanup(func() {
			if e := f(); e != nil {
				t.Error(e)
			}
		})
	}
	check := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	check(driver.Init(0))
	dev, err := driver.NewDeviceGet(0)
	check(err)
	ctx, err := driver.NewContext(nil, 0, dev)
	check(err)
	cleanup(ctx.Destroy)
	check(cufile.DriverOpen())
	cleanup(cufile.DriverClose)
	const n = 16384
	input := make([]byte, n)
	for i := range input {
		input[i] = byte((i*31 + i/17) % 251)
	}
	want := make([]byte, n)
	copy(want, input)
	if perturb() {
		want[n/2]++
	}
	f, err := os.CreateTemp(t.TempDir(), "cufile-transfer-*")
	check(err)
	cleanup(f.Close)
	_, err = f.Write(input)
	check(err)
	check(f.Sync())
	h, err := cufile.NewHandleRegister(&cufile.Descr{Type: cufile.HandleTypeOpaqueFd, Handle: [1]uint64{uint64(f.Fd())}})
	check(err)
	t.Cleanup(h.Deregister)
	p, err := driver.MemAlloc(n)
	check(err)
	cleanup(p.MemFree)
	gotn := h.Read(uintptr(p), n, 0, 0)
	if gotn != n {
		t.Fatal(fmt.Sprintf("read=%d want=%d", gotn, n))
	}
	got := make([]byte, n)
	check(p.MemcpyDtoH(unsafe.Pointer(&got[0]), n))
	if !bytes.Equal(got, want) {
		t.Fatal("read bytes differ")
	}
	written := h.Write(uintptr(p), n, n, 0)
	if written != n {
		t.Fatal(fmt.Sprintf("write=%d", written))
	}
	check(f.Sync())
	out := make([]byte, n)
	_, err = f.ReadAt(out, n)
	check(err)
	if !bytes.Equal(out, want) {
		t.Fatal("write bytes differ")
	}
	t.Logf("cuFile read and write each matched %d bytes through CUDA device memory; transport mode unqualified\n", n)
}
