package nccl

import (
	"testing"
	"unsafe"
)

// This checks wrapper storage forwarding, not native communicator execution.
func TestCommInitAllArray(t *testing.T) {
	oldFn, oldPresent, oldReady := fnCommInitAll, present, ready.Load()
	defer func() { fnCommInitAll, present = oldFn, oldPresent; ready.Store(oldReady) }()
	present = map[string]bool{"ncclCommInitAll": true}
	ready.Store(true)
	fnCommInitAll = func(out *Comm, n int32, devices *int32) Result {
		if n != 2 {
			t.Fatalf("count = %d", n)
		}
		devs := unsafe.Slice(devices, n)
		if devs[0] != 3 || devs[1] != 5 {
			t.Fatalf("devices = %v", devs)
		}
		handles := unsafe.Slice(out, n)
		handles[0], handles[1] = Comm(11), Comm(22)
		return Success
	}
	storage := []Comm{99, 0, 0, 99}
	devices := []int32{3, 5}
	if err := CommInitAll(&storage[1], 2, &devices[0]); err != nil {
		t.Fatal(err)
	}
	want := []Comm{99, 11, 22, 99}
	for i := range want {
		if storage[i] != want[i] {
			t.Fatalf("storage = %v, want %v", storage, want)
		}
	}
}
