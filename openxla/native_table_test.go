package openxla

import (
	"syscall"
	"testing"
)

// nativeTableBytes puts a table immediately before an inaccessible page. An
// overread faults instead of silently reading a neighboring Go allocation.
func nativeTableBytes(t *testing.T, size uintptr) []byte {
	t.Helper()
	page := syscall.Getpagesize()
	if size > uintptr(page) {
		t.Fatalf("table size %d exceeds page size %d", size, page)
	}
	region, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := syscall.Munmap(region); err != nil {
			t.Error(err)
		}
	})
	if err := syscall.Mprotect(region[page:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	return region[page-int(size) : page : page]
}
