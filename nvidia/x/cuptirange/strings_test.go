package cuptirange

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestMetricNames(t *testing.T) {
	names := []string{"sm__cycles_elapsed.avg", "dram__bytes_read.sum"}
	n, err := newMetricNames(names)
	if err != nil {
		t.Fatal(err)
	}
	defer n.close()
	names[0] = "changed"
	runtime.GC()
	for i, want := range []string{"sm__cycles_elapsed.avg\x00", "dram__bytes_read.sum\x00"} {
		if string(n.data[i]) != want {
			t.Fatalf("name %d=%q", i, n.data[i])
		}
		if n.pointers[i] != uintptr(unsafe.Pointer(&n.data[i][0])) {
			t.Fatal("native pointer table differs")
		}
	}
}

func TestMetricNamesValidation(t *testing.T) {
	for _, names := range [][]string{nil, {""}, {"metric\x00suffix"}, {"metric", "metric"}} {
		if n, err := newMetricNames(names); err == nil || n != nil {
			t.Fatalf("accepted %q", names)
		}
	}
}
