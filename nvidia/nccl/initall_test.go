package nccl

import (
	"errors"
	"reflect"
	"testing"
	"unsafe"
)

func TestInitAllOutputArray(t *testing.T) {
	wantErr := errors.New("partial initialization")
	got, err := initAll([]int32{2, 0, 1}, func(out *Comm, n int32, devices *int32) error {
		if n != 3 || !reflect.DeepEqual(unsafe.Slice(devices, int(n)), []int32{2, 0, 1}) {
			t.Fatal("device order changed")
		}
		copy(unsafe.Slice(out, int(n)), []Comm{11, 22, 33})
		return wantErr
	})
	if !errors.Is(err, wantErr) || !reflect.DeepEqual(got, []Comm{11, 22, 33}) {
		t.Fatalf("got %v, %v", got, err)
	}
}
func TestInitAllInvalidDevices(t *testing.T) {
	for _, devices := range [][]int32{nil, {0, 0}, {-1}} {
		_, err := initAll(devices, func(*Comm, int32, *int32) error { t.Fatal("called native code for invalid devices"); return nil })
		if err == nil {
			t.Fatalf("accepted %v", devices)
		}
	}
}
