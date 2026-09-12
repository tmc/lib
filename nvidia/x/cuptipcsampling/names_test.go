package cuptipcsampling

import (
	"errors"
	"reflect"
	"testing"
	"unsafe"
)

func TestSamplingNamesOwnership(t *testing.T) {
	var freed []uintptr
	reads := 0
	n, err := newSamplingNames(3, 10, 20, func(ptr uintptr, limit int) (string, error) {
		reads++
		if limit != 10 {
			t.Fatal("wrong read bound")
		}
		return "kernel", nil
	}, func(ptr uintptr) { freed = append(freed, ptr) })
	if err != nil {
		t.Fatal(err)
	}
	if err := n.reserve(3); err != nil {
		t.Fatal(err)
	}
	for _, ptr := range []uintptr{20, 10, 20} {
		if s, err := n.copy(ptr); err != nil || s != "kernel" {
			t.Fatalf("copy=%q %v", s, err)
		}
	}
	if reads != 2 || len(freed) != 0 {
		t.Fatal("duplicate copy or premature free")
	}
	if err := n.reserve(2); err == nil {
		t.Fatal("unbounded drain admitted")
	}
	n.release()
	n.release()
	if !reflect.DeepEqual(freed, []uintptr{10, 20}) {
		t.Fatalf("freed=%v", freed)
	}
	if _, err := n.copy(10); err == nil {
		t.Fatal("access after release")
	}
}

func TestSamplingNamesOwnString(t *testing.T) {
	storage := []byte("kernel")
	n, err := newSamplingNames(1, 10, 10, func(uintptr, int) (string, error) {
		return unsafe.String(&storage[0], len(storage)), nil
	}, func(uintptr) {})
	if err != nil {
		t.Fatal(err)
	}
	defer n.release()
	got, err := n.copy(1)
	if err != nil {
		t.Fatal(err)
	}
	storage[0] = 'X'
	if got != "kernel" {
		t.Fatal("name aliases native storage")
	}
}

func TestSamplingNamesFailedCopyRetainsOwnership(t *testing.T) {
	failure := errors.New("unterminated native string")
	for _, tt := range []struct {
		name, value string
		readError   error
		bytes       int
	}{
		{"native read", "", failure, 10}, {"empty", "", nil, 10}, {"name bound", "12345678", nil, 10}, {"total bound", "1234", nil, 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			reads, frees := 0, 0
			n, err := newSamplingNames(1, 8, tt.bytes, func(uintptr, int) (string, error) { reads++; return tt.value, tt.readError }, func(ptr uintptr) {
				if ptr != 7 {
					t.Fatal("wrong free")
				}
				frees++
			})
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < 2; i++ {
				s, err := n.copy(7)
				if err == nil || s != "" {
					t.Fatal("invalid name accepted")
				}
				if tt.readError != nil && !errors.Is(err, failure) {
					t.Fatal("lost error")
				}
			}
			n.release()
			if reads != 1 || frees != 1 {
				t.Fatalf("reads=%d frees=%d", reads, frees)
			}
		})
	}
}
