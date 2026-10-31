package cuptipcsampling

import (
	"errors"
	"testing"
	"unsafe"
)

func TestSamplingNamesOwnership(t *testing.T) {
	reads := 0
	n, err := newSamplingNames(3, 10, 20, func(ptr uintptr, limit int) (string, error) {
		reads++
		if limit != 10 {
			t.Fatal("wrong read bound")
		}
		return "kernel", nil
	})
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
	if reads != 2 {
		t.Fatal("duplicate copy")
	}
	if err := n.reserve(2); err == nil {
		t.Fatal("unbounded drain admitted")
	}
	n.release()
	n.release()
	if _, err := n.copy(10); err == nil {
		t.Fatal("access after release")
	}
}

func TestSamplingNamesOwnString(t *testing.T) {
	storage := []byte("kernel")
	n, err := newSamplingNames(1, 10, 10, func(uintptr, int) (string, error) {
		return unsafe.String(&storage[0], len(storage)), nil
	})
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

func TestSamplingNamesFailedCopyIsRetained(t *testing.T) {
	failure := errors.New("unterminated native string")
	for _, tt := range []struct {
		name, value string
		readError   error
		bytes       int
	}{
		{"native read", "", failure, 10}, {"empty", "", nil, 10}, {"name bound", "12345678", nil, 10}, {"total bound", "1234", nil, 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			reads := 0
			n, err := newSamplingNames(1, 8, tt.bytes, func(uintptr, int) (string, error) { reads++; return tt.value, tt.readError })
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
			if err := n.reserve(1); err == nil {
				t.Fatal("failed copy released its budget entry")
			}
			if reads != 1 {
				t.Fatalf("reads=%d", reads)
			}
		})
	}
}

// CUPTI returns one function-name pointer per function and hands the same
// pointer to later sessions in the process, so closing one session must leave
// the string readable for the next. Freeing it at release made the second
// session read freed memory and then free it again.
func TestSamplingNamesOutliveSession(t *testing.T) {
	native := map[uintptr]string{10: "gemm"}
	read := func(ptr uintptr, _ int) (string, error) {
		s, ok := native[ptr]
		if !ok {
			return "", errors.New("read of freed native name")
		}
		return s, nil
	}
	for session := range 2 {
		n, err := newSamplingNames(4, 16, 64, read)
		if err != nil {
			t.Fatal(err)
		}
		if got, err := n.copy(10); err != nil || got != "gemm" {
			t.Errorf("session %d: copy = %q, %v; want gemm", session, got, err)
		}
		n.release()
	}
}
