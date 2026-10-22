package dsv41

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"testing"
)

func TestVerifiedSourceRefusesUnverified(t *testing.T) {
	for _, m := range []*Model{nil, {}, {boundSource: bytes.NewReader([]byte{1}), boundSize: 1}} {
		if source, size, err := m.VerifiedSource(); err == nil || source != nil || size != 0 {
			t.Fatal("unverified source admitted")
		}
	}
}

func ExampleModel_VerifiedSource() {
	var model *Model
	_, _, err := model.VerifiedSource()
	fmt.Println(err)
	// Output: V4.1 source is not verified
}

// Private wrapper wiring, not a forged verified Model or BindVerified bypass.
func TestVerifiedSourceView(t *testing.T) {
	source := bytes.NewReader([]byte{1, 2, 3, 4})
	view := verifiedSource{source, 4}
	var reader io.ReaderAt = view
	if _, ok := reader.(io.Closer); ok {
		t.Fatal("view exposes source closure")
	}
	for _, tc := range []struct {
		offset int64
		n      int
		ok     bool
	}{{0, 4, true}, {4, 0, true}, {-1, 1, false}, {3, 2, false}, {5, 0, false}} {
		got := make([]byte, tc.n)
		n, err := reader.ReadAt(got, tc.offset)
		if tc.ok {
			if n != tc.n || err != nil {
				t.Fatalf("valid read %+v: %d %v", tc, n, err)
			}
		} else if !errors.Is(err, io.ErrUnexpectedEOF) || n != 0 {
			t.Fatalf("invalid read %+v: %d %v", tc, n, err)
		}
	}
	got := make([]byte, 4)
	if _, err := reader.ReadAt(got, 0); err != nil || !bytes.Equal(got, []byte{1, 2, 3, 4}) {
		t.Fatal("source byte identity changed")
	}
}
