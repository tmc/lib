package hopper

import (
	"bytes"
	"fmt"
	"testing"
)

func TestCompactNVFP4Scales(t *testing.T) {
	want := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	var expanded []byte
	for i := 0; i < len(want); i += 8 {
		for range 4 {
			expanded = append(expanded, want[i:i+8]...)
		}
	}
	got, err := CompactNVFP4Scales(expanded)
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("got %v, %v; want %v", got, err, want)
	}
	expanded[17] ^= 1
	if _, err := CompactNVFP4Scales(expanded); err == nil {
		t.Fatal("accepted different scales in a quad")
	}
	if _, err := CompactNVFP4Scales(expanded[:31]); err == nil {
		t.Fatal("accepted incomplete quad")
	}
}

func ExampleCompactNVFP4Scales() {
	quad := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	expanded := bytes.Repeat(quad, 4)
	compact, err := CompactNVFP4Scales(expanded)
	fmt.Println(compact, err)
	// Output: [1 2 3 4 5 6 7 8] <nil>
}
