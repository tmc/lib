package loader

import "testing"

func TestDefaultLibraryNames(t *testing.T) {
	names := defaultLibraryNames()
	if len(names) == 0 {
		t.Skip("unsupported platform")
	}
	for _, name := range names {
		if name == "" {
			t.Fatal("empty library name")
		}
	}
}
