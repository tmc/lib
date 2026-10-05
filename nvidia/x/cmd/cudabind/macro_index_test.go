package main

import "testing"

func TestMacroValueIndex(t *testing.T) {
	for _, tc := range []struct {
		name, file   string
		count, index int
		want         bool
	}{
		{"cudabind_macro_0", "helpers.cu", 1, 0, true},
		{"cudabind_macro_1", "helpers.cu", 2, 1, true},
		{"cudabind_macro_9999", "source.cu", 1, 0, false},
		{"cudabind_macro_0", "source.cu", 1, 0, false},
		{"cudabind_macro_9999", "helpers.cu", 1, 0, false},
		{"cudabind_macro_-1", "helpers.cu", 1, 0, false},
		{"cudabind_macro_9999999999999999999999", "helpers.cu", 1, 0, false},
		{"cudabind_macro_00", "helpers.cu", 1, 0, false},
		{"cudabind_macro_0", "helpers.cu", 0, 0, false},
		{"unrelated", "helpers.cu", 1, 0, false},
	} {
		t.Run(tc.name+"/"+tc.file, func(t *testing.T) {
			n, ok := macroValueIndex(tc.name, tc.file, "helpers.cu", tc.count)
			if n != tc.index || ok != tc.want {
				t.Fatalf("index=%d valid=%v, want %d %v", n, ok, tc.index, tc.want)
			}
		})
	}
}
