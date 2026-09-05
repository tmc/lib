package nvrtcx

import "testing"

func TestCompilerCacheKey(t *testing.T) {
	input := computeCacheKey("source", "name.cu", nil, nil)
	base := compilerCacheKey(input, "nvrtc 13.0 /cuda/libnvrtc.so")
	if base != compilerCacheKey(input, "nvrtc 13.0 /cuda/libnvrtc.so") {
		t.Fatal("same compiler identity changed key")
	}
	for _, identity := range []string{"nvrtc 13.1 /cuda/libnvrtc.so", "nvrtc 13.0 /other/libnvrtc.so"} {
		if base == compilerCacheKey(input, identity) {
			t.Fatalf("compiler identity %q reused key", identity)
		}
	}
	if base == input {
		t.Fatal("legacy unqualified key reused")
	}
}
