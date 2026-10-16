package tileir

import (
	"errors"
	"testing"
)

// NVRTC 13.0, 13.1 and 13.2 fail with "unrecognized option --tile-only";
// 13.3 and 13.4 compile the semantic source (measured on GB10).
func TestCheckTileCompiler(t *testing.T) {
	tests := []struct {
		major, minor int
		want         error
	}{
		{12, 9, ErrTileCompilerUnsupported},
		{13, 0, ErrTileCompilerUnsupported},
		{13, 2, ErrTileCompilerUnsupported},
		{13, 3, nil},
		{13, 4, nil},
		{14, 0, nil},
	}
	for _, tt := range tests {
		err := checkTileCompiler(tt.major, tt.minor)
		if !errors.Is(err, tt.want) || (err == nil) != (tt.want == nil) {
			t.Errorf("checkTileCompiler(%d, %d) = %v, want %v", tt.major, tt.minor, err, tt.want)
		}
	}
	const want = "tileir: NVRTC does not support tile-only compilation: NVRTC 13.2, need 13.3 or later"
	if err := checkTileCompiler(13, 2); err == nil || err.Error() != want {
		t.Errorf("checkTileCompiler(13, 2) = %v, want %q", err, want)
	}
}
