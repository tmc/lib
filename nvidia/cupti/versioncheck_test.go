package cupti

import (
	"errors"
	"testing"
)

func TestVersionCheckEncoding(t *testing.T) {
	loadMu.Lock()
	defer loadMu.Unlock()
	saved := fnGetVersion
	defer func() { fnGetVersion = saved }()
	for _, tc := range []struct {
		raw      uint32
		major    int
		mismatch bool
	}{
		{130001, 13, false}, {130200, 13, false}, {120901, 12, true}, {28, 0, false},
	} {
		fnGetVersion = func(out *uint32) Result { *out = tc.raw; return Success }
		err := checkVersion()
		var mismatch *VersionMismatchError
		if errors.As(err, &mismatch) != tc.mismatch {
			t.Fatalf("raw=%d: error=%v", tc.raw, err)
		}
		if mismatch != nil && (mismatch.Raw != int(tc.raw) || mismatch.Got != tc.major || mismatch.Want != 13) {
			t.Fatalf("raw=%d: mismatch=%+v", tc.raw, mismatch)
		}
	}
}
