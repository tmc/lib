//go:build !linux

package nvrtcx

import "testing"

func TestLoadedCompilerIdentityUnavailable(t *testing.T) {
	if id, supported, err := loadedCompilerIdentity("/unverified/compiler.so"); id != "" || supported || err != nil {
		t.Fatalf("unavailable mapping invented: %q %t %v", id, supported, err)
	}
}
