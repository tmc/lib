package cudatracehost

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestOpenAdmissionAndMappedIdentity(t *testing.T) {
	if !sameMappedIdentity(42, 0x801, 42, "08:01") {
		t.Fatal("matching inode and device rejected")
	}
	if sameMappedIdentity(43, 0x801, 42, "08:01") {
		t.Fatal("inode mismatch accepted")
	}
	if sameMappedIdentity(42, 0x802, 42, "08:01") {
		t.Fatal("device mismatch accepted")
	}
	if sameMappedFile(nil, 42, "08:01") {
		t.Fatal("nil FileInfo accepted")
	}

	if runtime.GOOS != "linux" || (runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64") {
		f := filepath.Join(t.TempDir(), "lib.so")
		if err := os.WriteFile(f, []byte("stub"), 0600); err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(f)
		if err != nil {
			t.Fatal(err)
		}
		if sameMappedFile(info, 0, "00:00") {
			t.Fatal("non-linux sameMappedFile fabricated identity")
		}
		_, err = Open(Config{PID: 1, Library: f, Version: "12.8", Layer: "runtime"})
		if err == nil || !strings.Contains(err.Error(), "cudatracehost: linux/amd64 or linux/arm64 required") {
			t.Fatalf("Open on %s/%s = %v, want linux/amd64 or linux/arm64 required", runtime.GOOS, runtime.GOARCH, err)
		}
		return
	}

	if _, err := Open(Config{PID: 1, Library: "relative.so", Version: "12.8", Layer: "runtime"}); err == nil || !strings.Contains(err.Error(), "library path must be absolute") {
		t.Fatalf("relative library path err = %v", err)
	}
	if _, err := Open(Config{PID: 0, Library: "/usr/lib/libcudart.so", Version: "12.8", Layer: "runtime"}); err == nil || !strings.Contains(err.Error(), "missing target/library/layer identity") {
		t.Fatalf("missing target identity err = %v", err)
	}
}
