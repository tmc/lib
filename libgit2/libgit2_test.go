package libgit2

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"unsafe"
)

func TestRepositoryBlob(t *testing.T) {
	path := os.Getenv("LIBGIT2_TEST_LIBRARY")
	if path == "" {
		t.Skip("set LIBGIT2_TEST_LIBRARY to run native tests")
	}
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	if Has("git_nonexistent_symbol_for_test") {
		t.Fatal("found nonexistent symbol")
	}
	check := func(status int32, err error) int32 {
		if err != nil {
			t.Fatal(err)
		}
		return status
	}
	if n := check(Libgit2Init()); n < 1 {
		t.Fatalf("init = %d", n)
	}
	defer Libgit2Shutdown()

	root := t.TempDir()
	name := append([]byte(filepath.Join(root, "repo")), 0)
	var repo *Repository
	if status := check(RepositoryInit(unsafe.Pointer(&repo), &name[0], 1)); status != 0 || repo == nil {
		t.Fatalf("init repository: status=%d repo=%p", status, repo)
	}
	defer RepositoryFree(repo)
	data := []byte("libgit2 bindings\n")
	var id Oid
	if status := check(BlobCreateFromBuffer(&id, repo, unsafe.Pointer(&data[0]), uint(len(data)))); status != 0 {
		t.Fatalf("create blob: %d", status)
	}
	var hex [41]byte
	if _, err := OidTostr(&hex[0], uint(len(hex)), &id); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("git", "--git-dir="+string(name[:len(name)-1]), "cat-file", "blob", string(hex[:40]))
	got, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git cat-file: %v: %s", err, got)
	}
	if !bytes.Equal(got, data) {
		t.Fatalf("blob = %q, want %q", got, data)
	}

	var parsed Oid
	if status := check(OidFromstr(&parsed, &hex[0])); status != 0 || parsed != id {
		t.Fatalf("parse oid: status=%d got=%x want=%x", status, parsed, id)
	}
	invalid := append([]byte("not-an-object-id"), 0)
	if status := check(OidFromstr(&parsed, &invalid[0])); status >= 0 {
		t.Fatalf("invalid oid accepted: %d", status)
	}
	missing := append([]byte(filepath.Join(root, "missing")), 0)
	var absent *Repository
	if status := check(RepositoryOpen(unsafe.Pointer(&absent), &missing[0])); status >= 0 || absent != nil {
		t.Fatalf("missing repository: status=%d repo=%p", status, absent)
	}
}

func Example() {
	// Load checks the library path before making native calls.
	if err := Load("/nonexistent/libgit2"); err != nil {
		fmt.Println("library unavailable")
	}
	// Output: library unavailable
}

func TestUnavailableLibrary(t *testing.T) {
	if err := Close(); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LIBGIT2_LIB_PATH", filepath.Join(t.TempDir(), "missing-library"))
	if n, err := Libgit2Init(); err == nil || n != 0 {
		t.Fatalf("init without library = %d, %v", n, err)
	}
	if err := RepositoryFree(nil); err == nil {
		t.Fatal("free without library returned no error")
	}
	if err := Close(); err != nil {
		t.Fatal(err)
	}
}
