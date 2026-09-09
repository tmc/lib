package git

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	raw "github.com/tmc/lib/libgit2"
)

func TestRepository(t *testing.T) {
	lib := os.Getenv("LIBGIT2_TEST_LIBRARY")
	if lib == "" {
		t.Skip("set LIBGIT2_TEST_LIBRARY for native tests")
	}
	if err := raw.Load(lib); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "repo")
	repo, err := Init(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer repo.Close()
	for _, data := range [][]byte{nil, []byte("generated Go API\x00with binary data\n")} {
		id, err := repo.CreateBlob(data)
		if err != nil {
			t.Fatal(err)
		}
		gotID, err := ParseOID(id.String())
		if err != nil || gotID != id {
			t.Fatalf("OID round trip: %v %v", gotID, err)
		}
		blob, err := repo.LookupBlob(id)
		if err != nil {
			t.Fatal(err)
		}
		got, err := blob.Contents()
		if err != nil || !bytes.Equal(got, data) {
			t.Fatalf("contents: %q %v", got, err)
		}
		if !errors.Is(repo.Close(), ErrChildren) {
			t.Fatal("repository closed with live blob")
		}
		if err := blob.Close(); err != nil {
			t.Fatal(err)
		}
		if err := blob.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := blob.Contents(); !errors.Is(err, ErrClosed) {
			t.Fatalf("closed blob: %v", err)
		}
		if !bytes.Equal(got, data) {
			t.Fatal("content copy changed after close")
		}
		out, err := exec.Command("git", "--git-dir="+dir, "cat-file", "blob", id.String()).CombinedOutput()
		if err != nil || !bytes.Equal(out, data) {
			t.Fatalf("git oracle: %q %v", out, err)
		}
	}
	if _, err := repo.LookupBlob(OID{}); err == nil {
		t.Fatal("missing object succeeded")
	} else {
		var ge *Error
		if !errors.As(err, &ge) || ge.Code >= 0 || ge.Message == "" || ge.Class == 0 {
			t.Fatalf("native error details: %#v", err)
		}
	}
	if err := repo.Close(); err != nil {
		t.Fatal(err)
	}
	if err := repo.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := repo.CreateBlob(nil); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed repository: %v", err)
	}
	reopened, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := reopened.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(filepath.Join(dir, "missing")); err == nil {
		t.Fatal("missing repository opened")
	}
}

func TestInput(t *testing.T) {
	for _, s := range []string{"", "abc", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"} {
		if _, err := ParseOID(s); err == nil {
			t.Errorf("accepted %q", s)
		}
	}
	if _, err := Open("bad\x00path"); err == nil {
		t.Fatal("accepted NUL")
	}
	if _, err := Init("bad\x00path", true); err == nil {
		t.Fatal("accepted NUL")
	}
}

func Example() {
	id, err := ParseOID("e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(id.String())
	// Output: e69de29bb2d1d6434b8b29ae775ad8c2e48c5391
}

func TestLoadFailure(t *testing.T) {
	if err := raw.Close(); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LIBGIT2_LIB_PATH", filepath.Join(t.TempDir(), "missing-library"))
	_, err := Open("unused")
	var loadErr *raw.LibraryError
	if !errors.As(err, &loadErr) {
		t.Fatalf("loading error was not preserved: %v", err)
	}
	if err := raw.Close(); err != nil {
		t.Fatal(err)
	}
}

func ExampleParseOID() {
	id, err := ParseOID("e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")
	fmt.Println(id.String(), err)
	// Output: e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 <nil>
}

func ExampleOID_String() {
	fmt.Println((OID{}).String())
	// Output: 0000000000000000000000000000000000000000
}

func ExampleError_Error() {
	err := &Error{Operation: "lookup blob", Code: -3, Message: "object not found"}
	fmt.Println(err)
	// Output: lookup blob: object not found (code -3)
}
