package libclang

import (
	"os"
	"path/filepath"
	"testing"
)

const src = `
// T is a test struct.
struct T {
	char c;
	double d;
	int a[3];
};

enum E { A, B = -2 };

int f(struct T *t, int n);
`

func TestParse(t *testing.T) {
	lib, err := Open(os.Getenv("CUDABIND_LIBCLANG"))
	if err != nil {
		t.Skip(err)
	}
	t.Logf("%s: %s", lib.Path(), lib.Version())
	file := filepath.Join(t.TempDir(), "t.c")
	if err := os.WriteFile(file, []byte(src), 0o666); err != nil {
		t.Fatal(err)
	}
	tu, err := lib.Parse(file, []string{"-x", "c", "-fparse-all-comments"})
	if err != nil {
		t.Fatal(err)
	}
	defer tu.Close()
	for _, d := range tu.Diagnostics() {
		if d.Severity >= Error {
			t.Fatal(d.Text)
		}
	}

	decls := make(map[string]Cursor)
	for _, c := range tu.Cursor().Children() {
		decls[c.Spelling()] = c
	}

	s := decls["T"]
	if s.Kind() != StructDecl {
		t.Fatalf("T is kind %d, want StructDecl", s.Kind())
	}
	if got := s.RawComment(); got != "// T is a test struct." {
		t.Errorf("T comment = %q", got)
	}
	if typ := s.Type(); typ.Size() != 32 || typ.Align() != 8 {
		t.Errorf("T size, align = %d, %d, want 32, 8", typ.Size(), typ.Align())
	}
	var fields []string
	var offs []int64
	for _, f := range s.Children() {
		if f.Kind() == FieldDecl {
			fields = append(fields, f.Spelling())
			offs = append(offs, f.OffsetOfField()/8)
		}
	}
	if len(fields) != 3 || offs[0] != 0 || offs[1] != 8 || offs[2] != 16 {
		t.Errorf("fields %v at %v, want [c d a] at [0 8 16]", fields, offs)
	}
	if a := s.Children()[2].Type(); a.Kind() != ConstantArray || a.Len() != 3 || a.Elem().Kind() != Int {
		t.Errorf("a is %s, want int[3]", a.Spelling())
	}

	var vals []int64
	for _, c := range decls["E"].Children() {
		if c.Kind() == EnumConstantDecl {
			vals = append(vals, c.EnumValue())
		}
	}
	if len(vals) != 2 || vals[0] != 0 || vals[1] != -2 {
		t.Errorf("E values = %v, want [0 -2]", vals)
	}

	f := decls["f"]
	args := f.Arguments()
	if f.Kind() != FunctionDecl || len(args) != 2 || f.ResultType().Kind() != Int {
		t.Fatalf("f: kind %d, %d args, result %s", f.Kind(), len(args), f.ResultType().Spelling())
	}
	if p := args[0].Type(); p.Kind() != Pointer || p.Pointee().Declaration().Spelling() != "T" {
		t.Errorf("f arg 0 is %s, want struct T *", p.Spelling())
	}
	if file, line, _ := f.Location(); filepath.Base(file) != "t.c" || line != 11 {
		t.Errorf("f at %s:%d, want t.c:11", file, line)
	}
}

func TestResourceDirSymlink(t *testing.T) {
	root := t.TempDir()
	multi := filepath.Join(root, "multiarch", "libclang.so")
	install := filepath.Join(root, "llvm-19", "lib")
	resource := filepath.Join(install, "clang", "19")
	for _, dir := range []string{filepath.Dir(multi), filepath.Join(resource, "include")} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{multi, filepath.Join(resource, "include", "stddef.h")} {
		if err := os.WriteFile(path, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	link := filepath.Join(install, "libclang.so")
	if err := os.Symlink(multi, link); err != nil {
		t.Fatal(err)
	}
	lib := &Library{path: link}
	want, err := filepath.EvalSymlinks(resource)
	if err != nil {
		t.Fatal(err)
	}
	got, err := filepath.EvalSymlinks(lib.ResourceDir())
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("ResourceDir() = %q, want %q", got, want)
	}
	// A symlink outside the installation still finds headers by the target.
	external := filepath.Join(root, "external.so")
	if err := os.Symlink(link, external); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(link); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(link, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	lib.path = external
	got, err = filepath.EvalSymlinks(lib.ResourceDir())
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("ResourceDir() = %q, want %q", got, want)
	}
}

func TestEnumIntegerType(t *testing.T) {
	lib, err := Open(os.Getenv("CUDABIND_LIBCLANG"))
	if err != nil {
		t.Skip(err)
	}
	file := filepath.Join(t.TempDir(), "enum.cc")
	tu, err := lib.Parse(file, []string{"-x", "c++", "-std=c++17"}, UnsavedFile{Name: file, Contents: `enum class U : unsigned long long { max = 0xffffffffffffffffULL };`})
	if err != nil {
		t.Fatal(err)
	}
	defer tu.Close()
	for _, d := range tu.Diagnostics() {
		if d.Severity >= Error {
			t.Fatal(d.Text)
		}
	}
	for _, c := range tu.Cursor().Children() {
		if c.Kind() != EnumDecl {
			continue
		}
		if got := c.EnumIntegerType().Kind(); got != ULongLong {
			t.Fatalf("underlying type = %v, want ULongLong", got)
		}
		for _, v := range c.Children() {
			if v.Kind() == EnumConstantDecl {
				if got := v.EnumUnsignedValue(); got != ^uint64(0) {
					t.Fatalf("value = %d, want max uint64", got)
				}
				return
			}
		}
	}
	t.Fatal("enum constant not found")
}
