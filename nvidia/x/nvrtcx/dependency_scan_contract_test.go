package nvrtcx

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestDependencyScanLiteralIdentity(t *testing.T) {
	dir := t.TempDir()
	header := filepath.Join(dir, "value.h")
	write := func(content string) {
		t.Helper()
		if err := os.WriteFile(header, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	for _, src := range []string{"#include <value.h>\n", "# include <value.h>\n", "\t#\tinclude <value.h> // comment\n"} {
		t.Run(src, func(t *testing.T) {
			write("#define VALUE 1\n")
			before, tracked := resolveExternalDependencies(src, []string{"-I" + dir}, nil)
			if !tracked || len(before) != 1 || !before[0].Tracked || before[0].Path != header {
				t.Fatalf("literal dependency not tracked: %+v, complete=%v", before, tracked)
			}
			write("#define VALUE 2\n")
			after, tracked := resolveExternalDependencies(src, []string{"-I" + dir}, nil)
			if !tracked || computeDependencyKey(before) == computeDependencyKey(after) {
				t.Fatalf("header mutation did not change complete identity: %+v", after)
			}
		})
	}
}

func TestDependencyScanSymlinkSiblingWitness(t *testing.T) {
	dir := t.TempDir()
	root, other := filepath.Join(dir, "root"), filepath.Join(dir, "other")
	for _, path := range []string{root, filepath.Join(other, "real")} {
		if err := os.MkdirAll(path, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	write := func(path, text string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(other, "real/top.h"), "#include \"../detail.h\"\n")
	write(filepath.Join(root, "detail.h"), "#define VALUE 1\n")
	write(filepath.Join(other, "detail.h"), "#define VALUE 2\n")
	if err := os.Symlink(filepath.Join(other, "real"), filepath.Join(root, "alias")); err != nil {
		t.Fatal(err)
	}
	// Keep the parent traversal in the filesystem path. filepath.Join would
	// clean alias/.. before the filesystem has followed alias.
	actual, err := os.ReadFile(root + "/alias/../detail.h")
	if err != nil || string(actual) != "#define VALUE 2\n" {
		t.Fatalf("bounded filesystem witness: data=%q error=%v", actual, err)
	}
	const src = "#include <alias/top.h>\n"
	before, trackedBefore := resolveExternalDependencies(src, []string{"-I" + root}, nil)
	write(filepath.Join(other, "detail.h"), "#define VALUE 99\n")
	after, trackedAfter := resolveExternalDependencies(src, []string{"-I" + root}, nil)
	if trackedBefore && trackedAfter && computeDependencyKey(before) == computeDependencyKey(after) {
		t.Fatalf("actual sibling mutation leaves admitted identity unchanged: before=%+v after=%+v", before, after)
	}
	if trackedAfter {
		sum := sha256.Sum256([]byte("#define VALUE 99\n"))
		found := false
		for _, dep := range after {
			found = found || dep.Tracked && dep.Digest == hex.EncodeToString(sum[:])
		}
		if !found {
			t.Fatalf("tracked closure omits actual sibling content: %+v", after)
		}
	}
}

func TestDependencyScanFiniteDirectoryAliases(t *testing.T) {
	dir := t.TempDir()
	real := filepath.Join(dir, "real")
	if err := os.Mkdir(real, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(real, "top.h"), []byte("#define VALUE 1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"a", "b"} {
		if err := os.Symlink(real, filepath.Join(dir, name)); err != nil {
			t.Fatal(err)
		}
	}
	deps, tracked := resolveExternalDependencies("#include <a/top.h>\n#include <b/top.h>\n", []string{"-I" + dir}, nil)
	if !tracked || len(deps) != 1 || !deps[0].Tracked {
		t.Fatalf("finite aliases must visit one physical header once: %+v complete=%v", deps, tracked)
	}
}

func TestDependencyScanParentRootAndFileSymlinkUntracked(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "top.h"), []byte("#define VALUE 1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(dir, "top.h"), filepath.Join(dir, "alias.h")); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, src, search string
	}{
		{"parent root", "#include <top.h>\n", dir + "/child/.."},
		{"file symlink", "#include <alias.h>\n", dir},
	} {
		t.Run(tt.name, func(t *testing.T) {
			deps, tracked := resolveExternalDependencies(tt.src, []string{"-I" + tt.search}, nil)
			if tracked {
				t.Fatalf("ambiguous filesystem path admitted: %+v", deps)
			}
		})
	}
}

func TestDependencyScanRefusesUnsupportedPreprocessing(t *testing.T) {
	for _, tt := range []struct {
		name, src string
	}{
		{"macro", "#define HEADER \"value.h\"\n#include HEADER\n"},
		{"spaced macro", "# include HEADER\n"},
		{"continued directive", "#inc\\\nlude \"value.h\"\n"},
		{"continued operand", "#include \\\r\n\"value.h\"\n"},
		{"comment in directive", "#/**/include \"value.h\"\n"},
		{"comment before directive", "/* comment */ #include \"value.h\"\n"},
		{"continued line comment", "// comment \\\n#include \"value.h\"\n"},
		{"digraph", "%:include \"value.h\"\n"},
		{"trigraph", "??=include \"value.h\"\n"},
		{"include next", "#include_next \"value.h\"\n"},
		{"pragma", "#pragma include_alias(\"a.h\", \"value.h\")\n"},
		{"incomplete literal", "#include <value.h\n"},
		{"numeric directive", "#123\n"},
		{"uppercase directive", "#INCLUDE\n"},
		{"punctuation directive", "#$\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			deps, tracked := resolveExternalDependencies(tt.src, nil, nil)
			if tracked {
				t.Fatalf("unsupported preprocessing admitted as complete: %+v", deps)
			}
			found := false
			for _, dep := range deps {
				if !dep.Tracked && dep.Path == "<source>" && dep.ErrorMsg != "" {
					found = true
				}
			}
			if !found {
				t.Fatalf("no source diagnostic for unsupported preprocessing: %+v", deps)
			}
		})
	}
}

func TestDependencyScanNullDirectives(t *testing.T) {
	deps, tracked := resolveExternalDependencies("#\n # \t\n#// comment\n", nil, nil)
	if !tracked || len(deps) != 0 {
		t.Fatalf("null directives are not external inputs: %+v complete=%v", deps, tracked)
	}
}

func TestDependencyScanInjectedHeadersUntracked(t *testing.T) {
	for _, flag := range []string{
		"--pre-include=value.h", "-include=value.h", "--pre-include", "-include",
		"--pch", "-pch", "--use-pch=value.pch", "-use-pch=value.pch",
		"--create-pch=value.pch", "-create-pch=value.pch", "--include-path", "-I",
	} {
		t.Run(flag, func(t *testing.T) {
			deps, tracked := resolveExternalDependencies("", []string{flag}, nil)
			if tracked || len(deps) != 1 || deps[0].Tracked || deps[0].Path != flag || deps[0].ErrorMsg == "" {
				t.Fatalf("injected input lacks explicit untracked diagnostic: %+v complete=%v", deps, tracked)
			}
		})
	}
}

func TestDependencyScanQuotedTransitiveContext(t *testing.T) {
	dir := t.TempDir()
	write := func(name, text string) {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("a/top.h", "#include \"detail.h\"\n#include <common.h>\n")
	write("b/top.h", "#include \"detail.h\"\n")
	write("a/detail.h", "#define A 1\n")
	write("b/detail.h", "#define B 2\n")
	write("a/common.h", "#define WRONG 1\n")
	write("common.h", "#define COMMON 3\n")
	const src = "#include <a/top.h>\n#include <b/top.h>\n"
	before, tracked := resolveExternalDependencies(src, []string{"--include-path=" + dir}, nil)
	if !tracked {
		t.Fatalf("literal closure incomplete: %+v", before)
	}
	paths := make(map[string]bool)
	for _, dep := range before {
		paths[dep.Path] = dep.Tracked
	}
	for _, name := range []string{"a/top.h", "b/top.h", "a/detail.h", "b/detail.h", "common.h"} {
		if !paths[filepath.Join(dir, name)] {
			t.Errorf("missing dependency %s: %+v", name, before)
		}
	}
	if paths[filepath.Join(dir, "a/common.h")] {
		t.Fatal("angle include incorrectly used the including header's directory")
	}
	write("b/detail.h", "#define B 99\n")
	after, tracked := resolveExternalDependencies(src, []string{"--include-path=" + dir}, nil)
	if !tracked || computeDependencyKey(before) == computeDependencyKey(after) {
		t.Fatalf("second same-name header mutation did not change identity: %+v", after)
	}
}

func TestDependencyScanUnsupportedTransitive(t *testing.T) {
	deps, tracked := resolveExternalDependencies("#include <mem.h>\n", nil, map[string]string{
		"mem.h": "#include HEADER\n",
	})
	if tracked || len(deps) != 1 || deps[0].Path != "memory:mem.h" || deps[0].ErrorMsg == "" {
		t.Fatalf("in-memory unsupported closure admitted: %+v complete=%v", deps, tracked)
	}
}

func TestDependencyScanRootQuotedOriginUntracked(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "value.h"), []byte("#define VALUE 1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, src := range []string{"#include \"value.h\"\n", "# include \"value.h\"\n"} {
		deps, tracked := resolveExternalDependencies(src, []string{"-I" + dir}, nil)
		if tracked {
			t.Fatalf("root quoted include admitted without source filename: %+v", deps)
		}
		found := false
		for _, dep := range deps {
			if dep.Path == "<source>" && !dep.Tracked && dep.ErrorMsg == "quoted include source origin is not available" {
				found = true
			}
		}
		if !found {
			t.Fatalf("missing quoted-origin diagnostic: %+v", deps)
		}
	}
}

func TestDependencyScanAmbiguousOrAbsoluteHeaderUntracked(t *testing.T) {
	dir := t.TempDir()
	header := filepath.Join(dir, "value.h")
	if err := os.WriteFile(header, []byte("#define VALUE 1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, src string
		headers   map[string]string
	}{
		{"overlapping search", "#include <value.h>\n", map[string]string{"value.h": "#define VALUE 2\n"}},
		{"absolute", "#include <" + header + ">\n", nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			deps, tracked := resolveExternalDependencies(tt.src, []string{"-I=" + dir}, tt.headers)
			if tracked {
				t.Fatalf("unsupported search admitted as complete: %+v", deps)
			}
		})
	}
	deps, tracked := resolveExternalDependencies("#include <value.h>\n", []string{"-I=" + dir}, nil)
	if !tracked || len(deps) != 1 || deps[0].Path != header || !deps[0].Tracked {
		t.Fatalf("supported include path did not track file: %+v complete=%v", deps, tracked)
	}
	deps, tracked = resolveExternalDependencies("#include <virtual/top.h>\n", []string{"-I" + dir}, map[string]string{
		"virtual/top.h":   "#include \"value.h\"\n",
		"virtual/value.h": "#define VALUE 2\n",
	})
	if tracked {
		t.Fatalf("relative virtual header search admitted as complete: %+v", deps)
	}
}
