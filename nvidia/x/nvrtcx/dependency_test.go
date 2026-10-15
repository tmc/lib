package nvrtcx

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveExternalDependenciesWalksIncludeClosure(t *testing.T) {
	dir := t.TempDir()
	write := func(name, content string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// top.cuh reaches deep.cuh, which only the closure walk can see.
	write("top.cuh", "#include \"deep.cuh\"\n")
	write("deep.cuh", "__device__ float deep(float v) { return v; }\n")

	const src = "#include <top.cuh>\n"
	flags := []string{"-I" + dir}

	deps, allTracked := resolveExternalDependencies(src, flags, nil)
	if !allTracked {
		t.Fatalf("allTracked = false, deps = %+v", deps)
	}
	tracked := make(map[string]string)
	for _, d := range deps {
		tracked[filepath.Base(d.Path)] = d.Digest
	}
	if _, ok := tracked["deep.cuh"]; !ok {
		t.Fatalf("transitive header not tracked, got %+v", deps)
	}

	// Editing only the transitive header must move the key. Without it, a
	// cached artifact compiled from the old deep.cuh is returned as current.
	before := computeDependencyKey(deps)
	write("deep.cuh", "__device__ float deep(float v) { return v * 77.0f; }\n")
	after, _ := resolveExternalDependencies(src, flags, nil)
	if computeDependencyKey(after) == before {
		t.Error("dependency key unchanged after editing a transitive header")
	}
}

func TestResolveExternalDependenciesWalksInMemoryHeaders(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "deep.cuh"), []byte("// deep\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// An in-memory header is hashed by the caller, but what it includes is not.
	headers := map[string]string{"mem.cuh": "#include \"deep.cuh\"\n"}
	deps, allTracked := resolveExternalDependencies("#include <mem.cuh>\n", []string{"-I" + dir}, headers)
	if !allTracked {
		t.Fatalf("allTracked = false, deps = %+v", deps)
	}
	if len(deps) != 1 || filepath.Base(deps[0].Path) != "deep.cuh" {
		t.Fatalf("deps = %+v, want deep.cuh reached through the in-memory header", deps)
	}
}

func TestResolveExternalDependenciesUntracksUnresolvedTransitive(t *testing.T) {
	dir := t.TempDir()
	// top.cuh resolves; the header it includes does not exist anywhere.
	if err := os.WriteFile(filepath.Join(dir, "top.cuh"), []byte("#include \"missing.cuh\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	deps, allTracked := resolveExternalDependencies("#include \"top.cuh\"\n", []string{"-I" + dir}, nil)
	if allTracked {
		t.Fatalf("allTracked = true with an unresolvable transitive header, deps = %+v", deps)
	}
}
