package cudaobjdump

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

func fake(t *testing.T, body string) Tool {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("POSIX subprocess fixture")
	}
	p := filepath.Join(t.TempDir(), "cuobjdump")
	if err := os.WriteFile(p, []byte("#!/bin/sh\n"+body+"\n"), 0700); err != nil {
		t.Fatal(err)
	}
	return Tool{Path: p, TempDir: t.TempDir()}
}

func input(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "-input with spaces")
	if err := os.WriteFile(p, []byte("original"), 0600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestTool(t *testing.T) {
	tool := fake(t, `case "$1" in
--version) printf 'fixture version';;
--list-elf) printf 'ELF file 1: kernel.cubin\n';;
--list-ptx) printf 'PTX file 1: kernel.ptx\n';;
--extract-elf) test -f "$3" || exit 3; printf 'elf image' > kernel.cubin;;
--extract-ptx) test -f "$3" || exit 3; printf 'ptx image' > kernel.ptx;;
--dump-sass|--dump-elf|--dump-resource-usage) printf '%s' "$1";;
*) exit 5;;
esac`)
	p := input(t)
	ctx := context.Background()
	if _, err := tool.Version(ctx); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []Kind{ELF, PTX} {
		list, r, err := tool.List(ctx, p, kind)
		if err != nil || len(list) != 1 || r.Path != tool.Path {
			t.Fatalf("list: %v %+v %v", list, r, err)
		}
		out, _, err := tool.Extract(ctx, p, kind)
		if err != nil || len(out) != 1 || len(out[0].Data) == 0 {
			t.Fatalf("extract: %v %v", out, err)
		}
	}
	for _, s := range []Section{SASS, ELFSections, Resources} {
		r, err := tool.Dump(ctx, p, s)
		if err != nil || len(r.Stdout) == 0 {
			t.Fatalf("dump: %+v %v", r, err)
		}
	}
	b, _ := os.ReadFile(p)
	if string(b) != "original" {
		t.Fatal("input overwritten")
	}
	entries, _ := os.ReadDir(tool.TempDir)
	if len(entries) != 0 {
		t.Fatal("scratch leaked")
	}
}

func TestExtractionFailure(t *testing.T) {
	for _, body := range []string{`printf partial > x; printf failure >&2; exit 1`, `ln -s /etc/passwd escape`, `mkdir unexpected`} {
		tool := fake(t, "if test \"$1\" = --list-elf; then printf 'ELF file 1: escape\\n'; exit 0; fi\n"+body)
		if _, _, err := tool.Extract(context.Background(), input(t), ELF); err == nil {
			t.Fatalf("accepted %q", body)
		}
		entries, _ := os.ReadDir(tool.TempDir)
		if len(entries) != 0 {
			t.Fatal("failed extraction leaked files")
		}
	}
}

func TestExtractionNames(t *testing.T) {
	for _, name := range []string{"../escape", "/absolute", "..", "x\\y", "safe\nELF file 2: safe"} {
		tool := fake(t, "if test \"$1\" = --list-elf; then printf '%s\\n' 'ELF file 1: "+name+"'; else exit 33; fi")
		_, _, err := tool.Extract(context.Background(), input(t), ELF)
		if err == nil || !strings.Contains(err.Error(), "unsafe or duplicate") {
			t.Fatalf("name %q: %v", name, err)
		}
	}
}

func TestInvalid(t *testing.T) {
	for _, s := range []string{"unknown", "ELF file x: bad", "ELF file 0: bad", "ELF file 1:"} {
		if _, err := ParseList(s, ELF); err == nil {
			t.Errorf("accepted %q", s)
		}
	}
	if _, err := ParseList("", Kind(99)); err == nil {
		t.Fatal("accepted bad kind")
	}
	tool := Tool{Path: "nonexistent-tool"}
	p := input(t)
	if _, _, err := tool.List(context.Background(), p, ELF); err == nil {
		t.Fatal("missing tool succeeded")
	}
	if _, err := tool.Dump(context.Background(), p, Section(99)); err == nil {
		t.Fatal("bad section succeeded")
	}
	if _, _, err := tool.Extract(context.Background(), t.TempDir(), ELF); err == nil {
		t.Fatal("directory input succeeded")
	}
}

func ExampleParseList() {
	images, err := ParseList("ELF file 1: kernel.sm_121.cubin\n", ELF)
	if err != nil {
		panic(err)
	}
	fmt.Println(images[0].Index, images[0].Name)
	// Output: 1 kernel.sm_121.cubin
}

func TestInstalledTool(t *testing.T) {
	p := os.Getenv("CUDAOBJDUMP_TEST_BINARY")
	if p == "" {
		t.Skip("set CUDAOBJDUMP_TEST_BINARY and CUDAOBJDUMP_TEST_{CUBIN,PTX} to a known embedded-artifact fixture")
	}
	original, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	references := make(map[Kind][]byte)
	for kind, env := range map[Kind]string{ELF: "CUDAOBJDUMP_TEST_CUBIN", PTX: "CUDAOBJDUMP_TEST_PTX"} {
		path := os.Getenv(env)
		if path == "" {
			t.Fatalf("%s is required for known-artifact comparison", env)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) == 0 {
			t.Fatalf("empty reference %s", path)
		}
		references[kind] = data
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	tool := Tool{Path: os.Getenv("CUOBJDUMP"), TempDir: t.TempDir()}
	v, err := tool.Version(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("version %s: %s%s", v.Path, v.Stdout, v.Stderr)
	for _, kind := range []Kind{ELF, PTX} {
		images, r, err := tool.List(ctx, p, kind)
		t.Logf("list: %s stderr: %s", r.Stdout, r.Stderr)
		if err != nil || len(images) != 1 {
			t.Fatalf("list: %v, %v", images, err)
		}
		artifacts, r, err := tool.Extract(ctx, p, kind)
		t.Logf("extract: %s stderr: %s", r.Stdout, r.Stderr)
		if err != nil || len(artifacts) != len(images) {
			t.Fatalf("extract: %d vs %d: %v", len(artifacts), len(images), err)
		}
		for _, a := range artifacts {
			got, want := a.Data, references[kind]
			if kind == PTX {
				// fatbinary strips comments and normalizes PTX whitespace. Compare
				// assembled code instead of treating formatting changes as corruption.
				got, want = assembledPTX(t, ctx, got, want)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("%s differs from reference: extracted %d bytes sha256 %x; reference %d bytes sha256 %x", a.Name, len(got), sha256.Sum256(got), len(want), sha256.Sum256(want))
			}
			t.Logf("verified %s (kind %d; PTX compared after assembly): %d bytes sha256 %x", a.Name, kind, len(got), sha256.Sum256(got))
		}
	}
	after, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(original, after) {
		t.Fatal("tool changed original input")
	}
	entries, err := os.ReadDir(tool.TempDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("extraction leaked %d scratch entries", len(entries))
	}
	for _, section := range []Section{SASS, ELFSections, Resources} {
		r, err := tool.Dump(ctx, p, section)
		t.Logf("dump %d: %s stderr: %s", section, r.Stdout, r.Stderr)
		if err != nil || strings.TrimSpace(string(r.Stdout)) == "" {
			t.Fatalf("dump: %v", err)
		}
	}
	bad := filepath.Join(t.TempDir(), "truncated.elf")
	if err := os.WriteFile(bad, []byte{0x7f, 'E', 'L', 'F'}, 0600); err != nil {
		t.Fatal(err)
	}
	_, diagnostic, err := tool.Extract(ctx, bad, ELF)
	if err == nil {
		t.Fatal("truncated ELF extraction succeeded")
	}
	if len(diagnostic.Stderr) == 0 {
		t.Fatalf("failed extraction lost tool diagnostics: %v", err)
	}
	t.Logf("invalid-input diagnostic: %s", diagnostic.Stderr)
}

// assembledPTX compiles both inputs with the same executable and options.
// A byte-identical output is stronger than a token or opcode-name comparison.
func assembledPTX(t *testing.T, ctx context.Context, a, b []byte) ([]byte, []byte) {
	t.Helper()
	tool := os.Getenv("PTXAS")
	if tool == "" {
		tool = "ptxas"
	}
	version, err := cudatool.Run(ctx, tool, "", nil, "--version")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("assembler %s: %s%s", version.Path, version.Stdout, version.Stderr)
	arch := os.Getenv("CUDAOBJDUMP_TEST_ARCH")
	if arch == "" {
		t.Fatal("CUDAOBJDUMP_TEST_ARCH is required for PTX assembly")
	}
	var compiled [2][]byte
	for i, src := range [][]byte{a, b} {
		dir := t.TempDir()
		in, out := filepath.Join(dir, "input.ptx"), filepath.Join(dir, "output.cubin")
		if err := os.WriteFile(in, bytes.TrimRight(src, "\x00"), 0600); err != nil {
			t.Fatal(err)
		}
		result, err := cudatool.Run(ctx, tool, "", nil, "--gpu-name", arch, in, "-o", out)
		t.Logf("assembler args %q stderr %s", result.Args, result.Stderr)
		if err != nil {
			t.Fatal(err)
		}
		compiled[i], err = os.ReadFile(out)
		if err != nil {
			t.Fatal(err)
		}
		if len(compiled[i]) == 0 {
			t.Fatal("assembler produced an empty cubin")
		}
	}
	return compiled[0], compiled[1]
}
