package cudainspect

import (
	"context"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestPairedPTXSASSComparison(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX subprocess fixture")
	}
	dir := t.TempDir()
	ptx, err := os.ReadFile("../cudalowering/testdata/lowering.sm90.ptx")
	if err != nil {
		t.Fatal(err)
	}
	sass, err := os.ReadFile("../cudalowering/testdata/lowering.sm90.sass")
	if err != nil {
		t.Fatal(err)
	}
	ptx = []byte(strings.ReplaceAll(string(ptx), "_Z8fmaChainPdPKdS1_i", "kernel"))
	sass = []byte(strings.ReplaceAll(string(sass), "_Z8fmaChainPdPKdS1_i", "kernel"))
	if err := os.WriteFile(filepath.Join(dir, "listing"), sass, 0600); err != nil {
		t.Fatal(err)
	}
	script := `#!/bin/sh
case "$1" in
--version) printf 'Cuda compilation tools, release 13.0, V13.0.88\n';;
--help) exit 0;;
*) cat "$(dirname "$0")/listing";;
esac
`
	path := filepath.Join(dir, "nvdisasm")
	if err := os.WriteFile(path, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	cubin := metadataELF(false, 0, elf.EM_CUDA)
	// CUDA ELF ABI 7 stores the SM in the low flags byte. This synthetic
	// known-target fixture must not rely on the unknown-ABI fallback.
	cubin[8] = 7
	binary.LittleEndian.PutUint32(cubin[48:], 90)
	i, _ := NewInspector(t.TempDir())
	tool := cudasass.Disassembler{Path: path}
	r, err := i.ComparePTX(context.Background(), tool, cubin, ptx, "kernel")
	if err != nil {
		t.Fatal(err)
	}
	if r.Comparison == nil || len(r.Comparison.Stats) == 0 || len(r.Comparison.Divergences) == 0 || r.Comparison.Err() == nil {
		t.Fatalf("lost static divergence: %+v", r)
	}
	if r.PTXDigest == "" || r.SASSDigest == "" || r.Version.Path != path || r.Disassembly.Path != path {
		t.Fatalf("lost provenance: %+v", r)
	}
	for _, target := range []string{"sm_121", "sm_90a"} {
		bad := []byte(strings.ReplaceAll(string(ptx), "sm_90", target))
		r, err := i.ComparePTX(context.Background(), tool, cubin, bad, "kernel")
		if err == nil || r.Comparison != nil || r.Version.Path != "" {
			t.Fatalf("target %s not refused before tool: %+v %v", target, r, err)
		}
	}
	// A changed listing for another artifact must surface unmatched kernels.
	other, err := os.ReadFile("../cudalowering/testdata/other.sm90.sass")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "listing"), other, 0600); err != nil {
		t.Fatal(err)
	}
	r, err = i.ComparePTX(context.Background(), tool, cubin, ptx, "kernel")
	if err != nil || r.Comparison == nil || len(r.Comparison.Stats) != 0 || r.Comparison.Err() == nil {
		t.Fatalf("unmatched artifacts hidden: %+v %v", r, err)
	}
	missing := cudasass.Disassembler{Path: filepath.Join(dir, "missing")}
	r, err = i.ComparePTX(context.Background(), missing, cubin, ptx, "kernel")
	if err == nil || r.Artifact == nil || r.Comparison != nil || len(r.Version.Args) == 0 {
		t.Fatalf("missing tool evidence: %+v %v", r, err)
	}
	failed := "#!/bin/sh\ncase \"$1\" in\n--version) printf 'Cuda compilation tools, release 13.0, V13.0.88\\n';;\n--help) exit 0;;\n*) printf 'disassembly failed' >&2; exit 4;;\nesac\n"
	if err := os.WriteFile(path, []byte(failed), 0700); err != nil {
		t.Fatal(err)
	}
	r, err = i.ComparePTX(context.Background(), tool, cubin, ptx, "kernel")
	if err == nil || r.Comparison != nil || string(r.Disassembly.Stderr) != "disassembly failed" {
		t.Fatalf("failed disassembly evidence: %+v %v", r, err)
	}
	entries, err := os.ReadDir(i.WorkDir)
	if err != nil || len(entries) != 0 {
		t.Fatal("scratch not cleaned")
	}
}

func ExampleInspector_ComparePTX() {
	i, _ := NewInspector(".")
	// Invalid artifact bytes are refused before any external command.
	_, err := i.ComparePTX(context.Background(), cudasass.Disassembler{}, []byte("invalid"), nil, "kernel")
	fmt.Println(err != nil)
	// Output: true
}
