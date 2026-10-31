package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestDiffExplicitKernelPair(t *testing.T) {
	dir := t.TempDir()
	base, target := filepath.Join(dir, "base.sass"), filepath.Join(dir, "target.sass")
	for path, source := range map[string]string{
		base:   ".section .text.old,\"ax\",@progbits\n/*0000*/ MOV R0, 1 ;\n/*0010*/ EXIT ;\n.section .text.unrelated,\"ax\",@progbits\n/*0000*/ EXIT ;\n",
		target: ".section .text.generated,\"ax\",@progbits\n/*0000*/ MOV R0, 2 ;\n/*0010*/ EXIT ;\n",
	} {
		if err := os.WriteFile(path, []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
	}
	var out bytes.Buffer
	if err := runDiff(&out, []string{"--no-color", "--base-kernel=old", "--target-kernel=generated", base, target}); err != nil {
		t.Fatal(err)
	}
	text := out.String()
	if !strings.Contains(text, "old -> generated") || strings.Contains(text, "unrelated") || !strings.Contains(text, "no runtime equivalence") {
		t.Fatalf("unexpected paired output:\n%s", text)
	}
	for _, flags := range [][]string{
		{"--base-kernel=old"},
		{"--base-kernel="},
		{"--base-kernel=old", "--target-kernel="},
		{"--base-kernel=missing", "--target-kernel=generated"},
		{"--base-kernel=old", "--target-kernel=missing"},
		{"--base-kernel=old", "--target-kernel=generated", "--kernel=old"},
		{"--base-kernel=old", "--target-kernel=generated", "--explain"},
	} {
		if err := runDiff(&bytes.Buffer{}, append(flags, base, target)); err == nil {
			t.Errorf("accepted invalid selection %v", flags)
		}
	}
}

func TestDiffExactFunctionRejectsDuplicates(t *testing.T) {
	d := &cudasass.Disassembly{Functions: []*cudasass.Function{{Name: "x"}, {Name: "x"}}}
	if _, err := exactDiffFunction(d, "x"); err == nil {
		t.Fatal("accepted ambiguous symbol")
	}
}
