package cudalowering

import (
	"bytes"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvvmx"
)

func addressSpaceABInput() []byte {
	return []byte(`target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@shared_mem = internal addrspace(3) global i32 undef, align 4

define void @kernel(ptr addrspace(1) %out, i32 %input) {
entry:
  %slot = alloca ptr, align 8
  store i32 %input, ptr addrspace(3) @shared_mem, align 4
  store ptr addrspacecast (ptr addrspace(3) @shared_mem to ptr), ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
  %value = load i32, ptr %p, align 4
  %result = add i32 %value, 7
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{ptr @kernel, !"kernel", i32 1}
`)
}

func TestAddressSpaceABInputVerifiesAfterInference(t *testing.T) {
	before := addressSpaceABInput()
	if !bytes.Contains(before, []byte("addrspacecast (ptr addrspace(3) @shared_mem to ptr)")) {
		t.Fatal("A/B input has no shared address-space conversion")
	}
	verifyABIR(t, before, "before")
	inferred, _, err := nvvmx.InferAddressSpaces(before)
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile("testdata/addrspace_shared_inferred.ll")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(inferred, want) {
		t.Fatalf("inference output differs from checked-in fixture:\n--- got ---\n%s\n--- want ---\n%s", inferred, want)
	}
	assertABSlotValueTypes(t, inferred)
	verifyABIR(t, inferred, "inferred")
}

func assertABSlotValueTypes(t *testing.T, ir []byte) {
	t.Helper()
	for _, line := range strings.Split(string(ir), "\n") {
		if !strings.Contains(line, "%slot") || strings.Contains(line, "= alloca ") {
			continue
		}
		if strings.HasPrefix(strings.TrimSpace(line), "store ") && !strings.Contains(line, "store ptr addrspace(3) @") {
			t.Errorf("store into A/B slot is not a shared pointer: %s", line)
		}
		if strings.Contains(line, " = load ") && strings.Contains(line, ", ptr %slot") && !strings.Contains(line, "load ptr addrspace(3), ptr %slot") {
			t.Errorf("load from A/B slot is not shared-typed: %s", line)
		}
	}
}

func verifyABIR(t *testing.T, ir []byte, name string) {
	t.Helper()
	clang, err := exec.LookPath("clang")
	if err != nil {
		t.Skip("clang is not on PATH")
	}
	cmd := exec.Command(clang, "-x", "ir", "-S", "-emit-llvm", "-o", "/dev/null", "-")
	cmd.Stdin = bytes.NewReader(ir)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("clang rejected %s A/B input: %v\n%s\n%s", name, err, out, ir)
	}
}
