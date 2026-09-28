package cudalowering

import (
	"bytes"
	"os/exec"
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
  store ptr addrspace(3) @shared_mem, ptr %slot, align 8
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
	clang, err := exec.LookPath("clang")
	if err != nil {
		t.Skip("clang is not on PATH")
	}
	inferred, _, err := nvvmx.InferAddressSpaces(addressSpaceABInput())
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(clang, "-x", "ir", "-S", "-emit-llvm", "-o", "/dev/null", "-")
	cmd.Stdin = bytes.NewReader(inferred)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("clang rejected inferred A/B input: %v\n%s\n%s", err, out, inferred)
	}
}
