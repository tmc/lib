target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@shared_mem = internal addrspace(3) global i32 undef, align 4

define void @kernel(ptr addrspace(1) %out, i32 %input) {
entry:
  %slot = alloca ptr addrspace(3), align 8
  store i32 %input, ptr addrspace(3) @shared_mem, align 4
  store ptr addrspace(3) @shared_mem, ptr %slot, align 8
  %p = load ptr addrspace(3), ptr %slot, align 8
  %value = load i32, ptr addrspace(3) %p, align 4
  %result = add i32 %value, 7
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{ptr @kernel, !"kernel", i32 1}
