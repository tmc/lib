target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

; Keep the host-visible entry point alive through device LTO.
@llvm.used = appending global [1 x ptr] [ptr @sine], section "llvm.metadata"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare float @__nv_sinf(float)

define ptx_kernel void @sine(ptr %data) {
entry:
  %i = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %p = getelementptr float, ptr %data, i32 %i
  %x = load float, ptr %p, align 4
  %y = call float @__nv_sinf(float %x)
  store float %y, ptr %p, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @sine, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 2}
