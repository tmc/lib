target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; The typed-pointer form of sin.ll, for NVVM targets below LLVM 17.
; Keep the host-visible entry point alive through device LTO.
@llvm.used = appending global [1 x i8*] [i8* bitcast (void (float*)* @sine to i8*)], section "llvm.metadata"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare float @__nv_sinf(float)

define void @sine(float* %data) {
entry:
  %i = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %p = getelementptr float, float* %data, i32 %i
  %x = load float, float* %p, align 4
  %y = call float @__nv_sinf(float %x)
  store float %y, float* %p, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{void (float*)* @sine, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0}
