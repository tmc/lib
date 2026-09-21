# NVTX loader policy

The handwritten `loader_unix.go` opens the injection provider with
`RTLD_NOW | RTLD_GLOBAL`. Keep eager relocation: unresolved provider dependencies
should produce a load error before `initInjection` invokes the entry point.
On a load error, initialization returns with instrumentation disabled.
Global symbol visibility is retained for compatibility.

The generated `../../driver/loader_unix.gen.go` instead requests
`RTLD_LAZY | RTLD_GLOBAL`. Changing that to `RTLD_NOW` can reject a library with
unresolved function references even when those functions are never called.
There is no demonstrated driver defect requiring that compatibility change.
This policy concerns relocations inside the loaded library; it is separate from
deferring library loading until first use and from probing exported CUDA
functions with `dlsym`.

The driver file is marked as output of `driver-survey`. Its recipe is delegated
to `nvgen` in `tmc/libgen`; fixes belong in that generator and must be
regenerated, not patched into this checkout's generated files. NVTX's loader is
owned here.

`TestOpenDynamicLibrary` builds synthetic libraries on Linux and Darwin with
`cc`: a valid export must be callable and globally visible, and an unused
unresolved function reference must cause an open error. It skips if `cc` is
unavailable. This tests the host loader, not CUDA or Nsight compatibility.
