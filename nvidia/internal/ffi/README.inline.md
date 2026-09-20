# ffi-eject inline mode

This package was ejected in inline mode from purego v0.11.0. It vendors purego's
syscallX trampoline and dlfcn loader stubs with a caller-provided symbol prefix,
so the package has no github.com/ebitengine/purego dependency and can coexist in
one binary with stock purego.

With -with-register, register_func.go supplies the local, error-returning
RegisterFunc API for linux/arm64 and darwin/arm64. This is tmc/ffi code copied
with its source provenance, not a patch to purego RegisterFunc. Small non-HFA
structs are copied as complete integer memory chunks, including a partial final
word. Unsupported signatures return an error and leave the destination unchanged;
consumers must check that error and must not fall back to stock registration.
Strings, slices, callbacks as arguments, HFA, structs over 16 bytes, pointer fields
in structs, and register exhaustion are unsupported. NewCallback is a separate
-with-callbacks tier. Go and C layouts must match, including padding.

The inline coexistence gate has run on darwin/amd64, darwin/arm64 and linux/arm64.
The loader regression covers runtime/cgo, fakecgo, and stock-purego coexistence.
On ELF the local assembly stubs are prefixed, but their C import names are shared
with cgo's linker symbols. The stubs remain enabled with cgo: excluding them would
make the loader call a data address. FreeBSD and NetBSD remain build-only targets.

By default inline mode requires CGO_ENABLED=1. It vendors the syscall trampoline and
dlfcn loader but not purego's fakecgo, so under CGO_ENABLED=0 there is no
runtime.cgocall bootstrap; the emitted vendor_cgo0_stub.go panics at init in that
configuration with a pointer to the fix.

-with-fakecgo lifts that restriction by vendoring purego's internal/fakecgo into
internal/fakecgo, which replaces the stub. Unlike the trampoline, that copy is not
symbol-prefixed and cannot be: the Go runtime resolves _cgo_init, runtime.iscgo and
runtime.set_crosscall2 by fixed name, so prefixing them disconnects the bootstrap
rather than relocating it. A binary therefore holds exactly one fakecgo, and an
ejected package built with this flag cannot coexist with stock purego — the link
fails with a duplicated-definition error naming the symbol. For a CGO_ENABLED=0
build that keeps purego as a dependency, use linkname mode instead.
