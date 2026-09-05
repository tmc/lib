// Package nvvmx compiles NVVM IR with owned inputs and copied output.
// Compile returns PTX by default; the native -gen-lto option selects LTOIR
// for linking with nvjitlinkx. Lazy inputs can supply libdevice bitcode.
// The selected NVVM library determines supported IR versions and options.
// Compilation is synchronous and does not require a CUDA device context.
// Do not unload the native library while compilation or cleanup is pending.
package nvvmx
