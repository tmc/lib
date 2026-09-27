// Package nvjitlinkx links CUDA device-code inputs with nvJitLink.
// Link owns the native linker, copied inputs, and their Go pins until destruction.
// Returned bytes and diagnostics remain valid after Link returns. Callers must
// not unload the raw nvjitlink library while a link is in progress.
//
// This package links code; it does not load modules or fuse kernel launches.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package nvjitlinkx
