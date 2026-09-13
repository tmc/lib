// Package cudaobjdump inspects CUDA code embedded in host binaries and cubins
// using the installed cuobjdump executable. It does not load a CUDA driver.
//
// Tool methods return command arguments, stdout and stderr on failure as well
// as success. Call Version to record the selected toolkit. Output formats and
// supported input architectures depend on that toolkit. Extract runs in a
// fresh private directory and returns owned byte slices, never tool-selected
// output paths in the caller's working directory.
//
// # Toolkit versions
//
// Every option this package emits is accepted, with the same arity, by CUDA
// 12.6, 13.0 and 13.4: cuobjdump's option names and arities are identical
// across them, and the only measured difference is which architectures
// --gpu-architecture enumerates, which this package never passes. So no
// command line here is gated on a toolkit version. TestOptionsAgainstRecordedToolkits
// holds that: it checks the options the package emits against every toolkit the
// option corpus in x/internal/cudatool records, and a version that starts to
// differ fails there rather than in a caller's command line.
package cudaobjdump
