// Package cudaobjdump inspects CUDA code embedded in host binaries and cubins
// using the installed cuobjdump executable. It does not load a CUDA driver.
//
// Tool methods return command arguments, stdout and stderr on failure as well
// as success. Call Version to record the selected toolkit. Output formats and
// supported input architectures depend on that toolkit. Extract runs in a
// fresh private directory and returns owned byte slices, never tool-selected
// output paths in the caller's working directory.
package cudaobjdump
