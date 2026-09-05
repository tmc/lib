// Package nvrtcx provides runtime compilation of CUDA C++ source code
// into explicit compiler artifacts with content-addressed caching.
//
// WithOutput selects PTX, CUBIN, LTOIR, or TileIR. The default selects PTX
// and optional CUBIN. Available outputs depend on the loaded NVRTC library,
// source language, and compiler options. LTOIR requires linking before loading;
// see github.com/tmc/lib/nvidia/x/nvjitlinkx. TileIR requires a compatible driver.
//
// Cache keys include source, program name, in-memory headers, flags, output
// selection, and the NVRTC version and library path. They do not hash library
// contents or headers read from external include paths. Use WithNoCache when
// those inputs can change without changing the key.
//
// # Usage
//
// Illustrative fragment; sourceCode and dev are supplied by the caller:
//
//	mod, err := nvrtcx.Compile(sourceCode,
//		nvrtcx.WithArch(8, 0),
//		nvrtcx.WithFlag("--use_fast_math"),
//	)
//	if err != nil {
//		return err
//	}
//	loadedMod, err := mod.Load(dev)
package nvrtcx
