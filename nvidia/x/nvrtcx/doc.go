// Package nvrtcx provides high-level runtime compilation (JIT) of CUDA C++ source code
// into PTX or CUBIN binaries with content-addressed caching.
//
// It wraps the purego dynamic bindings in [github.com/tmc/lib/nvidia/nvrtc] and provides
// automatic caching (keyed by source hash, include headers, and compiler flags) in memory
// and persistent storage to eliminate redundant JIT compilation overhead.
//
// # Usage
//
//	mod, err := nvrtcx.Compile(ctx, sourceCode,
//		nvrtcx.WithArch(8, 0),
//		nvrtcx.WithFlag("--use_fast_math"),
//	)
//	if err != nil {
//		return err
//	}
//	loadedMod, err := mod.Load(dev)
package nvrtcx
