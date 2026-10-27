// Package tilebridge lowers row and GEMM kernel specifications into NVIDIA
// CUDA Tile IR modules, formats and parses textual Tile IR, encodes Tile IR
// bytecode, evaluates lowered modules with a reference SSA interpreter, and
// verifies numerical agreement against CUDA C++ host emulation.
//
// Basic usage:
//
//	rk := tilebridge.NewRMSNormKernel("rmsnorm_f32", 64, 1e-5)
//	mod, err := tilebridge.StepToTileIR(tilebridge.Step{Row: &rk})
//	bc, err := tilebridge.EncodeTileIR(mod)
package tilebridge
