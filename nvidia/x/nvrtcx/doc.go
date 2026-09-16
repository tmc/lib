// Package nvrtcx provides runtime compilation of CUDA C++ source code
// into explicit compiler artifacts with content-addressed caching.
//
// WithOutput selects PTX, CUBIN, LTOIR, or TileIR. The default selects PTX
// and optional CUBIN. Available outputs depend on the loaded NVRTC library,
// source language, and compiler options. LTOIR requires linking before loading;
// see github.com/tmc/lib/nvidia/x/nvjitlinkx. TileIR requires a compatible driver.
//
// By default, Compile generates source line information (--generate-line-info)
// for profiling and debugging without optimization penalties. When source includes
// headers and no include paths are supplied, Compile automatically discovers and
// adds the CUDA toolkit include directory via ToolkitIncludeDir.
//
// Compiler options configure the compilation pass:
//   - Target architecture: [WithArch], [WithPhysicalArch], [WithDeviceArch]
//   - Language and execution space: [WithStandard], [WithDefaultDevice]
//   - Code generation and optimization: [WithFastMath], [WithOptimizationLevel], [WithExtraDeviceVectorization], [WithRelocatableDeviceCode], [WithMaxRegisters]
//   - Debugging and diagnostics: [WithLineInfo], [WithNoLineInfo], [WithDeviceDebug], [WithWarningsAsErrors], [WithDisableWarnings]
//   - Preprocessor and search paths: [WithIncludePaths], [WithToolkitIncludes], [WithNoToolkitIncludes], [WithDefines], [WithUndefines], [WithHeader]
//   - Output and artifacts: [WithOutput], [WithNameExpressions], [WithName], [WithNoCache], [WithFlag]
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
//		nvrtcx.WithStandard("c++17"),
//		nvrtcx.WithFastMath(),
//	)
//	if err != nil {
//		return err
//	}
//	loadedMod, err := mod.Load(dev)
package nvrtcx
