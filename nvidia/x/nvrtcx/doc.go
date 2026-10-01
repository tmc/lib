// Package nvrtcx provides runtime compilation of CUDA C++ source code
// into explicit compiler artifacts with content-addressed caching.
//
// WithOutput selects PTX, CUBIN, LTOIR, TileIR, or OptiXIR. The default selects PTX
// and optional CUBIN. Available outputs depend on the loaded NVRTC library,
// source language, and compiler options. LTOIR requires linking before loading;
// see github.com/tmc/lib/nvidia/x/nvjitlinkx. TileIR requires a compatible driver.
// OptiX IR must be loaded via optixModuleCreate.
//
// By default, Compile generates source line information (--generate-line-info)
// for profiling and debugging without optimization penalties. When source includes
// headers and no include paths are supplied, Compile automatically discovers and
// adds the CUDA toolkit include directory via ToolkitIncludeDir, or extracts and
// uses bundled headers via InstallBundledHeaders when running without a local toolkit.
//
// Compilation supports context cancellation via [WithContext] and [CompileContext],
// aborting in-flight compilation via NVRTC flow callbacks.
//
// Compiler options configure the compilation pass:
//   - Target architecture: [WithArch], [WithPhysicalArch], [WithDeviceArch]
//   - Language and execution space: [WithStandard], [WithDefaultDevice]
//   - Code generation and optimization: [WithFastMath], [WithFastCompile], [WithOptimizationLevel], [WithExtraDeviceVectorization], [WithRelocatableDeviceCode], [WithExtensibleWholeProgram], [WithMaxRegisters], [WithRandomSeed], [WithDeviceStackProtector]
//   - Precompiled headers: [WithPCH], [WithCreatePCH], [WithUsePCH]
//   - Debugging and diagnostics: [WithLineInfo], [WithNoLineInfo], [WithDeviceDebug], [WithWarningsAsErrors], [WithDisableWarnings], [WithTimeTrace]
//   - Preprocessor and search paths: [WithIncludePaths], [WithToolkitIncludes], [WithNoToolkitIncludes], [WithBundledHeaders], [WithDefines], [WithUndefines], [WithHeader]
//   - Context and cancellation: [WithContext], [CompileContext]
//   - Output and artifacts: [WithOutput], [WithNameExpressions], [WithName], [WithNoCache], [WithFlag]
//
// Cache keys include source, program name, in-memory headers, flags, output
// selection, and the NVRTC version and library path. They do not hash library
// contents or headers read from external include paths. Use WithNoCache when
// those inputs can change without changing the key.
//
// # Spliced Device Functions
//
// [SpliceAndCompile] sends the fully spliced source through the same cache.
// The key also includes name expressions and tracked external dependency
// contents; caching is disabled when an external dependency cannot be tracked.
// The compiler identity includes the NVRTC version and path, output kind, and
// cache format. JIT options are not part of the key even though SpliceAndCompile
// forwards them and the returned CompiledModule retains them. Callers that use
// different JIT options with the same compile inputs should disable the cache
// until that identity boundary includes JIT options.
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
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package nvrtcx
