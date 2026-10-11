# Semantic Tile source

GenerateSemanticTileCPP emits a finite Float32 reference kernel from an immutable
cudair snapshot. CompileSemanticTile uses NVRTC to produce an executable-format
TileIR artifact with the emitted ABI, target, flags, and compiler version.
Compilation does not create a CUDA context or launch the kernel.

Supported operations are add, sub, mul, div, neg, relu, scalar constants,
rank-one/two sum/max/mean reductions, and two-dimensional matmul. Inputs must be
contiguous and finite. Reduction and contraction extents are at most 64;
individual tensors at most 1 Mi elements. Each Tile block computes one output
element. This is a reference lowering, not a tensor-core optimization or a claim
about NaNs, signed zero, division accuracy, or performance.

Unsupported types, strided tensors, effects, non-scalar broadcasts, excessive
source expansion, and malformed shapes return errors. Online attention remains
unsupported: the semantic snapshot does not define its state, masks, scaling,
or empty-row behavior. It must not be interpreted as matmul.

The AST does not retain complete NVIDIA operation encoding. Module.ToBytecode
and Module.EncodeBytecode return ErrUnsupportedASTEncoding instead of producing
opcode-only bodies. The package-level EncodeBytecode preserves decoded vendor
bodies and remains separate. GeneratePinnedTileCPP deliberately fails compilation
because shape-only declarations cannot specify a kernel body.

The host tests execute emitted source against independent arithmetic expectations
using a C++ shim. Optional CUDA_TILE_COMPILE_TEST and CUDA_TILE_NVRTC_TEST gates
exercise actual NVIDIA compilation. Neither is physical execution proof.

References: [CUDA Tile C++](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/general_principles.html),
[arithmetic operations](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/arithmetic_operations.html),
[CUDA 13.4 Tile compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/tile-compilation-in-cuda.html).
