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

## Ordered vector reductions

LowerSemanticRegion supports direct-input rank-one/two FP32 sum and max, one
axis removed per output, with row-major contiguous inputs. TileM is the vector
width (default 16); it must be a positive power of two. Each vector lane owns an
independent output row or column. Grid is ceil(largest output count / TileM),1,1;
a zero output count has Grid.x=0 and requires no launch. TileN and TileK do not
configure this route. Inputs precede outputs in snapshot order in the pointer
ABI. Every memory operation has an explicit valid-lane mask and token chain.
Inactive lanes select the complete offset, including the reduction position,
to zero before pointer arithmetic. Zero-output branches emit no pointer, load
or store operations, even when another output requires a launch. Zero-term
reductions with positive outputs emit identities without input pointer formation
or loads. Empty bindings may therefore be null. Output bindings must
not alias inputs or other outputs during execution; this pointer ABI does not
carry sizes or validate runtime allocations.

Sum visits reduction positions in their original order, widens each load to
FP64, performs explicit nearest-even addition with preserved subnormals, and
casts once to FP32. Max initializes from the first value and uses ordered `>`
and select, retaining the first equal value and its signed zero. A later NaN
selects a quiet NaN; NaN payload and signaling-state parity are not promised.
Empty sum is positive zero and empty max is negative infinity. NaNs and
infinities are permitted by this route. Generic `ct::sum` is deliberately not
used because its unspecified grouping does not match the sequential reference.

The route emits documented Tile C++ full/iota, pointer arithmetic, load_masked,
store_masked, select, isnan, element_cast and add APIs, rather than a scalar
kernel per output. The analyzed AST records typed operands, masks, numerical
attributes and semantic source coordinates. Its verifier checks this emitted
SSA subset and token predecessors; it is not a vendor assembler. The existing
GenerateSemanticTileCPP scalar route remains separate; CompileSemanticTile
selects vector sum/max when applicable and the earlier source for other graphs.
Callers compiling vector reductions must use the corresponding LoweringResult
Grid when launching. Neither compilation nor source metadata submits a launch.

Source bounds remain 64 reduction terms, 1 Mi elements per tensor and 1 MiB
emitted source, with at most 256 semantic nodes. The power-of-two width budget
is not a hardware resource qualification. Composed reductions, other ranks,
types, layouts, multiple axes, attention and general optimized reductions remain
open. Offline Go controls interpret emitted typed SSA against cudaref, challenge
address/mask/token/precision omissions, and check documented source expressions.
These controls do not execute emitted C++ or establish NVIDIA compiler, FP64
Tile target support, resource usage, correctness on a device or performance.

API grounding (captured 2026-10-11):
[load/store](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/memory_operations.html),
[full/iota/select/isnan/element_cast](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/tile_operations.html),
[addition/comparison](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/arithmetic_operations.html),
[nearest-even tile conversions](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/general_principles.html).

Pointer formation is checked separately from masked dereference in the offline
SSA model. NVIDIA's pointer-addition rules follow C++ pointer arithmetic in every
lane; a false load mask cannot make an already invalid pointer addition safe.
See [pointer arithmetic](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/arithmetic_operations.html).
