package cudashared

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprepack"
)

// Q8CopyFixture is a synchronous Q8_0 payload-and-scale consumer. Launch
// Source's staged_q8_copy entry with one-dimensional grid and blocks using
// GridBlocks, BlockThreads and DynamicBytes. Arguments are payload, scales,
// and output device pointers, in that order. Output holds standard 34-byte
// Q8_0 blocks; no scale conversion or arithmetic is performed.
//
// PrepackReadBytes and PrepackWriteBytes charge the preceding block repack;
// WorkspaceBytes includes its payload and scale storage. The kernel reads
// and writes RawBytes each. These are byte counts, not timing evidence.
// The caller owns nonaliasing device buffers through completion and configures
// any required dynamic-shared opt-in. Source is not native qualification.
type Q8CopyFixture struct {
	Source                              string
	GridBlocks, BlockThreads            int
	DynamicBytes                        int64
	RawBytes                            int
	PrepackReadBytes, PrepackWriteBytes int
	WorkspaceBytes                      int
}

// StagedQ8CopyFixture consumes stage0 of plan and the checked host layout of
// tensor. It uses ordinary byte loads and a CTA barrier, not a tensor map or
// TMA swizzle. Target limits must exist as probed device records. Only dynamic
// alignments up to 128 bytes are supported by this source fixture.
func StagedQ8CopyFixture(plan *Plan, tensor *cudaprepack.PrepackedTensor) (Q8CopyFixture, error) {
	if plan == nil {
		return Q8CopyFixture{}, fmt.Errorf("nil shared plan")
	}
	if plan.StaticBytes() != 0 {
		return Q8CopyFixture{}, fmt.Errorf("q8 copy supports only the dynamic shared pool")
	}
	storage, err := cudaprepack.DescribeQ8(tensor)
	if err != nil {
		return Q8CopyFixture{}, fmt.Errorf("q8 copy storage: %w", err)
	}
	if tensor.Arch != "sm_100a" && tensor.Arch != "sm_121a" {
		return Q8CopyFixture{}, fmt.Errorf("unsupported q8 copy target %q", tensor.Arch)
	}
	limits, err := LimitsFromDevice(tensor.Arch)
	if err != nil {
		return Q8CopyFixture{}, fmt.Errorf("q8 copy target: %w", err)
	}
	if plan.StaticBytes() > limits.StaticMax || plan.DynamicBytes() > limits.DynamicMax || plan.FootprintBytes() > limits.TotalMax {
		return Q8CopyFixture{}, fmt.Errorf("q8 copy exceeds target shared-memory limits")
	}
	var region Region
	alignment := int64(1)
	for _, r := range plan.Regions() {
		if r.Dynamic && r.Alignment > alignment {
			alignment = r.Alignment
		}
		if r.Name == "stage0" {
			region = r
		}
	}
	if !region.Dynamic || region.Size < int64(storage.RawBytes) {
		return Q8CopyFixture{}, fmt.Errorf("dynamic stage0 is too small for q8 copy")
	}
	if alignment > 128 {
		return Q8CopyFixture{}, fmt.Errorf("q8 copy shared alignment exceeds supported 128 bytes")
	}
	if err := plan.ValidateBase(0); err != nil {
		return Q8CopyFixture{}, err
	}
	threads := min(128, storage.RawBytes)
	blocks := (storage.RawBytes-1)/threads + 1
	source := fmt.Sprintf(`// cudashared Q8_0 target=%s shared_offset=%d shared_bytes=%d
extern "C" __global__ __launch_bounds__(%d) void staged_q8_copy(
    const unsigned char *payload, const unsigned char *scales,
    unsigned char *out) {
  extern __shared__ __align__(%d) unsigned char raw[];
  unsigned char *tile = raw + %d;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < %d) {
    size_t block = i / 34;
    size_t byte = i %% 34;
    size_t row = block / %d;
    size_t column = block %% %d;
    tile[threadIdx.x] = byte < 2 ? scales[block * 2 + byte]
        : payload[row * %d + column * 32 + byte - 2];
  }
  __syncthreads();
  if (i < %d) out[i] = tile[threadIdx.x];
}
`, tensor.Arch, region.Offset, plan.DynamicBytes(), threads, alignment,
		region.Offset, storage.RawBytes, storage.Cols/32, storage.Cols/32,
		storage.RowStride, storage.RawBytes)
	return Q8CopyFixture{
		Source: source, GridBlocks: blocks, BlockThreads: threads,
		DynamicBytes: plan.DynamicBytes(), RawBytes: storage.RawBytes,
		PrepackReadBytes: storage.RawBytes, PrepackWriteBytes: storage.WorkspaceBytes,
		WorkspaceBytes: storage.WorkspaceBytes,
	}, nil
}
