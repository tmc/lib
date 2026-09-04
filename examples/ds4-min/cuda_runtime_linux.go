//go:build linux

package main

import (
	"errors"
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const cudaProbeSource = `
extern "C" __global__ void ds4_probe(float *p) {
  if (blockIdx.x == 0 && threadIdx.x == 0) p[0] = 4.0f;
}

__device__ float ds4_half_to_float(unsigned short h) {
  const unsigned int sign = unsigned(h & 0x8000) << 16;
  unsigned int exp = (h >> 10) & 0x1f;
  unsigned int frac = h & 0x03ff;
  if (exp == 0) {
    if (frac == 0) return __uint_as_float(sign);
    int e = -14;
    while ((frac & 0x0400) == 0) { frac <<= 1; e--; }
    frac &= 0x03ff;
    return __uint_as_float(sign | unsigned(e + 127) << 23 | frac << 13);
  }
  if (exp == 0x1f)
    return __uint_as_float(sign | 0x7f800000 | frac << 13);
  return __uint_as_float(sign | (exp - 15 + 127) << 23 | frac << 13);
}

extern "C" __global__ void ds4_read_u16(const unsigned short *p, unsigned int *out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = p[0];
}

extern "C" __global__ void ds4_read_u8(const unsigned char *p, unsigned long long offset, unsigned int *out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = p[offset];
}

// ds4_q8_0_matvec computes one output row per CUDA block directly from GGUF
// Q8_0 blocks. The input and output are ordinary device allocations owned by
// the caller.
extern "C" __global__ void ds4_q8_0_matvec(
    const unsigned char *weights, const float *x, float *out,
    unsigned int cols, unsigned int rows) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float partial[256];
  float sum = 0.0f;
  const unsigned int blocks_per_row = cols / 32;
  const unsigned char *row_weights = weights + row * blocks_per_row * 34;
  for (unsigned int block = threadIdx.x; block < blocks_per_row; block += blockDim.x) {
    const unsigned char *p = row_weights + block * 34;
    const unsigned short scale_bits = unsigned(p[0]) | unsigned(p[1]) << 8;
    const float scale = ds4_half_to_float(scale_bits);
    const signed char *q = reinterpret_cast<const signed char *>(p + 2);
    for (unsigned int i = 0; i < 32; i++)
      sum += scale * float(q[i]) * x[block * 32 + i];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = partial[0];
}

extern "C" __global__ void ds4_rms(
    const float *x, const float *weight, float *out,
    unsigned int width, unsigned int groups, unsigned int weighted) {
  const unsigned int group = blockIdx.x;
  if (group >= groups) return;
  __shared__ float partial[256];
  const unsigned int offset = group * width;
  float sum = 0.0f;
  for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = x[offset + i];
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(partial[0] / float(width) + 1.0e-6f);
  for (unsigned int i = threadIdx.x; i < width; i += blockDim.x)
    out[offset + i] = x[offset + i] * inv * (weighted ? weight[i] : 1.0f);
}

extern "C" __global__ void ds4_f16_matvec(
    const unsigned short *weights, const float *x, float *out,
    unsigned int cols, unsigned int rows) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float partial[256];
  float sum = 0.0f;
  const unsigned long long offset = (unsigned long long)row * cols;
  for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x)
    sum += ds4_half_to_float(weights[offset + i]) * x[i];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = partial[0];
}

// ds4_hc_pre applies the four-stream hyper-connection input transform and
// writes the 4096-value current stream followed by its weighted RMS norm.
// Only the first four projection rows contribute to the input transform. The
// remaining rows describe the output transform and are consumed by HCPost.
extern "C" __global__ void ds4_hc_pre(
    const float *state, const unsigned short *hc_fn, const float *scale,
    const float *base, const float *norm_weight, float *out) {
  if (blockIdx.x != 0) return;
  __shared__ float partial[256];
  __shared__ float inv;
  __shared__ float pre[4];
  float ss = 0.0f;
  for (unsigned int i = threadIdx.x; i < 16384; i += blockDim.x)
    ss += state[i] * state[i];
  partial[threadIdx.x] = ss;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) inv = rsqrtf(partial[0] / 16384.0f + 1.0e-6f);
  __syncthreads();
  for (unsigned int row = 0; row < 4; row++) {
    float sum = 0.0f;
    const unsigned long long off = (unsigned long long)row * 16384;
    for (unsigned int i = threadIdx.x; i < 16384; i += blockDim.x)
      sum += ds4_half_to_float(hc_fn[off + i]) * state[i] * inv;
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
      if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      pre[row] = 1.0f / (1.0f + expf(-(partial[0] * scale[0] + base[row]))) + 1.0e-6f;
    __syncthreads();
  }
  float cur_ss = 0.0f;
  for (unsigned int d = threadIdx.x; d < 4096; d += blockDim.x) {
    float v = 0.0f;
    for (unsigned int h = 0; h < 4; h++) v += state[h * 4096 + d] * pre[h];
    out[d] = v;
    cur_ss += v * v;
  }
  partial[threadIdx.x] = cur_ss;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) inv = rsqrtf(partial[0] / 4096.0f + 1.0e-6f);
  __syncthreads();
  for (unsigned int d = threadIdx.x; d < 4096; d += blockDim.x)
    out[4096 + d] = out[d] * inv * norm_weight[d];
}`

const (
	cudaMemHostRegisterDeviceMap = 2
	cudaMaxRegisteredModelRange  = 8 << 20
)

type cudaModelRange struct {
	host       uintptr
	device     driver.DevicePtr
	bytes      uint
	fileOffset uint64
	view       []byte
}

type cudaModelTensor struct {
	name       string
	typeID     uint32
	dimensions []uint64
	device     driver.DevicePtr
	fileOffset uint64
	bytes      uint64
}

// cudaFoundation owns the resources needed to begin a CUDA model runtime.
// It deliberately does not implement sessionBackend until token evaluation is
// available on CUDA.
type cudaFoundation struct {
	device *cuda.Device
	file   *os.File
	model  []byte
	name   string
	major  int
	minor  int
	hmm    bool
	gguf   *ggufFile
	mapped cudaModelRange

	probeTensor string
	probeHost   uint16
	probeDevice uint16
	probeCols   uint32
	probeRows   uint32
	probeBytes  uint64
	probeOffset uint64
	probePtr    driver.DevicePtr
	matrixMod   driver.Module
	matrixFn    driver.Function
	rmsFn       driver.Function
	f16Fn       driver.Function
	hcPreFn     driver.Function
}

func newCUDAFoundation(path string) (_ *cudaFoundation, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cuda: open model: %w", err)
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("cuda: stat model: %w", err)
	}
	if info.Size() != pinnedModelBytes {
		return nil, fmt.Errorf("cuda: model size %d, want %d", info.Size(), int64(pinnedModelBytes))
	}
	g, err := readGGUF(f, info.Size())
	if err != nil {
		return nil, fmt.Errorf("cuda: read model: %w", err)
	}
	if err := validatePinnedProfile(g); err != nil {
		return nil, fmt.Errorf("cuda: model profile: %w", err)
	}
	model, err := syscall.Mmap(int(f.Fd()), 0, int(info.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("cuda: map model: %w", err)
	}
	defer func() {
		if err != nil {
			syscall.Munmap(model)
		}
	}()

	dev, err := cuda.NewDevice(0)
	if err != nil {
		return nil, fmt.Errorf("cuda: device: %w", err)
	}
	defer func() {
		if err != nil {
			dev.Close()
		}
	}()
	name, err := dev.Name()
	if err != nil {
		return nil, err
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, err
	}
	hmm, err := cudaSupportsHostPageTables(dev)
	if err != nil {
		return nil, err
	}
	r := &cudaFoundation{device: dev, file: f, model: model, name: name, major: major, minor: minor, hmm: hmm, gguf: g}
	if err := r.probe(); err != nil {
		return nil, err
	}
	if err := r.probeModel(g); err != nil {
		if r.mapped.host != 0 {
			_ = dev.WithContext(func() error { return driver.MemHostUnregister(r.mapped.host) })
			_ = syscall.Munmap(r.mapped.view)
			r.mapped = cudaModelRange{}
		}
		return nil, err
	}
	return r, nil
}

// cudaSupportsHostPageTables reports whether ordinary system allocations use
// the same virtual addresses on the CPU and GPU. All four properties are
// required: accepting a host pointer based only on UVA is not sufficient.
func cudaSupportsHostPageTables(dev *cuda.Device) (bool, error) {
	attrs := [...]driver.DeviceAttribute{
		driver.DevAttrIntegrated,
		driver.DevAttrUnifiedAddressing,
		driver.DevAttrPageableMemoryAccess,
		driver.DevAttrPageableMemoryAccessUsesHostPageTables,
	}
	for _, attr := range attrs {
		value, err := dev.Attribute(attr)
		if err != nil {
			return false, fmt.Errorf("cuda: query HMM capability %v: %w", attr, err)
		}
		if value == 0 {
			return false, nil
		}
	}
	return true, nil
}

// modelRange returns a device view of bytes in the stable GGUF mapping. On an
// integrated device using the host page tables this is the mmap address itself
// and requires neither registration nor a CUDA allocation. Other devices use
// the bounded registered fallback.
func (r *cudaFoundation) modelRange(fileOffset, length uint64) (driver.DevicePtr, error) {
	if length == 0 {
		return 0, fmt.Errorf("cuda: empty model range")
	}
	if fileOffset > uint64(len(r.model)) || length > uint64(len(r.model))-fileOffset {
		return 0, fmt.Errorf("cuda: model range [%d,%d) outside mapping", fileOffset, fileOffset+length)
	}
	if r.hmm {
		return driver.DevicePtr(uintptr(unsafe.Pointer(&r.model[fileOffset]))), nil
	}
	return r.mapModelRange(fileOffset, length)
}

func (r *cudaFoundation) tensor(name string) (cudaModelTensor, error) {
	if r.gguf == nil {
		return cudaModelTensor{}, fmt.Errorf("cuda: model is closed")
	}
	for _, tensor := range r.gguf.tensors {
		if tensor.name != name {
			continue
		}
		bytes, err := tensorBytes(tensor)
		if err != nil {
			return cudaModelTensor{}, fmt.Errorf("cuda: size tensor %q: %w", name, err)
		}
		offset := r.gguf.dataOffset + tensor.offset
		ptr, err := r.modelRange(offset, bytes)
		if err != nil {
			return cudaModelTensor{}, fmt.Errorf("cuda: bind tensor %q: %w", name, err)
		}
		return cudaModelTensor{name: name, typeID: tensor.typeID, dimensions: append([]uint64(nil), tensor.dimensions...), device: ptr, fileOffset: offset, bytes: bytes}, nil
	}
	return cudaModelTensor{}, fmt.Errorf("cuda: model has no tensor %q", name)
}

func (r *cudaFoundation) probe() error {
	compiled, err := nvrtcx.Compile(cudaProbeSource,
		nvrtcx.WithName("ds4_probe.cu"),
		nvrtcx.WithArch(r.major, r.minor),
	)
	if err != nil {
		return fmt.Errorf("cuda: compile probe: %w", err)
	}
	return r.device.WithContext(func() error {
		mod, err := compiled.Load(r.device)
		if err != nil {
			return fmt.Errorf("cuda: load probe: %w", err)
		}
		defer mod.Unload()
		fn, err := mod.GetFunction("ds4_probe")
		if err != nil {
			return fmt.Errorf("cuda: find probe: %w", err)
		}
		buf, err := cudamem.Alloc[float32](r.device, 1)
		if err != nil {
			return fmt.Errorf("cuda: allocate probe: %w", err)
		}
		defer buf.Free()
		if err := cudalaunch.Kernel(fn).Launch(buf); err != nil {
			return fmt.Errorf("cuda: launch probe: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize probe: %w", err)
		}
		var got [1]float32
		if err := buf.CopyToHost(got[:]); err != nil {
			return fmt.Errorf("cuda: read probe: %w", err)
		}
		if got[0] != 4 {
			return fmt.Errorf("cuda: probe result %g, want 4", got[0])
		}
		return nil
	})
}

// mapModelRange makes a small, page-aligned part of the GGUF directly
// addressable by CUDA. It uses a private writable mapping because GB10 does not
// advertise CUDA's read-only host-registration extension. Device writes cannot
// modify the model file. Large ranges are rejected: pinning the entire 86 GB
// model would deprive the host of pageable memory.
func (r *cudaFoundation) mapModelRange(fileOffset, length uint64) (driver.DevicePtr, error) {
	if length == 0 || length > cudaMaxRegisteredModelRange {
		return 0, fmt.Errorf("cuda: model range length %d exceeds limit %d", length, cudaMaxRegisteredModelRange)
	}
	if fileOffset > uint64(len(r.model)) || length > uint64(len(r.model))-fileOffset {
		return 0, fmt.Errorf("cuda: model range [%d,%d) outside mapping", fileOffset, fileOffset+length)
	}
	if r.mapped.host != 0 {
		return 0, fmt.Errorf("cuda: model range already mapped")
	}
	page := uint64(syscall.Getpagesize())
	start := fileOffset &^ (page - 1)
	end := (fileOffset + length + page - 1) &^ (page - 1)
	if end > uint64(len(r.model)) {
		end = uint64(len(r.model))
	}
	bytes := uint(end - start)
	view, err := syscall.Mmap(int(r.file.Fd()), int64(start), int(bytes), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE)
	if err != nil {
		return 0, fmt.Errorf("cuda: map private model range: %w", err)
	}
	host := uintptr(unsafe.Pointer(&view[0]))
	err = r.device.WithContext(func() error {
		if err := driver.MemHostRegister(host, bytes, cudaMemHostRegisterDeviceMap); err != nil {
			return fmt.Errorf("register model range: %w", err)
		}
		device, err := driver.MemHostGetDevicePointer(host, 0)
		if err != nil {
			_ = driver.MemHostUnregister(host)
			return fmt.Errorf("get model device pointer: %w", err)
		}
		r.mapped = cudaModelRange{host: host, device: device, bytes: bytes, fileOffset: start, view: view}
		return nil
	})
	if err != nil {
		_ = syscall.Munmap(view)
		return 0, fmt.Errorf("cuda: map model range: %w", err)
	}
	return r.mapped.device + driver.DevicePtr(fileOffset-start), nil
}

func (r *cudaFoundation) probeModel(g *ggufFile) error {
	var tensor *ggufTensor
	for i := range g.tensors {
		if g.tensors[i].name == "blk.0.attn_kv.weight" {
			tensor = &g.tensors[i]
			break
		}
	}
	if tensor == nil {
		return fmt.Errorf("cuda: model has no tensor %q", "blk.0.attn_kv.weight")
	}
	if tensor.typeID != 8 || len(tensor.dimensions) != 2 || tensor.dimensions[0]%32 != 0 {
		return fmt.Errorf("cuda: tensor %q is not a Q8_0 matrix", tensor.name)
	}
	tensorBytes, err := tensorBytes(*tensor)
	if err != nil {
		return fmt.Errorf("cuda: size tensor %q: %w", tensor.name, err)
	}
	if tensorBytes > cudaMaxRegisteredModelRange {
		return fmt.Errorf("cuda: Q8_0 tensor %q is %d bytes, limit %d", tensor.name, tensorBytes, cudaMaxRegisteredModelRange)
	}
	offset := g.dataOffset + tensor.offset
	want := uint16(r.model[offset]) | uint16(r.model[offset+1])<<8
	ptr, err := r.modelRange(offset, tensorBytes)
	if err != nil {
		return fmt.Errorf("cuda: bind tensor %q: %w", tensor.name, err)
	}
	compiled, err := nvrtcx.Compile(cudaProbeSource,
		nvrtcx.WithName("ds4_model_probe.cu"),
		nvrtcx.WithArch(r.major, r.minor),
	)
	if err != nil {
		return fmt.Errorf("cuda: compile model probe: %w", err)
	}
	return r.device.WithContext(func() error {
		mod, err := compiled.Load(r.device)
		if err != nil {
			return fmt.Errorf("cuda: load model probe: %w", err)
		}
		defer mod.Unload()
		fn, err := mod.GetFunction("ds4_read_u16")
		if err != nil {
			return fmt.Errorf("cuda: find model probe: %w", err)
		}
		out, err := cudamem.Alloc[uint32](r.device, 1)
		if err != nil {
			return fmt.Errorf("cuda: allocate model probe: %w", err)
		}
		defer out.Free()
		if err := cudalaunch.Kernel(fn).Launch(ptr, out); err != nil {
			return fmt.Errorf("cuda: launch model probe: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize model probe: %w", err)
		}
		var got [1]uint32
		if err := out.CopyToHost(got[:]); err != nil {
			return fmt.Errorf("cuda: read model probe: %w", err)
		}
		if got[0] != uint32(want) {
			return fmt.Errorf("cuda: model probe result %#x, want %#x", got[0], want)
		}
		r.probeTensor = tensor.name
		r.probeHost = want
		r.probeDevice = uint16(got[0])
		r.probeCols = uint32(tensor.dimensions[0])
		r.probeRows = uint32(tensor.dimensions[1])
		r.probeBytes = tensorBytes
		r.probeOffset = offset
		r.probePtr = ptr
		return nil
	})
}

// q80Matvec multiplies the registered Q8_0 matrix by x. The caller owns both
// x and out; the foundation owns only the bounded registered model tensor.
func (r *cudaFoundation) q80Matvec(x, out cudamem.Slice[float32]) error {
	if r.probePtr == 0 || r.probeCols == 0 || r.probeRows == 0 {
		return fmt.Errorf("cuda: Q8_0 matrix is not bound")
	}
	if x.Len() != int(r.probeCols) || out.Len() != int(r.probeRows) {
		return fmt.Errorf("cuda: Q8_0 matrix input/output lengths %d/%d, want %d/%d", x.Len(), out.Len(), r.probeCols, r.probeRows)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.matrixFn).Grid(r.probeRows).Block(256).Launch(r.probePtr, x, out, r.probeCols, r.probeRows); err != nil {
			return fmt.Errorf("cuda: launch Q8_0 matrix: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize Q8_0 matrix: %w", err)
		}
		return nil
	})
}

// encodeQ80Matvec appends a matrix-vector product to the foundation's CUDA
// stream. It does not synchronize or copy results to the host.
func (r *cudaFoundation) encodeQ80Matvec(t cudaModelTensor, x, out cudamem.Slice[float32]) error {
	if t.typeID != 8 || len(t.dimensions) != 2 || t.dimensions[0]%32 != 0 ||
		uint64(x.Len()) != t.dimensions[0] || uint64(out.Len()) != t.dimensions[1] {
		return fmt.Errorf("cuda: matvec %q: invalid Q8_0 tensor or input", t.name)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.matrixFn).Grid(uint32(t.dimensions[1])).Block(256).On(r.device.DefaultStream()).Launch(
			t.device, x, out, uint32(t.dimensions[0]), uint32(t.dimensions[1])); err != nil {
			return fmt.Errorf("cuda: launch matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeRMS appends grouped RMS normalization to the foundation's CUDA
// stream. Weight must be an F32 vector when weighted is true.
func (r *cudaFoundation) encodeRMS(x cudamem.Slice[float32], weight cudaModelTensor, out cudamem.Slice[float32], groups int, weighted bool) error {
	if x.Len() == 0 || out.Len() != x.Len() || groups <= 0 || x.Len()%groups != 0 ||
		(weighted && (weight.typeID != 0 || len(weight.dimensions) != 1 || weight.dimensions[0] != uint64(x.Len()/groups))) {
		return fmt.Errorf("cuda: RMS: invalid input, weight, or groups")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		var weightPtr driver.DevicePtr
		if weighted {
			weightPtr = weight.device
		}
		weightedArg := uint32(0)
		if weighted {
			weightedArg = 1
		}
		if err := cudalaunch.Kernel(r.rmsFn).Grid(uint32(groups)).Block(256).On(r.device.DefaultStream()).Launch(
			x, weightPtr, out, uint32(x.Len()/groups), uint32(groups), weightedArg); err != nil {
			return fmt.Errorf("cuda: launch RMS: %w", err)
		}
		return nil
	})
}

// encodeF16Matvec appends an F16 matrix-vector product to the default stream.
func (r *cudaFoundation) encodeF16Matvec(t cudaModelTensor, x, out cudamem.Slice[float32]) error {
	if t.typeID != 1 || len(t.dimensions) != 2 || uint64(x.Len()) != t.dimensions[0] || uint64(out.Len()) != t.dimensions[1] {
		return fmt.Errorf("cuda: matvec %q: invalid F16 tensor or input", t.name)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.f16Fn).Grid(uint32(t.dimensions[1])).Block(256).On(r.device.DefaultStream()).Launch(
			t.device, x, out, uint32(t.dimensions[0]), uint32(t.dimensions[1])); err != nil {
			return fmt.Errorf("cuda: launch matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeHCPre appends the DS4 hyper-connection input transform. Out contains
// the current stream followed by its attention-normalized form.
func (r *cudaFoundation) encodeHCPre(state cudamem.Slice[float32], hcFn, scale, base, norm cudaModelTensor, out cudamem.Slice[float32]) error {
	if state.Len() != 4*4096 || out.Len() != 2*4096 || hcFn.typeID != 1 || len(hcFn.dimensions) != 2 || hcFn.dimensions[0] != 4*4096 || hcFn.dimensions[1] != 24 ||
		scale.typeID != 0 || base.typeID != 0 || norm.typeID != 0 || len(scale.dimensions) != 1 || scale.dimensions[0] != 3 ||
		len(base.dimensions) != 1 || base.dimensions[0] != 24 || len(norm.dimensions) != 1 || norm.dimensions[0] != 4096 {
		return fmt.Errorf("cuda: HC pre: invalid tensor profile")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.hcPreFn).Grid(1).Block(256).On(r.device.DefaultStream()).Launch(
			state, hcFn.device, scale.device, base.device, norm.device, out); err != nil {
			return fmt.Errorf("cuda: launch HC pre: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) loadMathKernels() error {
	if r.matrixFn != 0 && r.rmsFn != 0 && r.f16Fn != 0 && r.hcPreFn != 0 {
		return nil
	}
	compiled, err := nvrtcx.Compile(cudaProbeSource,
		nvrtcx.WithName("ds4_math.cu"),
		nvrtcx.WithArch(r.major, r.minor),
	)
	if err != nil {
		return fmt.Errorf("cuda: compile math kernels: %w", err)
	}
	mod, err := compiled.Load(r.device)
	if err != nil {
		return fmt.Errorf("cuda: load math kernels: %w", err)
	}
	matrixFn, err := mod.GetFunction("ds4_q8_0_matvec")
	if err != nil {
		_ = mod.Unload()
		return fmt.Errorf("cuda: find Q8_0 matrix: %w", err)
	}
	rmsFn, err := mod.GetFunction("ds4_rms")
	if err != nil {
		_ = mod.Unload()
		return fmt.Errorf("cuda: find RMS: %w", err)
	}
	f16Fn, err := mod.GetFunction("ds4_f16_matvec")
	if err != nil {
		_ = mod.Unload()
		return fmt.Errorf("cuda: find F16 matrix: %w", err)
	}
	hcPreFn, err := mod.GetFunction("ds4_hc_pre")
	if err != nil {
		_ = mod.Unload()
		return fmt.Errorf("cuda: find HC pre: %w", err)
	}
	r.matrixMod = mod
	r.matrixFn = matrixFn
	r.rmsFn = rmsFn
	r.f16Fn = f16Fn
	r.hcPreFn = hcPreFn
	return nil
}

// q80Row computes the first registered Q8_0 tensor row. The caller owns x and
// out; this foundation owns only the bounded, registered model row.
func (r *cudaFoundation) q80Row(x, out cudamem.Slice[float32]) error {
	if r.probePtr == 0 || r.probeCols == 0 {
		return fmt.Errorf("cuda: Q8_0 row is not bound")
	}
	if x.Len() != int(r.probeCols) || out.Len() != 1 {
		return fmt.Errorf("cuda: Q8_0 row input/output lengths %d/%d, want %d/1", x.Len(), out.Len(), r.probeCols)
	}
	compiled, err := nvrtcx.Compile(cudaProbeSource,
		nvrtcx.WithName("ds4_q8_0_row.cu"),
		nvrtcx.WithArch(r.major, r.minor),
	)
	if err != nil {
		return fmt.Errorf("cuda: compile Q8_0 row: %w", err)
	}
	return r.device.WithContext(func() error {
		mod, err := compiled.Load(r.device)
		if err != nil {
			return fmt.Errorf("cuda: load Q8_0 row: %w", err)
		}
		defer mod.Unload()
		fn, err := mod.GetFunction("ds4_q8_0_matvec")
		if err != nil {
			return fmt.Errorf("cuda: find Q8_0 row: %w", err)
		}
		if err := cudalaunch.Kernel(fn).Grid(1).Block(256).Launch(r.probePtr, x, out, r.probeCols, uint32(1)); err != nil {
			return fmt.Errorf("cuda: launch Q8_0 row: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize Q8_0 row: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) Close() error {
	if r == nil {
		return nil
	}
	var errs []error
	if r.matrixMod != 0 && r.device != nil {
		err := r.device.WithContext(func() error { return r.matrixMod.Unload() })
		errs = append(errs, err)
		r.matrixMod = 0
		r.matrixFn = 0
		r.rmsFn = 0
		r.f16Fn = 0
		r.hcPreFn = 0
	}
	if r.mapped.host != 0 && r.device != nil {
		err := r.device.WithContext(func() error { return driver.MemHostUnregister(r.mapped.host) })
		errs = append(errs, err)
		errs = append(errs, syscall.Munmap(r.mapped.view))
		r.mapped = cudaModelRange{}
	}
	if r.device != nil {
		errs = append(errs, r.device.Close())
		r.device = nil
	}
	if r.model != nil {
		errs = append(errs, syscall.Munmap(r.model))
		r.model = nil
	}
	r.gguf = nil
	if r.file != nil {
		errs = append(errs, r.file.Close())
		r.file = nil
	}
	return errors.Join(errs...)
}
