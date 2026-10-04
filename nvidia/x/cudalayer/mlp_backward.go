package cudalayer

import (
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// MLPBackwardOptions bounds one batch of FP32 differentiation. The loss is
// sum(output*outputGradient)/Samples. Input selects input gradients. Ignore
// suppresses parameter gradients; overwrite replaces them, while accumulate
// adds them to the supplied previous gradient. At least one output is required.
// MaxBytes is a positive limit on the combined device allocation sizes.
type MLPBackwardOptions struct {
	Samples       int
	Input         bool
	ParameterMode cudair.GradientMode
	MaxBytes      uint64
}

// MLPGradients contains owned host copies after device synchronization. Each
// layer's weights and gradients use input-major, unpadded output-width order.
type MLPGradients struct {
	Output     []float32
	Input      []float32
	Parameters []float32
}

// ParameterCount returns the number of weights, with no bias or padding.
func (m *FusedMLP) ParameterCount() int {
	if m == nil {
		return 0
	}
	n := 0
	for _, l := range m.layers {
		n += l.inputWidth * l.outputWidth
	}
	return n
}

type mlpBatchLayout struct{ input, output, parameters, scratch, context, bytes int }

func (m *FusedMLP) backwardLayout(input, params, dy, previous []float32, o MLPBackwardOptions) (mlpBatchLayout, error) {
	var z mlpBatchLayout
	if m == nil || len(m.layers) == 0 || o.Samples <= 0 || o.Samples > math.MaxInt32-31 || o.MaxBytes == 0 || o.ParameterMode < cudair.GradientModeIgnore || o.ParameterMode > cudair.GradientModeAccumulate || !o.Input && o.ParameterMode == cudair.GradientModeIgnore {
		return z, fmt.Errorf("cudalayer: invalid backward batch options")
	}
	checked := func(a, b int) (int, error) {
		if a < 0 || b < 0 || b != 0 && a > math.MaxInt32/b {
			return 0, fmt.Errorf("cudalayer: backward extent overflow")
		}
		return a * b, nil
	}
	var err error
	if z.input, err = checked(o.Samples, m.inputWidth); err != nil {
		return z, err
	}
	if z.output, err = checked(o.Samples, m.outputWidth); err != nil {
		return z, err
	}
	z.parameters = m.ParameterCount()
	if len(input) != z.input || len(params) != z.parameters || len(dy) != z.output {
		return z, fmt.Errorf("cudalayer: backward buffer length mismatch")
	}
	if o.ParameterMode == cudair.GradientModeAccumulate && len(previous) != z.parameters || o.ParameterMode != cudair.GradientModeAccumulate && len(previous) != 0 {
		return z, fmt.Errorf("cudalayer: invalid previous parameter gradient length")
	}
	for _, values := range [][]float32{input, params, dy, previous} {
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return z, fmt.Errorf("cudalayer: nonfinite backward input")
			}
		}
	}
	ctx := 0
	for _, l := range m.layers {
		n, _ := l.FwdCtxBytes()
		if n > math.MaxInt32-ctx {
			return z, fmt.Errorf("cudalayer: context extent overflow")
		}
		ctx += n
	}
	if z.context, err = checked((o.Samples+31)/32*32, ctx); err != nil {
		return z, err
	}
	if o.ParameterMode != cudair.GradientModeIgnore {
		if z.scratch, err = checked(o.Samples, z.parameters); err != nil {
			return z, err
		}
	}
	// Context has a 32-byte guard at either end. All arithmetic precedes native
	// allocation, compilation and host result allocation.
	total := uint64(z.context) + 64
	counts := []int{z.input, z.parameters, z.output, z.output}
	if o.Input {
		counts = append(counts, z.input)
	}
	if z.scratch != 0 {
		counts = append(counts, z.scratch, z.parameters)
	}
	for _, n := range counts {
		total += uint64(n) * 4
	}
	if total > o.MaxBytes || total > uint64(^uint(0)>>1) {
		return z, fmt.Errorf("cudalayer: backward allocation exceeds byte budget")
	}
	z.bytes = int(total)
	return z, nil
}

func (m *FusedMLP) backwardSource(o MLPBackwardOptions) (string, int, error) {
	f, err := cudair.EmitTree(m, "mlp_forward")
	if err != nil {
		return "", 0, err
	}
	b, err := cudair.EmitBackwardTree(m, "mlp_backward", 32, cudair.GradientModeOverwrite)
	if err != nil {
		return "", 0, err
	}
	if f.FwdCtxBytes != b.FwdCtxBytes {
		return "", 0, fmt.Errorf("cudalayer: forward/backward context mismatch")
	}
	p := m.ParameterCount()
	dxarg, dparg := "nullptr", "nullptr"
	if o.Input {
		dxarg = "&dx"
	}
	if o.ParameterMode != cudair.GradientModeIgnore {
		dparg = fmt.Sprintf("sample_grad + idx * %d", p)
	}
	source := fmt.Sprintf(`
using uint8_t = unsigned char;
template<typename T,int N> struct tvec { T data[N]; __device__ T& operator[](int i) { return data[i]; } __device__ const T& operator[](int i) const { return data[i]; } };
static constexpr int WARP_SIZE=32;
#define lane_id() (threadIdx.x & 31)
%s
%s
extern "C" __global__ void mlp_batch(int n,const float* x,const float* w,const float* dy,float* y,float* dxout,float* sample_grad,uint8_t* ctx) {
 int idx=blockIdx.x*blockDim.x+threadIdx.x;
 if(idx>=n) return;
 tvec<float,%d> in;
 for(int i=0;i<%d;++i) in[i]=x[idx*%d+i];
 uint8_t* saved=ctx+(idx/32)*32*%d;
 auto out=mlp_forward(in,w,saved);
 for(int i=0;i<%d;++i) y[idx*%d+i]=out[i];
}
extern "C" __global__ void mlp_backward_batch(int n,const float* x,const float* w,const float* dy,float* y,float* dxout,float* sample_grad,uint8_t* ctx) {
 int idx=blockIdx.x*blockDim.x+threadIdx.x;
 if(idx>=n) return;
 tvec<float,%d> grad;
 tvec<float,%d> dx;
 for(int i=0;i<%d;++i) grad[i]=dy[idx*%d+i]/static_cast<float>(n);
 uint8_t* saved=ctx+(idx/32)*32*%d;
 mlp_backward(grad,w,saved,%s,%s);
`, f.Source, b.Source, m.inputWidth, m.inputWidth, m.inputWidth, f.FwdCtxBytes, m.outputWidth, m.outputWidth, m.outputWidth, m.inputWidth, m.outputWidth, m.outputWidth, f.FwdCtxBytes, dparg, dxarg)
	if o.Input {
		source += fmt.Sprintf(" for(int i=0;i<%d;++i) dxout[idx*%d+i]=dx[i];\n", m.inputWidth, m.inputWidth)
	}
	source += "}\n"
	op := "="
	if o.ParameterMode == cudair.GradientModeAccumulate {
		op = "+="
	}
	source += fmt.Sprintf(`extern "C" __global__ void mlp_reduce(int n,const float* sample_grad,float* dw) {
 int k=blockIdx.x*blockDim.x+threadIdx.x;
 if(k>=%d) return;
 float sum=0.0f;
 for(int sample=0;sample<n;++sample) sum+=sample_grad[sample*%d+k];
 dw[k] %s sum;
}
`, p, p, op)
	return source, f.FwdCtxBytes, nil
}

// BackwardBatch runs separate forward and backward CUDA kernels, then a
// separate deterministic parameter reduction in increasing sample order.
// It does not update weights or mutate any supplied host slice. Context is
// owned for this invocation and freed only after synchronous completion. An
// unresolved cleanup error implements io.Closer, reachable with errors.As;
// retry Close before closing dev. A failed synchronization retains all device
// buffers and the loaded module until a retry establishes completion.
// SiLU uses NVRTC __expf; ReLU's derivative at zero is zero. Parameter-only,
// input-only and combined requests share the same checked forward context.
func (m *FusedMLP) BackwardBatch(dev *cuda.Device, input, params, outputGradient, previous []float32, o MLPBackwardOptions) (result MLPGradients, err error) {
	l, err := m.backwardLayout(input, params, outputGradient, previous, o)
	if err != nil {
		return result, err
	}
	if dev == nil {
		return result, fmt.Errorf("cudalayer: nil backward device")
	}
	source, ctxStride, err := m.backwardSource(o)
	if err != nil {
		return result, err
	}
	compiled, err := nvrtcx.Compile(source, nvrtcx.WithDeviceArch(dev))
	if err != nil {
		return result, mlpAdoptCleanup(fmt.Errorf("compile MLP backward: %w", err))
	}
	module, _, err := compiled.LoadOwnedOutput(dev, nvrtcx.OutputPTX)
	if err != nil {
		return result, mlpAdoptCleanup(err)
	}
	owner := &mlpCleanup{synchronize: func() error { return dev.WithContext(driver.CtxSynchronize) }}
	owner.closers = append(owner.closers, module)
	defer func() {
		if e := owner.Close(); e != nil {
			err = &mlpCleanupError{cause: errors.Join(err, e), owner: owner}
			result = MLPGradients{}
		}
	}()
	kernel, err := module.Function("mlp_batch")
	if err != nil {
		return result, err
	}
	backward, err := module.Function("mlp_backward_batch")
	if err != nil {
		return result, err
	}
	reduce, err := module.Function("mlp_reduce")
	if err != nil {
		return result, err
	}
	result.Output = make([]float32, l.output)
	if o.Input {
		result.Input = make([]float32, l.input)
	}
	if l.scratch != 0 {
		result.Parameters = make([]float32, l.parameters)
	}
	err = dev.WithContext(func() (callErr error) {
		alloc := func(n int) (driver.DevicePtr, error) {
			if n == 0 {
				return 0, nil
			}
			p, e := driver.MemAllocBytes(uint(n))
			if p != 0 {
				owner.frees = append(owner.frees, mlpFreeBuffer(p, dev.WithContext, func(pointer driver.DevicePtr) error { return pointer.Free() }))
			}
			return p, e
		}
		var pointers [8]driver.DevicePtr
		counts := []int{l.input * 4, l.parameters * 4, l.output * 4, l.output * 4, len(result.Input) * 4, l.scratch * 4, len(result.Parameters) * 4, l.context + 64}
		for i, n := range counts {
			pointers[i], callErr = alloc(n)
			if callErr != nil {
				return callErr
			}
		}
		for i, v := range [][]float32{input, params, outputGradient} {
			if e := pointers[i].CopyFromHost(v); e != nil {
				return e
			}
		}
		if len(previous) != 0 {
			if e := pointers[6].CopyFromHost(previous); e != nil {
				return e
			}
		}
		guard := make([]uint8, l.context+64)
		for i := range guard {
			guard[i] = 0xa5
		}
		if e := pointers[7].CopyFromHost(guard); e != nil {
			return e
		}
		n := int32(o.Samples)
		ctx := pointers[7] + 32
		args := []unsafe.Pointer{unsafe.Pointer(&n), unsafe.Pointer(&pointers[0]), unsafe.Pointer(&pointers[1]), unsafe.Pointer(&pointers[2]), unsafe.Pointer(&pointers[3]), unsafe.Pointer(&pointers[4]), unsafe.Pointer(&pointers[5]), unsafe.Pointer(&ctx)}
		owner.inFlight = true
		if e := kernel.Launch([3]uint32{uint32((o.Samples + 31) / 32), 1, 1}, [3]uint32{32, 1, 1}, 0, 0, args); e != nil {
			return e
		}
		if e := backward.Launch([3]uint32{uint32((o.Samples + 31) / 32), 1, 1}, [3]uint32{32, 1, 1}, 0, 0, args); e != nil {
			return e
		}
		if l.scratch != 0 {
			args = []unsafe.Pointer{unsafe.Pointer(&n), unsafe.Pointer(&pointers[5]), unsafe.Pointer(&pointers[6])}
			if e := reduce.Launch([3]uint32{uint32((l.parameters + 31) / 32), 1, 1}, [3]uint32{32, 1, 1}, 0, 0, args); e != nil {
				return e
			}
		}
		if e := driver.CtxSynchronize(); e != nil {
			return e
		}
		owner.inFlight = false
		for _, read := range []struct {
			p driver.DevicePtr
			v []float32
		}{{pointers[3], result.Output}, {pointers[4], result.Input}, {pointers[6], result.Parameters}} {
			if len(read.v) != 0 {
				if e := read.p.CopyToHost(read.v); e != nil {
					return e
				}
			}
		}
		if e := pointers[7].CopyToHost(guard); e != nil {
			return e
		}
		for i, v := range guard {
			if i < 32 || i >= l.context+32 {
				if v != 0xa5 {
					return fmt.Errorf("cudalayer: backward context guard changed at %d", i)
				}
			}
		}
		// Each child stores lane-major data inside its warp region. Padded lanes
		// in the final warp must remain untouched in every child region.
		if tail := o.Samples % 32; tail != 0 {
			base := 32 + (o.Samples/32)*32*ctxStride
			offset := 0
			for _, layer := range m.layers {
				size, _ := layer.FwdCtxBytes()
				for lane := tail; lane < 32; lane++ {
					for j := 0; j < size; j++ {
						if guard[base+offset*32+lane*size+j] != 0xa5 {
							return fmt.Errorf("cudalayer: padded context lane changed")
						}
					}
				}
				offset += size
			}
		}
		return nil
	})
	if err != nil {
		return MLPGradients{}, fmt.Errorf("run MLP backward: %w", err)
	}
	for _, values := range [][]float32{result.Output, result.Input, result.Parameters} {
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return MLPGradients{}, fmt.Errorf("cudalayer: nonfinite MLP backward result")
			}
		}
	}
	return result, nil
}
