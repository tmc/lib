// Features reports the NVIDIA binding features available on the current host.
//
// It creates only short-lived handles and descriptors. The companion
// simple-addition example performs a complete driver, NVRTC, allocation,
// kernel-launch, and copy-back workflow.
package main

import (
	"fmt"
	"log"
	"net"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/cudnn"
	"github.com/tmc/lib/nvidia/cufft"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/curand"
	"github.com/tmc/lib/nvidia/cusolver"
	"github.com/tmc/lib/nvidia/cusparse"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/fabricmanager"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/moonep"
	"github.com/tmc/lib/nvidia/x/nvlshm"
	"github.com/tmc/lib/nvidia/x/nvlshm/tcpboot"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev, err := cudaDevice()
	if err != nil {
		return err
	}
	defer dev.PrimaryCtxRelease()

	report("driver", func() error {
		version, err := driver.DriverGetVersion()
		if err != nil {
			return err
		}
		name, err := dev.GetName()
		if err != nil {
			return err
		}
		fmt.Printf("CUDA %d on %s\n", version, name)
		return nil
	})
	report("nvrtc", nvrtcFeature)
	report("nvml", nvmlFeature)
	report("cublas", cublasFeature)
	report("cublaslt", cublasltFeature)
	report("cudnn", cudnnFeature)
	report("cufft", cufftFeature)
	report("cupti", cuptiFeature)
	report("curand", curandFeature)
	report("cusolver", cusolverFeature)
	report("cusparse", cusparseFeature)
	report("fabricmanager", fabricmanagerFeature)
	report("nccl", ncclFeature)
	report("nvlshm", func() error { return nvlshmFeature(dev) })
	report("moonep", moonepFeature)
	report("tcpboot", tcpbootFeature)
	return nil
}

func report(name string, f func() error) {
	if err := f(); err != nil {
		fmt.Printf("%s: UNAVAILABLE: %v\n", name, err)
		return
	}
	fmt.Printf("%s: OK\n", name)
}

func cudaDevice() (driver.Device, error) {
	if err := driver.Init(0); err != nil {
		return 0, fmt.Errorf("cuInit: %w", err)
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		return 0, fmt.Errorf("cuDeviceGet: %w", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		return 0, fmt.Errorf("cuDevicePrimaryCtxRetain: %w", err)
	}
	if err := ctx.SetCurrent(); err != nil {
		return 0, fmt.Errorf("cuCtxSetCurrent: %w", err)
	}
	return dev, nil
}

func nvrtcFeature() error {
	major, minor, err := nvrtc.Version()
	if err != nil {
		return err
	}
	p, err := nvrtc.CreateProgram(`extern "C" __global__ void noop() {}`, "noop.cu", nil, nil)
	if err != nil {
		return err
	}
	defer nvrtc.DestroyProgram(&p)
	if err := p.Compile(nil); err != nil {
		return err
	}
	ptx, err := p.GetPTX()
	if err != nil {
		return err
	}
	fmt.Printf("NVRTC %d.%d compiled %d bytes of PTX\n", major, minor, len(ptx))
	return nil
}

func nvmlFeature() error {
	if err := nvml.Init(); err != nil {
		return err
	}
	defer nvml.Shutdown()
	count, err := nvml.DeviceGetCount()
	if err != nil {
		return err
	}
	dev, err := nvml.NewDeviceByIndex(0)
	if err != nil {
		return err
	}
	name, err := dev.GetName()
	if err != nil {
		return err
	}
	fmt.Printf("%d device(s); %s\n", count, name)
	memory, err := dev.GetMemoryInfo()
	if err != nil {
		fmt.Printf("NVML memory query: UNAVAILABLE: %v\n", err)
		return nil
	}
	fmt.Printf("NVML memory: %d/%d bytes used\n", memory.Used, memory.Total)
	return nil
}

func cublasFeature() error {
	h, err := cublas.Create()
	if err != nil {
		return err
	}
	defer h.Destroy()
	version, err := h.GetVersion()
	if err != nil {
		return err
	}
	if err := h.SetPointerMode(cublas.PointerModeHost); err != nil {
		return err
	}
	a := []float32{1, 3, 2, 4}
	b := []float32{5, 7, 6, 8}
	c := make([]float32, 4)
	ptrs, err := deviceFloat32s(a, b, c)
	if err != nil {
		return err
	}
	defer freeDevicePtrs(ptrs)
	alpha, beta := float32(1), float32(0)
	if err := h.SgemmEx(cublas.OpN, cublas.OpN, 2, 2, 2, &alpha,
		uintptr(ptrs[0]), cublas.R32F, 2,
		uintptr(ptrs[1]), cublas.R32F, 2, &beta,
		uintptr(ptrs[2]), cublas.R32F, 2); err != nil {
		return err
	}
	runtime.KeepAlive(ptrs)
	if err := ptrs[2].MemcpyDtoH(unsafe.Pointer(&c[0]), uint(len(c))*4); err != nil {
		return err
	}
	want := []float32{19, 43, 22, 50}
	for i := range want {
		if c[i] != want[i] {
			return fmt.Errorf("SGEMM result[%d] = %v, want %v", i, c[i], want[i])
		}
	}
	fmt.Printf("cuBLAS %d SGEMM = %v\n", version, c)
	return nil
}

func cublasltFeature() error {
	h, err := cublaslt.Create()
	if err != nil {
		return err
	}
	defer h.Destroy()
	layout, err := cublaslt.NewMatrixLayout(cublaslt.R32F, 2, 2, 2)
	if err != nil {
		return err
	}
	defer layout.Destroy()
	fmt.Println("created a 2x2 FP32 matrix layout")
	return nil
}

func cudnnFeature() error {
	h, err := cudnn.Create()
	if err != nil {
		return err
	}
	defer h.Destroy()
	desc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return err
	}
	defer desc.Destroy()
	fmt.Println("created a tensor descriptor")
	return nil
}

//go:nocheckptr
func cufftFeature() error {
	version, err := cufft.GetVersion()
	if err != nil {
		return err
	}
	workspace, err := cufft.Estimate1d(8, cufft.C2C, 1)
	if err != nil {
		return err
	}
	fmt.Printf("cuFFT %d estimated an 8-point C2C workspace at %d bytes\n", version, workspace)
	return nil
}

func cuptiFeature() error {
	version, err := cupti.GetVersion()
	if err != nil {
		return err
	}
	fmt.Printf("CUPTI API %d\n", version)
	return nil
}

func curandFeature() error {
	g, err := curand.CreateGeneratorHost(curand.CURAND_RNG_PseudoDefault)
	if err != nil {
		return err
	}
	defer g.Destroy()
	if err := g.SetPseudoRandomGeneratorSeed(1); err != nil {
		return err
	}
	values := make([]float32, 4)
	if err := g.GenerateUniform(&values[0], uint(len(values))); err != nil {
		return err
	}
	fmt.Printf("host random values %v\n", values)
	return nil
}

func cusolverFeature() error {
	h, err := cusolver.Create()
	if err != nil {
		return err
	}
	defer h.Destroy()
	version, err := cusolver.GetVersion()
	if err != nil {
		return err
	}
	fmt.Printf("cuSOLVER %d\n", version)
	return nil
}

func cusparseFeature() error {
	h, err := cusparse.Create()
	if err != nil {
		return err
	}
	defer h.Destroy()
	version, err := h.GetVersion()
	if err != nil {
		return err
	}
	fmt.Printf("cuSPARSE %d\n", version)
	return nil
}

func fabricmanagerFeature() error {
	if err := fabricmanager.LibInit(); err != nil {
		return err
	}
	return fabricmanager.LibShutdown()
}

func ncclFeature() error {
	version, err := nccl.GetVersion()
	if err != nil {
		return err
	}
	if _, err := nccl.GetUniqueId(); err != nil {
		return err
	}
	device := int32(0)
	comm, err := nccl.NewCommInitAll(1, &device)
	if err != nil {
		return err
	}
	defer comm.Destroy()
	values := []float32{3, 5, 7, 11}
	p, err := driver.MemAlloc(uint(len(values)) * 4)
	if err != nil {
		return err
	}
	defer p.MemFree()
	if err := p.MemcpyHtoD(unsafe.Pointer(&values[0]), uint(len(values))*4); err != nil {
		return err
	}
	if err := comm.AllReduce(nccl.Buffer(p), nccl.Buffer(p), uint(len(values)), nccl.Float32, nccl.Sum, 0); err != nil {
		return err
	}
	if err := p.MemcpyDtoH(unsafe.Pointer(&values[0]), uint(len(values))*4); err != nil {
		return err
	}
	fmt.Printf("NCCL %d one-rank all-reduce = %v\n", version, values)
	return nil
}

func deviceFloat32s(values ...[]float32) ([]driver.DevicePtr, error) {
	ptrs := make([]driver.DevicePtr, len(values))
	for i, value := range values {
		p, err := driver.MemAlloc(uint(len(value)) * 4)
		if err != nil {
			freeDevicePtrs(ptrs[:i])
			return nil, err
		}
		if err := p.MemcpyHtoD(unsafe.Pointer(&value[0]), uint(len(value))*4); err != nil {
			p.MemFree()
			freeDevicePtrs(ptrs[:i])
			return nil, err
		}
		ptrs[i] = p
	}
	return ptrs, nil
}

func freeDevicePtrs(ptrs []driver.DevicePtr) {
	for _, p := range ptrs {
		if p != 0 {
			p.MemFree()
		}
	}
}

func nvlshmFeature(dev driver.Device) error {
	granularity, err := nvlshm.Granularity(dev)
	if err != nil {
		return err
	}
	multicast, err := nvlshm.MulticastSupported(dev)
	if err != nil {
		return err
	}
	fmt.Printf("VMM granularity %d bytes; multicast supported=%t\n", granularity, multicast)
	return nil
}

func moonepFeature() error {
	l, err := moonep.NewLayout(moonep.Config{S: 8, H: 16, K: 2, E: 4, R: 1}, 64*1024, 64*1024)
	if err != nil {
		return err
	}
	fmt.Printf("layout: hidden chunk=%d bytes, meta chunk=%d bytes\n", l.HiddenChunkBytes(), l.MetaChunkBytes())
	return nil
}

func tcpbootFeature() error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil {
		return err
	}

	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for rank := range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			b, err := tcpboot.New(rank, 2, addr)
			if err != nil {
				errs <- err
				return
			}
			defer b.Close()
			value, err := b.BroadcastString("ready", 0)
			if err == nil && value != "ready" {
				err = fmt.Errorf("broadcast = %q, want ready", value)
			}
			if err == nil {
				err = b.Barrier()
			}
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			return err
		}
	}
	fmt.Println("two-rank broadcast and barrier")
	return nil
}
