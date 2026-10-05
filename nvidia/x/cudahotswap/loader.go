package cudahotswap

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Source describes one kernel source and its entry point. Flags are NVRTC
// flags. The caller must not change Flags during compilation.
type Source struct {
	Code     string
	Filename string
	Entry    string
	Flags    []string
}

// Compilation records the compiler's PTX, cache key and header dependencies.
// Treat a Compilation as immutable after Compile returns. It owns no loaded
// CUDA module. Native loader identity is retained privately.
type Compilation struct {
	PTX          string
	Key          string
	Entry        string
	Dependencies []cudakernel.Dependency
	native       *nvrtcx.CompiledModule
	entry        string
	loader       *NVRTCLoader
}

// Loader compiles without a GPU lease and loads under the service's per-request
// lease. A nonnil Kernel returned with an error is still owned by the caller.
// Cleanup errors may additionally implement io.Closer and require retry before
// the device closes. Implementations must retain those owners on failure.
type Loader interface {
	Compile(context.Context, Source) (*Compilation, error)
	Load(context.Context, *Compilation) (Kernel, error)
}

// Kernel is one loaded function and its module owner. Launch submits work;
// Wait confirms completion of the supplied stream, or the whole owning context
// for a nil stream. A canceled or failed Wait does not establish completion.
// Close must retain unresolved resources on error and permit retry.
type Kernel interface {
	Raw() driver.Function
	Launch(cudalaunch.Config, ...any) error
	Wait(context.Context, *cuda.Stream) error
	Close() error
}

// NVRTCLoader compiles with nvrtcx's toolchain-aware cache and loads owned CUDA
// modules. The device must remain open until the service and all pending
// cleanup owners have closed. Its zero value is not usable.
type NVRTCLoader struct {
	dev      *cuda.Device
	includes []string
}

// NewNVRTCLoader uses disk include directories, so header edits can participate
// in compilation and Dependencies. It copies includes and does not take a GPU
// lease or create a CUDA device.
func NewNVRTCLoader(dev *cuda.Device, includes ...string) (*NVRTCLoader, error) {
	if dev == nil {
		return nil, fmt.Errorf("create loader: nil device")
	}
	return &NVRTCLoader{dev: dev, includes: append([]string(nil), includes...)}, nil
}

// Compile produces native CUBIN while retaining PTX for the service ABI check.
// Compiler context cancellation is independent of a GPU request lease.
func (l *NVRTCLoader) Compile(ctx context.Context, source Source) (*Compilation, error) {
	if l == nil || l.dev == nil {
		return nil, fmt.Errorf("compile kernel: nil loader device")
	}
	if ctx == nil {
		return nil, fmt.Errorf("compile kernel: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source.Code == "" || source.Filename == "" || source.Entry == "" {
		return nil, fmt.Errorf("compile kernel: source, filename and entry are required")
	}
	opts := []nvrtcx.Option{nvrtcx.WithName(source.Filename), nvrtcx.WithDeviceArch(l.dev)}
	if len(l.includes) != 0 {
		opts = append(opts, nvrtcx.WithIncludePaths(l.includes...))
	}
	for _, flag := range source.Flags {
		opts = append(opts, nvrtcx.WithFlag(flag))
	}
	cm, err := nvrtcx.CompileContext(ctx, source.Code, opts...)
	if err != nil {
		return nil, fmt.Errorf("compile kernel: %w", err)
	}
	if _, err := cudalaunch.SignatureFromPTX(cm.PTX(), source.Entry); err != nil {
		return nil, fmt.Errorf("compile kernel entry: %w", err)
	}
	return &Compilation{PTX: cm.PTX(), Key: cm.Key(), Entry: source.Entry, Dependencies: cm.Dependencies(), native: cm, entry: source.Entry, loader: l}, nil
}

// Load loads this loader's native compilation and binds its declared entry.
// A partial module owner remains reachable through the result or cleanup error.
func (l *NVRTCLoader) Load(ctx context.Context, c *Compilation) (Kernel, error) {
	if l == nil || l.dev == nil {
		return nil, fmt.Errorf("load kernel: nil loader device")
	}
	if ctx == nil {
		return nil, fmt.Errorf("load kernel: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if c == nil || c.loader != l || c.native == nil || c.PTX != c.native.PTX() || c.Key != c.native.Key() || c.Entry != c.entry {
		return nil, fmt.Errorf("load kernel: invalid or changed native compilation")
	}
	sig, err := cudalaunch.SignatureFromPTX(c.PTX, c.Entry)
	if err != nil {
		return nil, fmt.Errorf("load kernel signature: %w", err)
	}
	module, _, err := c.native.LoadOwnedOutput(l.dev, nvrtcx.OutputCUBIN)
	if module == nil {
		return nil, err
	}
	result := &nativeKernel{module: module, dev: l.dev, signature: sig}
	if err != nil {
		return result, err
	}
	result.function, err = module.Function(c.Entry)
	if err != nil {
		return result, fmt.Errorf("bind kernel: %w", err)
	}
	return result, nil
}

type nativeKernel struct {
	module    *cuda.Module
	function  *cuda.Kernel
	dev       *cuda.Device
	signature cudalaunch.Signature
}

func (k *nativeKernel) Raw() driver.Function {
	if k == nil || k.function == nil {
		return 0
	}
	return k.function.Raw()
}

func (k *nativeKernel) Launch(cfg cudalaunch.Config, args ...any) error {
	if k == nil || k.function == nil {
		return fmt.Errorf("launch kernel: missing function")
	}
	return cudalaunch.OwnedKernel(k.function).Config(cfg).WithSignature(k.signature).Launch(args...)
}

func (k *nativeKernel) Wait(ctx context.Context, stream *cuda.Stream) error {
	if k == nil || k.module == nil || k.dev == nil {
		return fmt.Errorf("wait for kernel: missing module")
	}
	if ctx == nil {
		return fmt.Errorf("wait for kernel: nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if stream != nil {
		if stream.Device() != k.dev {
			return fmt.Errorf("wait for kernel: stream belongs to another device owner")
		}
		return stream.Synchronize(ctx)
	}
	// Default-stream completion requires a context synchronization. Once the
	// driver call begins it is synchronous; a native process watchdog bounds it.
	return k.dev.WithContext(driver.CtxSynchronize)
}

func (k *nativeKernel) Close() error {
	if k == nil || k.module == nil {
		return nil
	}
	return k.module.Close()
}
