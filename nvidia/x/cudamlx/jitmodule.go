package cudamlx

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
)

// KernelSource is what a kernel builder produces: either CUDA C++ source
// to compile with nvrtc, or precompiled PTX/cubin. Port of
// KernelBuilderResult.
type KernelSource struct {
	Precompiled bool
	Source      string
	KernelNames []string
}

// KernelBuilder produces the source of a JIT module on cache miss.
type KernelBuilder func() (KernelSource, error)

// JitModule compiles, caches, and loads one CUDA module. Port of
// JitModule in jit_module.cpp, minus the embedded MLX kernel headers;
// include directories come from IncludePaths.
type JitModule struct {
	dev    *Device
	module driver.Module

	mu      sync.Mutex
	kernels map[string]*kernelEntry
}

type kernelEntry struct {
	fn         driver.Function
	mangled    string
	configured bool
	blockDim   uint32
}

// IncludePaths returns the nvrtc include directories: $CUDA_HOME or
// $CUDA_PATH include when set, else /usr/local/cuda/include. Port of
// include_path_args without the bundled-header lookups.
func IncludePaths() []string {
	home := os.Getenv("CUDA_HOME")
	if home == "" {
		home = os.Getenv("CUDA_PATH")
	}
	if home == "" {
		home = "/usr/local/cuda"
	}
	inc := filepath.Join(home, "include")
	if _, err := os.Stat(inc); err != nil {
		return nil
	}
	return []string{inc}
}

// ptxCacheDir returns the on-disk PTX cache directory, or "" when
// unavailable. Override with CUDAMLX_PTX_CACHE_DIR.
func ptxCacheDir() string {
	dir := os.Getenv("CUDAMLX_PTX_CACHE_DIR")
	if dir == "" {
		dir = filepath.Join(os.TempDir(), "cudamlx", "ptx")
	}
	if err := os.MkdirAll(dir, 0o777); err != nil {
		return ""
	}
	return dir
}

func readCachedPTX(dir, moduleName string) (ptx []byte, kernels [][2]string, ok bool) {
	if dir == "" {
		return nil, nil, false
	}
	ptx, err := os.ReadFile(filepath.Join(dir, moduleName+".ptx"))
	if err != nil {
		return nil, nil, false
	}
	txt, err := os.ReadFile(filepath.Join(dir, moduleName+".txt"))
	if err != nil {
		return nil, nil, false
	}
	for _, line := range strings.Split(string(txt), "\n") {
		name, mangled, found := strings.Cut(line, "\t")
		if found {
			kernels = append(kernels, [2]string{name, mangled})
		}
	}
	return ptx, kernels, true
}

func writeCachedPTX(dir, moduleName string, ptx []byte, kernels [][2]string, source string) {
	if dir == "" {
		return
	}
	os.WriteFile(filepath.Join(dir, moduleName+".ptx"), ptx, 0o666)
	var sb strings.Builder
	for _, k := range kernels {
		fmt.Fprintf(&sb, "%s\t%s\n", k[0], k[1])
	}
	os.WriteFile(filepath.Join(dir, moduleName+".txt"), []byte(sb.String()), 0o666)
	os.WriteFile(filepath.Join(dir, moduleName+".cu"), []byte(source), 0o666)
}

// compilerSupportsDeviceSASS reports whether nvrtc can compile directly
// to the device's SASS. Port of compiler_supports_device_sass.
func compilerSupportsDeviceSASS(dev *Device) bool {
	major, minor, err := nvrtc.Version()
	if err != nil {
		return false
	}
	le := func(maj, min int) bool {
		if dev.ComputeCapabilityMajor() != maj {
			return dev.ComputeCapabilityMajor() < maj
		}
		return dev.ComputeCapabilityMinor() <= min
	}
	switch {
	case major < 9:
		return false
	case major == 9:
		return le(7, 2)
	case major == 10:
		return le(7, 5)
	case major == 11 && minor == 0:
		return le(8, 0)
	case major == 11 && minor < 8:
		return le(8, 6)
	default:
		return true
	}
}

func compile(dev *Device, moduleName, source string, kernelNames, extraArgs []string) (ptx []byte, kernels [][2]string, err error) {
	prog, err := nvrtc.CreateProgram(source, moduleName+".cu", nil, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("create program: %w", err)
	}
	defer nvrtc.DestroyProgram(&prog)

	names := make([][]byte, len(kernelNames))
	for i, name := range kernelNames {
		names[i] = append([]byte(name), 0)
		if err := prog.AddNameExpression(&names[i][0]); err != nil {
			return nil, nil, fmt.Errorf("add name expression %q: %w", name, err)
		}
	}

	useSASS := compilerSupportsDeviceSASS(dev)
	targetKind := "compute"
	if useSASS {
		targetKind = "sm"
	}
	archTag := ""
	if dev.ComputeCapabilityMajor() >= 9 {
		archTag = "a"
	}
	args := []string{
		"--device-as-default-execution-space",
		fmt.Sprintf("--gpu-architecture=%s_%d%d%s",
			targetKind, dev.ComputeCapabilityMajor(), dev.ComputeCapabilityMinor(), archTag),
	}
	for _, inc := range IncludePaths() {
		args = append(args, "--include-path="+inc)
	}
	args = append(args, extraArgs...)

	if err := prog.Compile(args); err != nil {
		log, logErr := prog.GetProgramLog()
		if logErr != nil {
			log = logErr.Error()
		}
		return nil, nil, fmt.Errorf("compile kernel: %w: %s", err, log)
	}

	for i, name := range kernelNames {
		var mangled *byte
		if err := prog.GetLoweredName(&names[i][0], &mangled); err != nil {
			return nil, nil, fmt.Errorf("lowered name %q: %w", name, err)
		}
		kernels = append(kernels, [2]string{name, goString(mangled)})
	}

	if useSASS {
		ptx, err = prog.GetCUBIN()
	} else {
		ptx, err = prog.GetPTX()
	}
	if err != nil {
		return nil, nil, fmt.Errorf("get compiled code: %w", err)
	}
	return ptx, kernels, nil
}

// NewJitModule builds (or loads from the disk cache) and loads a module.
// extraArgs are passed to nvrtc in addition to the architecture and
// include flags.
func NewJitModule(dev *Device, moduleName string, builder KernelBuilder, useDiskCache bool, extraArgs ...string) (*JitModule, error) {
	cacheDir := ""
	if useDiskCache {
		cacheDir = ptxCacheDir()
	}
	ptx, kernels, ok := readCachedPTX(cacheDir, moduleName)
	if !ok {
		src, err := builder()
		if err != nil {
			return nil, err
		}
		if src.Precompiled {
			ptx = []byte(src.Source)
			for _, name := range src.KernelNames {
				kernels = append(kernels, [2]string{name, name})
			}
		} else {
			ptx, kernels, err = compile(dev, moduleName, src.Source, src.KernelNames, extraArgs)
			if err != nil {
				return nil, err
			}
		}
		if useDiskCache {
			writeCachedPTX(cacheDir, moduleName, ptx, kernels, src.Source)
		}
	}

	m := &JitModule{dev: dev, kernels: make(map[string]*kernelEntry)}
	err := dev.WithCurrent(func() error {
		// cuModuleLoadData requires PTX text to be NUL-terminated.
		image := ptx
		if len(image) == 0 || image[len(image)-1] != 0 {
			image = append(image, 0)
		}
		mod, err := driver.NewModuleLoadData(unsafe.Pointer(&image[0]))
		if err != nil {
			return fmt.Errorf("load module %s: %w", moduleName, err)
		}
		m.module = mod
		for _, k := range kernels {
			fn, err := mod.GetFunction(k[1])
			if err != nil {
				return fmt.Errorf("kernel %s (%s): %w", k[0], k[1], err)
			}
			m.kernels[k[0]] = &kernelEntry{fn: fn, mangled: k[1]}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return m, nil
}

// GetKernel returns the kernel and its max-occupancy block size. On the
// first lookup configure, when non-nil, runs once against the raw
// function (for setting attributes such as shared memory size).
func (m *JitModule) GetKernel(kernelName string, configure func(driver.Function) error) (driver.Function, uint32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.kernels[kernelName]
	if !ok {
		return 0, 0, fmt.Errorf("no kernel named %s", kernelName)
	}
	if !e.configured {
		var blockDim int32
		err := m.dev.WithCurrent(func() error {
			if configure != nil {
				if err := configure(e.fn); err != nil {
					return err
				}
			}
			_, block, err := e.fn.MaxPotentialBlockSize(0, 0, 0)
			if err != nil {
				return err
			}
			blockDim = block
			return nil
		})
		if err != nil {
			return 0, 0, err
		}
		e.configured = true
		e.blockDim = uint32(blockDim)
	}
	return e.fn, e.blockDim, nil
}

var (
	moduleCacheMu sync.RWMutex
	moduleCache   = make(map[string]*JitModule)
)

// GetJitModule returns the cached module for name, building it with
// builder on first use. Port of get_jit_module.
func GetJitModule(dev *Device, name string, builder KernelBuilder) (*JitModule, error) {
	moduleCacheMu.RLock()
	m, ok := moduleCache[name]
	moduleCacheMu.RUnlock()
	if ok {
		return m, nil
	}
	moduleCacheMu.Lock()
	defer moduleCacheMu.Unlock()
	if m, ok := moduleCache[name]; ok {
		return m, nil
	}
	m, err := NewJitModule(dev, name, builder, true)
	if err != nil {
		return nil, err
	}
	moduleCache[name] = m
	return m, nil
}

func goString(p *byte) string {
	if p == nil {
		return ""
	}
	var b []byte
	for ptr := p; *ptr != 0; ptr = (*byte)(unsafe.Add(unsafe.Pointer(ptr), 1)) {
		b = append(b, *ptr)
	}
	return string(b)
}
