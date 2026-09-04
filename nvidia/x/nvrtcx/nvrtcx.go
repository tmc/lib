package nvrtcx

import (
	"fmt"
	"sort"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// CompiledModule represents the compiled output of an NVRTC JIT compilation.
type CompiledModule struct {
	ptx   string
	cubin []byte
	log   string
	key   string
}

// PTX returns the compiled PTX assembly text.
func (cm *CompiledModule) PTX() string {
	if cm == nil {
		return ""
	}
	return cm.ptx
}

// CUBIN returns the compiled CUBIN binary if compiled for a physical architecture.
func (cm *CompiledModule) CUBIN() []byte {
	if cm == nil {
		return nil
	}
	return append([]byte(nil), cm.cubin...)
}

// Log returns the compiler diagnostic log.
func (cm *CompiledModule) Log() string {
	if cm == nil {
		return ""
	}
	return cm.log
}

// Load loads the compiled module into the CUDA driver on dev.
func (cm *CompiledModule) Load(dev *cuda.Device) (driver.Module, error) {
	if cm == nil {
		return 0, fmt.Errorf("nvrtcx: nil compiled module")
	}
	if dev == nil {
		return 0, fmt.Errorf("nvrtcx: nil device passed to Load")
	}

	var mod driver.Module
	err := dev.WithContext(func() error {
		if len(cm.cubin) > 0 {
			m, err := driver.NewModuleLoadData(unsafe.Pointer(&cm.cubin[0]))
			if err == nil {
				mod = m
				return nil
			}
		}
		// Fallback to loading PTX string
		if cm.ptx != "" {
			cstr := append([]byte(cm.ptx), 0)
			m, err := driver.NewModuleLoadData(unsafe.Pointer(&cstr[0]))
			if err != nil {
				return err
			}
			mod = m
			return nil
		}
		return fmt.Errorf("nvrtcx: module contains no ptx or cubin image")
	})
	if err != nil {
		return 0, fmt.Errorf("nvrtcx: load module: %w", err)
	}
	return mod, nil
}

type options struct {
	name     string
	flags    []string
	headers  map[string]string
	useCache bool
	err      error
}

// Option configures JIT compilation.
type Option func(*options)

// WithName specifies the virtual filename of the CUDA program.
func WithName(name string) Option {
	return func(o *options) {
		if name == "" {
			o.err = fmt.Errorf("nvrtcx: empty program name")
			return
		}
		o.name = name
	}
}

// WithArch configures the target virtual compute architecture (for example,
// compute_80 for major=8, minor=0).
func WithArch(major, minor int) Option {
	return func(o *options) {
		if major <= 0 || minor < 0 {
			o.err = fmt.Errorf("nvrtcx: invalid compute architecture %d.%d", major, minor)
			return
		}
		o.flags = append(o.flags, fmt.Sprintf("--gpu-architecture=compute_%d%d", major, minor))
	}
}

// WithFlag adds a compiler flag to the NVRTC compilation.
func WithFlag(flag string) Option {
	return func(o *options) {
		if flag == "" {
			o.err = fmt.Errorf("nvrtcx: empty compiler flag")
			return
		}
		o.flags = append(o.flags, flag)
	}
}

// WithHeader adds an in-memory header that can be included by the source code.
func WithHeader(name, content string) Option {
	return func(o *options) {
		if name == "" {
			o.err = fmt.Errorf("nvrtcx: empty header name")
			return
		}
		if o.headers == nil {
			o.headers = make(map[string]string)
		}
		o.headers[name] = content
	}
}

// WithNoCache disables the compilation cache for this compilation pass.
func WithNoCache() Option {
	return func(o *options) {
		o.useCache = false
	}
}

// Compile compiles the given CUDA C++ source code into a CompiledModule.
func Compile(src string, opts ...Option) (*CompiledModule, error) {
	if src == "" {
		return nil, fmt.Errorf("nvrtcx: empty source")
	}
	cfg := options{
		name:     "jit_program.cu",
		useCache: true,
	}
	for _, opt := range opts {
		if opt == nil {
			return nil, fmt.Errorf("nvrtcx: nil option")
		}
		opt(&cfg)
	}
	if cfg.err != nil {
		return nil, cfg.err
	}

	key := computeCacheKey(src, cfg.name, cfg.flags, cfg.headers)
	if cfg.useCache {
		if cached, ok := lookupCache(key); ok {
			return cached, nil
		}
	}

	return singleFlightCompile(key, func() (*CompiledModule, error) {
		// Recheck cache inside singleflight
		if cfg.useCache {
			if cached, ok := lookupCache(key); ok {
				return cached, nil
			}
		}

		headerNames := make([]string, 0, len(cfg.headers))
		for k := range cfg.headers {
			headerNames = append(headerNames, k)
		}
		sort.Strings(headerNames)

		headerContents := make([]string, len(headerNames))
		for i, k := range headerNames {
			headerContents[i] = cfg.headers[k]
		}

		prog, err := nvrtc.CreateProgram(src, cfg.name, headerContents, headerNames)
		if err != nil {
			return nil, fmt.Errorf("nvrtcx: create program: %w", err)
		}
		defer func() {
			_ = nvrtc.DestroyProgram(&prog)
		}()

		compileErr := prog.Compile(cfg.flags)
		log, _ := prog.GetProgramLog()

		if compileErr != nil {
			return nil, fmt.Errorf("nvrtcx: compile error: %w\ncompiler log:\n%s", compileErr, log)
		}

		ptxBytes, err := prog.GetPTX()
		if err != nil {
			return nil, fmt.Errorf("nvrtcx: get PTX: %w", err)
		}
		cubin, _ := prog.GetCUBIN()

		res := &CompiledModule{
			ptx:   string(ptxBytes),
			cubin: cubin,
			log:   log,
			key:   key,
		}

		if cfg.useCache {
			if err := storeCache(key, res); err != nil {
				return nil, err
			}
		}

		return res, nil
	})
}
