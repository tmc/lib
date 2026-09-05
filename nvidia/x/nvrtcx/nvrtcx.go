package nvrtcx

import (
	"errors"
	"fmt"
	"sort"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// CompiledModule represents the compiled output of an NVRTC JIT compilation.
type CompiledModule struct {
	output OutputKind
	ltoir  []byte
	tileir []byte
	ptx    string
	cubin  []byte
	log    string
	key    string
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
	if len(cm.ltoir) != 0 {
		return 0, fmt.Errorf("nvrtcx: LTOIR requires linking before loading")
	}
	if dev == nil {
		return 0, fmt.Errorf("nvrtcx: nil device passed to Load")
	}

	var mod driver.Module
	err := dev.WithContext(func() error {
		if len(cm.tileir) > 0 {
			m, err := driver.NewModuleLoadData(unsafe.Pointer(&cm.tileir[0]))
			mod = m
			return err
		}
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
	output   OutputKind
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
// If program destruction fails, the returned error retains the native program
// and implements io.Closer (reachable with errors.As). Call Close on that error
// to retry cleanup. Such a result is not added to the compilation cache.
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
	flags, err := outputFlags(cfg.output, cfg.flags)
	if err != nil {
		return nil, err
	}
	cfg.flags = flags

	// Resolve the loaded compiler before looking up any compiled artifact.
	// A cache hit must not bypass a missing or incompatible compiler installation.
	major, minor, err := nvrtc.Version()
	if err != nil {
		return nil, fmt.Errorf("nvrtcx: compiler version: %w", err)
	}
	identity := fmt.Sprintf("nvrtc %d.%d %s output=%d", major, minor, nvrtc.Path(), cfg.output)
	key := compilerCacheKey(computeCacheKey(src, cfg.name, cfg.flags, cfg.headers), identity)
	if cfg.useCache {
		if cached, ok := lookupCache(key); ok {
			return cached, nil
		}
	}

	return singleFlightCompile(key, func() (result *CompiledModule, resultErr error) {
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
			result, resultErr = finishProgram(result, resultErr, func() error { return nvrtc.DestroyProgram(&prog) })
			if resultErr == nil && cfg.useCache {
				if err := storeCache(key, result); err != nil {
					result, resultErr = nil, err
				}
			}
		}()

		compileErr := prog.Compile(cfg.flags)
		log, logErr := prog.GetProgramLog()

		if err := compilerDiagnostic(compileErr, logErr, log); err != nil {
			return nil, err
		}

		res, err := getArtifact(prog, cfg.output)
		if err != nil {
			return nil, fmt.Errorf("%w\ncompiler log:\n%s", err, log)
		}
		res.log = log
		res.key = key

		return res, nil
	})
}

func compilerDiagnostic(compileErr, logErr error, log string) error {
	var errs []error
	if compileErr != nil {
		errs = append(errs, fmt.Errorf("nvrtcx: compile: %w\ncompiler log:\n%s", compileErr, log))
	}
	if logErr != nil {
		errs = append(errs, fmt.Errorf("nvrtcx: retrieve compiler log: %w", logErr))
	}
	return errors.Join(errs...)
}
