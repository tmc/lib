package nvrtcx

import (
	"errors"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// CompiledModule represents the compiled output of an NVRTC JIT compilation.
type CompiledModule struct {
	lowered map[string]string
	output  OutputKind
	ltoir   []byte
	tileir  []byte
	ptx     string
	cubin   []byte
	log     string
	key     string
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
// Mixed PTX/TileIR results require LoadOutput to select an artifact.
// If restoring the context fails after loading, Load attempts to unload the
// module. A failed unload is retained by the returned error, which implements
// io.Closer (reachable with errors.As); Close retries cleanup before dev closes.
func (cm *CompiledModule) Load(dev *cuda.Device) (driver.Module, error) {
	if cm == nil {
		return 0, fmt.Errorf("nvrtcx: nil compiled module")
	}
	if cm.output == OutputPTXTileIR {
		return 0, fmt.Errorf("nvrtcx: mixed output requires LoadOutput")
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
			if cm.ptx == "" {
				return fmt.Errorf("load cubin: %w", err)
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
	runtime.KeepAlive(cm)
	pending := mod
	return finishModuleLoad(mod, err, func() error {
		if pending == 0 {
			return nil
		}
		return dev.WithContext(func() error {
			if err := pending.Unload(); err != nil {
				return err
			}
			pending = 0
			return nil
		})
	})
}

type options struct {
	expressions  []string
	output       OutputKind
	outputSet    bool
	physicalArch bool
	name         string
	flags        []string
	headers      map[string]string
	useCache     bool
	err          error
}

// Option configures JIT compilation.
type Option func(*options)

// WithNameExpressions registers CUDA name expressions before compilation.
// Expressions must be nonempty and contain no NUL bytes. The option copies
// its input so later caller mutations cannot change compilation.
func WithNameExpressions(expressions ...string) Option {
	names := append([]string(nil), expressions...)
	return func(o *options) {
		for _, name := range names {
			if name == "" || strings.ContainsRune(name, 0) {
				o.err = fmt.Errorf("nvrtcx: invalid name expression %q", name)
				return
			}
		}
		o.expressions = append(o.expressions, names...)
	}
}

// LoweredName returns the copied compiler name for a registered expression.
// The name remains valid after the NVRTC program is destroyed.
func (cm *CompiledModule) LoweredName(expression string) (string, bool) {
	if cm == nil {
		return "", false
	}
	name, ok := cm.lowered[expression]
	return name, ok
}

// WithName specifies the virtual filename of the CUDA program.
func WithName(name string) Option {
	return func(o *options) {
		if name == "" {
			o.err = fmt.Errorf("nvrtcx: empty program name")
			return
		}
		if strings.ContainsRune(name, 0) {
			o.err = fmt.Errorf("nvrtcx: program name contains NUL")
			return
		}
		o.name = name
	}
}

// WithArch configures the target virtual compute architecture (for example,
// compute_80 for major=8, minor=0). Use WithArch when generating virtual PTX
// for multi-target binaries or cross-compilation.
func WithArch(major, minor int) Option {
	return func(o *options) {
		if major <= 0 || minor < 0 {
			o.err = fmt.Errorf("nvrtcx: invalid compute architecture %d.%d", major, minor)
			return
		}
		o.flags = append(o.flags, fmt.Sprintf("--gpu-architecture=compute_%d%d", major, minor))
	}
}

// WithPhysicalArch configures the target physical GPU architecture (for example,
// sm_80 for major=8, minor=0). When compiled with default OutputAuto, this generates
// native CUBIN machine code while retaining PTX for inspection. Load automatically
// loads the CUBIN onto the device, avoiding driver-side JIT compilation and
// preventing PTX version incompatibilities.
func WithPhysicalArch(major, minor int) Option {
	return func(o *options) {
		if major <= 0 || minor < 0 {
			o.err = fmt.Errorf("nvrtcx: invalid compute architecture %d.%d", major, minor)
			return
		}
		if o.outputSet && (o.output == OutputLTOIR || o.output == OutputTileIR || o.output == OutputPTXTileIR) {
			o.err = fmt.Errorf("nvrtcx: physical architecture cannot combine with WithOutput(%s)", outputKindName(o.output))
			return
		}
		o.flags = append(o.flags, fmt.Sprintf("--gpu-architecture=sm_%d%d", major, minor))
		o.physicalArch = true
	}
}

// WithDeviceArch queries dev's compute capability and targets its physical GPU
// architecture with direct OutputCUBIN binary generation. This avoids driver-side
// JIT compilation and prevents PTX version incompatibilities.
func WithDeviceArch(dev *cuda.Device) Option {
	if dev == nil {
		return func(o *options) {
			o.err = fmt.Errorf("nvrtcx: nil device passed to WithDeviceArch")
		}
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return func(o *options) {
			o.err = fmt.Errorf("nvrtcx: query device compute capability: %w", err)
		}
	}
	return WithPhysicalArch(major, minor)
}

// WithFlag adds a compiler flag to the NVRTC compilation.
func WithFlag(flag string) Option {
	return func(o *options) {
		if flag == "" {
			o.err = fmt.Errorf("nvrtcx: empty compiler flag")
			return
		}
		if strings.ContainsRune(flag, 0) {
			o.err = fmt.Errorf("nvrtcx: compiler flag contains NUL")
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
		if strings.ContainsRune(name, 0) || strings.ContainsRune(content, 0) {
			o.err = fmt.Errorf("nvrtcx: header name or content contains NUL")
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

// LoadOutput loads one selected artifact. It never falls back to another format.
// The caller must unload the returned module when it is no longer needed.
func (cm *CompiledModule) LoadOutput(dev *cuda.Device, kind OutputKind) (driver.Module, error) {
	if cm == nil {
		return 0, fmt.Errorf("nvrtcx: nil compiled module")
	}
	selected := &CompiledModule{output: kind}
	switch kind {
	case OutputPTX:
		selected.ptx = cm.ptx
	case OutputCUBIN:
		selected.cubin = cm.cubin
	case OutputTileIR:
		selected.tileir = cm.tileir
	default:
		return 0, fmt.Errorf("nvrtcx: output %d cannot be loaded directly", kind)
	}
	if !selected.validArtifact() {
		return 0, fmt.Errorf("nvrtcx: output %d is unavailable", kind)
	}
	return selected.Load(dev)
}

// Compile compiles the given CUDA C++ source code into a CompiledModule.
// Source, filenames, header contents, and flags must not contain NUL bytes.
// If program destruction fails, the returned error retains the native program
// and implements io.Closer (reachable with errors.As). Call Close on that error
// to retry cleanup. Such a result is not added to the compilation cache.
func Compile(src string, opts ...Option) (*CompiledModule, error) {
	if src == "" {
		return nil, fmt.Errorf("nvrtcx: empty source")
	}
	if strings.ContainsRune(src, 0) {
		return nil, fmt.Errorf("nvrtcx: source contains NUL")
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
	if cfg.physicalArch && cfg.outputSet && (cfg.output == OutputLTOIR || cfg.output == OutputTileIR || cfg.output == OutputPTXTileIR) {
		return nil, fmt.Errorf("nvrtcx: physical architecture cannot combine with WithOutput(%s)", outputKindName(cfg.output))
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
	identity := fmt.Sprintf("nvrtc %d.%d %s output=%d cache=%q", major, minor, nvrtc.Path(), cfg.output, cacheMagic)
	inputKey := computeCacheKey(src, cfg.name, cfg.flags, cfg.headers)
	inputKey = computeCacheKey(inputKey, "name-expressions-v1", cfg.expressions, nil)
	key := compilerCacheKey(inputKey, identity)
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

		for _, expression := range cfg.expressions {
			name := append([]byte(expression), 0)
			err := prog.AddNameExpression(&name[0])
			runtime.KeepAlive(name)
			if err != nil {
				return nil, fmt.Errorf("nvrtcx: register %q: %w", expression, err)
			}
		}
		compileErr := prog.Compile(cfg.flags)
		log, logErr := prog.GetProgramLog()

		if err := compilerDiagnostic(compileErr, logErr, log); err != nil {
			return nil, err
		}

		res, err := getArtifact(prog, cfg.output)
		if err != nil {
			return nil, fmt.Errorf("%w\ncompiler log:\n%s", err, log)
		}
		if len(cfg.expressions) > 0 {
			res.lowered = make(map[string]string)
		}
		for _, expression := range cfg.expressions {
			name := append([]byte(expression), 0)
			var ptr *byte
			err := prog.GetLoweredName(&name[0], &ptr)
			runtime.KeepAlive(name)
			if err != nil {
				return nil, fmt.Errorf("nvrtcx: lower %q: %w", expression, err)
			}
			if ptr == nil {
				return nil, fmt.Errorf("nvrtcx: lower %q: nil name", expression)
			}
			var copied []byte
			for p := unsafe.Pointer(ptr); ; p = unsafe.Add(p, 1) {
				b := *(*byte)(p)
				if b == 0 {
					break
				}
				copied = append(copied, b)
			}
			if len(copied) == 0 {
				return nil, fmt.Errorf("nvrtcx: lower %q: empty name", expression)
			}
			res.lowered[expression] = string(copied)
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
