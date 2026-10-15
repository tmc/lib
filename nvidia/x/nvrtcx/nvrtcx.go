package nvrtcx

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// CompiledModule represents the compiled output of an NVRTC JIT compilation.
type CompiledModule struct {
	lowered      map[string]string
	output       OutputKind
	ltoir        []byte
	tileir       []byte
	optixir      []byte
	ptx          string
	cubin        []byte
	log          string
	key          string
	jitOptions   cudakernel.JITOptions
	dependencies []cudakernel.Dependency
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
	if cm.output == OutputOptiXIR || len(cm.optixir) > 0 {
		return 0, fmt.Errorf("nvrtcx: OptiX IR must be passed to optixModuleCreate")
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
			m, err := driver.ModuleLoadData(cm.tileir)
			mod = m
			return err
		}
		if len(cm.cubin) > 0 {
			m, err := driver.ModuleLoadData(cm.cubin)
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
			jitOpts, jitVals := cm.jitOptions.DriverOptions()
			var m driver.Module
			var err error
			if len(jitOpts) > 0 {
				m, err = driver.ModuleLoadDataEx(cstr, uint32(len(jitOpts)), &jitOpts[0], &jitVals[0])
			} else {
				m, err = driver.ModuleLoadData(cstr)
			}
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
			if err := pending.Close(); err != nil {
				return err
			}
			pending = 0
			return nil
		})
	})
}

// JITOptions returns the driver JIT compiler options attached to the module.
func (cm *CompiledModule) JITOptions() cudakernel.JITOptions {
	if cm == nil {
		return cudakernel.JITOptions{}
	}
	return cm.jitOptions
}

// SetJITOptions updates the driver JIT compiler options attached to the module.
func (cm *CompiledModule) SetJITOptions(opts cudakernel.JITOptions) {
	if cm != nil {
		cm.jitOptions = opts
	}
}

// Dependencies returns recorded external compilation dependencies.
func (cm *CompiledModule) Dependencies() []cudakernel.Dependency {
	if cm == nil {
		return nil
	}
	return append([]cudakernel.Dependency(nil), cm.dependencies...)
}

// ToArtifact converts the CompiledModule into a validated cudakernel.Artifact.
// It distinguishes PTX, CUBIN, LTOIR, and TileIR, and marks unlinked formats.
func (cm *CompiledModule) ToArtifact(entryPoint string, schema cudakernel.Schema, target string) (*cudakernel.Artifact, error) {
	if cm == nil {
		return nil, fmt.Errorf("nvrtcx: nil compiled module")
	}

	var kind cudakernel.Kind
	var raw []byte

	switch cm.output {
	case OutputPTX:
		kind = cudakernel.KindPTX
		raw = []byte(cm.ptx)
	case OutputCUBIN:
		kind = cudakernel.KindCUBIN
		raw = cm.cubin
	case OutputLTOIR:
		kind = cudakernel.KindLTOIR
		raw = cm.ltoir
	case OutputTileIR:
		kind = cudakernel.KindTileIR
		raw = cm.tileir
	case OutputAuto:
		if len(cm.cubin) > 0 {
			kind = cudakernel.KindCUBIN
			raw = cm.cubin
		} else if cm.ptx != "" {
			kind = cudakernel.KindPTX
			raw = []byte(cm.ptx)
		}
	case OutputPTXTileIR:
		if len(cm.tileir) > 0 {
			kind = cudakernel.KindTileIR
			raw = cm.tileir
		} else {
			kind = cudakernel.KindPTX
			raw = []byte(cm.ptx)
		}
	default:
		return nil, fmt.Errorf("nvrtcx: unsupported output kind %d for artifact conversion", cm.output)
	}

	if len(raw) == 0 {
		return nil, fmt.Errorf("nvrtcx: compiled module contains no %s data", kind)
	}

	var res cudakernel.ResourceRequirements
	if kind == cudakernel.KindCUBIN {
		if elfInfo, err := cudakernel.ParseCUBIN(raw); err == nil {
			if regs, ok := elfInfo.AllocatedRegisters[entryPoint]; ok {
				res.AllocatedRegisters = regs
			}
			if smem, ok := elfInfo.StaticSharedMemory[entryPoint]; ok {
				res.StaticSharedMemory = smem
			}
		}
	}

	return cudakernel.NewArtifact(
		kind,
		raw,
		entryPoint,
		schema,
		target,
		cudakernel.WithResources(res),
		cudakernel.WithJIT(cm.jitOptions),
		cudakernel.WithToolchain("nvrtc"),
		cudakernel.WithDependencies(cm.dependencies),
	)
}

type options struct {
	expressions       []string
	output            OutputKind
	outputSet         bool
	physicalArch      bool
	name              string
	flags             []string
	headers           map[string]string
	useCache          bool
	err               error
	lineInfo          bool
	noLineInfo        bool
	deviceDebug       bool
	toolkitIncludes   bool
	noToolkitIncludes bool
	bundledHeaders    bool
	bundledHeadersDir string
	ctx               context.Context
	jitOptions        cudakernel.JITOptions
	strictNVRTC       bool
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
		o.flags = append(o.flags, nvrtc.FlagGpuArchitecture+"="+nvrtc.ComputeArch(major, minor))
	}
}

// WithPhysicalArch configures the target physical GPU architecture (for example,
// sm_80 for major=8, minor=0, or sm_121a for major=12, minor=1, suffix="a").
// When compiled with default OutputAuto, this generates native CUBIN machine code
// while retaining PTX for inspection. Load automatically loads the CUBIN onto the
// device, avoiding driver-side JIT compilation and preventing PTX version incompatibilities.
func WithPhysicalArch(major, minor int, suffix ...string) Option {
	return func(o *options) {
		if major <= 0 || minor < 0 {
			o.err = fmt.Errorf("nvrtcx: invalid compute architecture %d.%d", major, minor)
			return
		}
		if o.outputSet && (o.output == OutputLTOIR || o.output == OutputTileIR || o.output == OutputPTXTileIR) {
			o.err = fmt.Errorf("nvrtcx: physical architecture cannot combine with WithOutput(%s)", outputKindName(o.output))
			return
		}
		target := nvrtc.SmArch(major, minor)
		if len(suffix) > 0 && suffix[0] != "" {
			target += suffix[0]
		}
		o.flags = append(o.flags, nvrtc.FlagGpuArchitecture+"="+target)
		o.physicalArch = true
	}
}

// WithPhysicalArchTarget configures the target physical GPU architecture by name
// (for example, "sm_121a" or "sm_100a").
func WithPhysicalArchTarget(target string) Option {
	return func(o *options) {
		target = strings.TrimSpace(target)
		if target == "" {
			o.err = fmt.Errorf("nvrtcx: empty physical architecture target")
			return
		}
		if !strings.HasPrefix(target, "sm_") {
			target = "sm_" + target
		}
		if o.outputSet && (o.output == OutputLTOIR || o.output == OutputTileIR || o.output == OutputPTXTileIR) {
			o.err = fmt.Errorf("nvrtcx: physical architecture cannot combine with WithOutput(%s)", outputKindName(o.output))
			return
		}
		o.flags = append(o.flags, nvrtc.FlagGpuArchitecture+"="+target)
		o.physicalArch = true
	}
}

// WithJITOptions configures driver JIT compilation options for PTX loading.
func WithJITOptions(jit cudakernel.JITOptions) Option {
	return func(o *options) {
		o.jitOptions = jit
	}
}

// WithStrictNVRTC enforces strict source-level option validation,
// explicitly rejecting offline nvcc/ptxas flags.
func WithStrictNVRTC() Option {
	return func(o *options) {
		o.strictNVRTC = true
	}
}

// WithDeviceArch queries dev's compute capability and targets its physical GPU
// architecture. When compiled with default OutputAuto, this generates native CUBIN
// machine code while retaining PTX for inspection. Load automatically loads the CUBIN
// onto the device, avoiding driver-side JIT compilation and preventing PTX version
// incompatibilities.
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
		if err := nvrtc.ValidateOption(flag); err != nil {
			o.err = fmt.Errorf("nvrtcx: invalid compiler flag %q: %w", flag, err)
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

// WithNoCache disables the compilation cache for this compilation pass,
// and passes --no-cache to disable NVRTC's own cache. NVRTC releases before
// 12.9 have no cache and reject the flag, so Compile omits it there.
func WithNoCache() Option {
	return func(o *options) {
		o.useCache = false
		if !hasFlag(o.flags, "--no-cache") {
			o.flags = append(o.flags, "--no-cache")
		}
	}
}

// hasNVRTCCache reports whether NVRTC major.minor has its own cache
// and accepts --no-cache. Both arrived in 12.9; earlier releases
// reject the flag with NVRTC_ERROR_INVALID_OPTION.
func hasNVRTCCache(major, minor int32) bool {
	return major > 12 || major == 12 && minor >= 9
}

func isNoCacheFlag(f string) bool {
	return f == "--no-cache" || f == "-no-cache"
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
	case OutputOptiXIR:
		return 0, fmt.Errorf("nvrtcx: OptiX IR must be passed to optixModuleCreate")
	default:
		return 0, fmt.Errorf("nvrtcx: output %d cannot be loaded directly", kind)
	}
	if !selected.validArtifact() {
		return 0, fmt.Errorf("nvrtcx: output %d is unavailable", kind)
	}
	return selected.Load(dev)
}

// LoadModule loads the compiled module into the CUDA driver on dev and wraps it
// in a managed Module. If wrapping fails, it unloads the driver module. If
// unloading also fails, the error implements io.Closer for a cleanup retry.
func (cm *CompiledModule) LoadModule(dev *cuda.Device) (*cuda.Module, error) {
	raw, err := cm.Load(dev)
	if err != nil {
		return nil, err
	}
	mod, err := cuda.OwnModule(dev, raw)
	if err == nil {
		return mod, nil
	}
	_, err = finishModuleLoad(raw, err, func() error {
		return dev.WithContext(raw.Close)
	})
	return nil, err
}

// LoadOutputModule loads one selected artifact into the CUDA driver on dev and
// wraps it in a managed Module. If wrapping fails, it unloads the driver module.
// If unloading also fails, the error implements io.Closer for a cleanup retry.
func (cm *CompiledModule) LoadOutputModule(dev *cuda.Device, kind OutputKind) (*cuda.Module, error) {
	raw, err := cm.LoadOutput(dev, kind)
	if err != nil {
		return nil, err
	}
	mod, err := cuda.OwnModule(dev, raw)
	if err == nil {
		return mod, nil
	}
	_, err = finishModuleLoad(raw, err, func() error {
		return dev.WithContext(raw.Close)
	})
	return nil, err
}

// Bind loads the compiled module onto dev and binds its exported kernels to
// fields in kernels using [cudalaunch.BindOwned]. If binding fails, the loaded
// module is closed before returning the error. If closing fails, Bind returns
// both the module and the error so the caller can retry Close.
func (cm *CompiledModule) Bind(dev *cuda.Device, kernels any) (*cuda.Module, error) {
	mod, err := cm.LoadModule(dev)
	if err != nil {
		return nil, err
	}
	if err := cudalaunch.BindOwned(mod, kernels); err != nil {
		if closeErr := mod.Close(); closeErr != nil {
			return mod, errors.Join(err, closeErr)
		}
		return nil, err
	}
	return mod, nil
}

// CompileContext compiles the given CUDA C++ source code with context cancellation support.
func CompileContext(ctx context.Context, src string, opts ...Option) (*CompiledModule, error) {
	if ctx == nil {
		return nil, fmt.Errorf("nvrtcx: nil context")
	}
	allOpts := make([]Option, 0, len(opts)+1)
	allOpts = append(allOpts, WithContext(ctx))
	allOpts = append(allOpts, opts...)
	return Compile(src, allOpts...)
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
		lineInfo: true,
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
	if cfg.ctx != nil && cfg.ctx.Err() != nil {
		return nil, fmt.Errorf("nvrtcx: compile: %w", cfg.ctx.Err())
	}
	if cfg.physicalArch && cfg.outputSet && (cfg.output == OutputLTOIR || cfg.output == OutputTileIR || cfg.output == OutputPTXTileIR || cfg.output == OutputOptiXIR) {
		return nil, fmt.Errorf("nvrtcx: physical architecture cannot combine with WithOutput(%s)", outputKindName(cfg.output))
	}

	// Disable cache if --no-cache was requested in custom flags
	if slices.ContainsFunc(cfg.flags, isNoCacheFlag) {
		cfg.useCache = false
	}

	// Apply line info default: on by default for profilers and debuggers,
	// unless WithNoLineInfo was requested, device debug is active (-G),
	// or line info is already passed explicitly in custom flags.
	if cfg.lineInfo && !cfg.noLineInfo && !cfg.deviceDebug && !hasLineInfoFlag(cfg.flags) {
		cfg.flags = append(cfg.flags, "--generate-line-info")
	}

	// Apply toolkit includes or bundled headers:
	if !cfg.noToolkitIncludes {
		needHeaders := cfg.toolkitIncludes || cfg.bundledHeaders || (strings.Contains(src, "#include") && !hasIncludeFlag(cfg.flags))
		if needHeaders {
			dir := ToolkitIncludeDir()
			if dir != "" && !cfg.bundledHeaders {
				flag := "--include-path=" + dir
				if !hasFlag(cfg.flags, flag) {
					cfg.flags = append(cfg.flags, flag)
				}
			} else if cfg.bundledHeaders || nvrtc.HasInstallBundledHeaders() {
				targetDir := cfg.bundledHeadersDir
				if targetDir == "" {
					targetDir = DefaultBundledHeadersDir()
				}
				installedDir, err := InstallBundledHeaders(targetDir, false)
				if err != nil {
					if cfg.bundledHeaders || cfg.toolkitIncludes {
						return nil, err
					}
				} else {
					flag := "--include-path=" + installedDir
					if !hasFlag(cfg.flags, flag) {
						cfg.flags = append(cfg.flags, flag)
					}
					ccclDir := filepath.Join(installedDir, "cccl")
					if fi, err := os.Stat(ccclDir); err == nil && fi.IsDir() {
						ccclFlag := "--include-path=" + ccclDir
						if !hasFlag(cfg.flags, ccclFlag) {
							cfg.flags = append(cfg.flags, ccclFlag)
						}
					}
				}
			} else if cfg.toolkitIncludes {
				return nil, fmt.Errorf("nvrtcx: CUDA toolkit include directory not found")
			}
		}
	}

	flags, err := outputFlags(cfg.output, cfg.flags)
	if err != nil {
		return nil, err
	}
	cfg.flags = flags

	// Two-tier option separation:
	// NVRTC Phase strictly accepts source-level options.
	// Offline flags (-maxrregcount, -Xptxas, -O3, etc.) are stripped from
	// the compilation flags and translated into Driver JIT options.
	sourceFlags, offlineFlags, extractedJIT, _ := SeparateFlags(cfg.flags)
	if cfg.strictNVRTC && len(offlineFlags) > 0 {
		return nil, fmt.Errorf("nvrtcx: strict NVRTC rejected offline flags: %s", strings.Join(offlineFlags, ", "))
	}
	if cfg.jitOptions.MaxRegisters == 0 && extractedJIT.MaxRegisters > 0 {
		cfg.jitOptions.MaxRegisters = extractedJIT.MaxRegisters
	}
	if cfg.jitOptions.OptLevel == 0 && extractedJIT.OptLevel > 0 {
		cfg.jitOptions.OptLevel = extractedJIT.OptLevel
	}

	// External dependency tracking:
	// Check external headers included via #include and hash their contents.
	// If any external dependency cannot be validated, disable cache to prevent
	// silent reuse of stale cache entries when untracked inputs change.
	deps, allTracked := resolveExternalDependencies(src, cfg.flags, cfg.headers)
	if !allTracked {
		cfg.useCache = false
	}

	// Resolve the loaded compiler before looking up any compiled artifact.
	// A cache hit must not bypass a missing or incompatible compiler installation.
	major, minor, err := nvrtc.Version()
	if err != nil {
		return nil, fmt.Errorf("nvrtcx: compiler version: %w", err)
	}
	if !hasNVRTCCache(major, minor) {
		cfg.flags = slices.DeleteFunc(cfg.flags, isNoCacheFlag)
		sourceFlags = slices.DeleteFunc(sourceFlags, isNoCacheFlag)
	}
	toolIdentity, toolQualified, err := loadedCompilerIdentity(nvrtc.Path())
	if err != nil {
		return nil, err
	}
	if !toolQualified {
		cfg.useCache = false
	}
	identity := fmt.Sprintf("nvrtc %d.%d %s %s output=%d cache=%q", major, minor, nvrtc.Path(), toolIdentity, cfg.output, cacheMagic)
	inputKey := computeCacheKey(src, cfg.name, cfg.flags, cfg.headers)
	inputKey = computeCacheKey(inputKey, "name-expressions-v1", cfg.expressions, nil)
	if len(deps) > 0 && allTracked {
		depKey := computeDependencyKey(deps)
		inputKey = computeCacheKey(inputKey, "external-deps-v1", []string{depKey}, nil)
	}
	key := compilerCacheKey(inputKey, identity)
	if cfg.useCache {
		if cached, ok := lookupCache(key); ok {
			if err := verifyLoadedCompilerIdentity(nvrtc.Path(), toolIdentity); err != nil {
				return nil, err
			}
			return cached, nil
		}
	}

	return singleFlightCompile(key, func() (result *CompiledModule, resultErr error) {
		// Recheck cache inside singleflight
		if cfg.useCache {
			if cached, ok := lookupCache(key); ok {
				if err := verifyLoadedCompilerIdentity(nvrtc.Path(), toolIdentity); err != nil {
					return nil, err
				}
				return cached, nil
			}
		}

		headers := make([]nvrtc.Header, 0, len(cfg.headers))
		for name, source := range cfg.headers {
			headers = append(headers, nvrtc.Header{Name: name, Source: source})
		}
		sort.Slice(headers, func(i, j int) bool { return headers[i].Name < headers[j].Name })

		// compileOnce creates and compiles a program. A non-nil err means
		// no program is left to destroy.
		compileOnce := func() (prog nvrtc.Program, log string, compileErr, logErr, err error) {
			prog, err = nvrtc.CreateProgram(src, cfg.name, headers...)
			if err != nil {
				return 0, "", nil, nil, fmt.Errorf("nvrtcx: create program: %w", err)
			}
			for _, expression := range cfg.expressions {
				name := append([]byte(expression), 0)
				err := prog.AddNameExpression(&name[0])
				runtime.KeepAlive(name)
				if err != nil {
					_ = prog.Destroy()
					return 0, "", nil, nil, fmt.Errorf("nvrtcx: register %q: %w", expression, err)
				}
			}
			if cfg.ctx != nil && nvrtc.HasSetFlowCallback() {
				token, unregister := registerFlowContext(cfg.ctx)
				defer unregister()
				if err := prog.SetFlowCallback(getFlowCallback(), token); err != nil {
					_ = prog.Destroy()
					return 0, "", nil, nil, fmt.Errorf("nvrtcx: set flow callback: %w", err)
				}
			}
			compileErr = prog.Compile(sourceFlags...)
			log, logErr = prog.Log()
			return prog, log, compileErr, logErr, nil
		}

		prog, log, compileErr, logErr, err := compileOnce()
		if err != nil {
			return nil, err
		}
		if compileErr != nil && cfg.ctx != nil && cfg.ctx.Err() != nil {
			_ = prog.Destroy()
			return nil, fmt.Errorf("nvrtcx: compile: %w", cfg.ctx.Err())
		}

		// Handle PCH heap exhaustion retry
		isHeapExhausted := errors.Is(compileErr, nvrtc.ErrPchCreateHeapExhausted)
		if !isHeapExhausted && compileErr == nil && nvrtc.HasGetPCHCreateStatus() {
			if errors.Is(prog.PCHCreateStatus(), nvrtc.ErrPchCreateHeapExhausted) {
				isHeapExhausted = true
			}
		}
		if isHeapExhausted && nvrtc.HasGetPCHHeapSizeRequired() && nvrtc.HasSetPCHHeapSize() {
			reqSize, reqErr := prog.PCHHeapSizeRequired()
			if reqErr == nil && reqSize > 0 {
				if setErr := nvrtc.SetPCHHeapSize(reqSize); setErr == nil {
					_ = prog.Destroy()
					prog, log, compileErr, logErr, err = compileOnce()
					if err != nil {
						return nil, err
					}
					if compileErr != nil && cfg.ctx != nil && cfg.ctx.Err() != nil {
						_ = prog.Destroy()
						return nil, fmt.Errorf("nvrtcx: compile: %w", cfg.ctx.Err())
					}
				}
			}
		}

		defer func() {
			var cache func(*CompiledModule) error
			if cfg.useCache {
				cache = func(result *CompiledModule) error { return storeCache(key, result) }
			}
			result, resultErr = finishCompilation(result, resultErr,
				func() error { return prog.Destroy() },
				func() error { return verifyLoadedCompilerIdentity(nvrtc.Path(), toolIdentity) }, cache)
		}()

		if err := compilerDiagnostic(compileErr, logErr, log); err != nil {
			return nil, err
		}

		res, err := getArtifact(prog, cfg.output)
		if err != nil {
			return nil, fmt.Errorf("%w\ncompiler log:\n%s", err, log)
		}
		if cfg.physicalArch && len(res.cubin) == 0 {
			return nil, fmt.Errorf("nvrtcx: physical architecture compilation produced no CUBIN\ncompiler log:\n%s", log)
		}
		if len(cfg.expressions) > 0 {
			res.lowered = make(map[string]string)
		}
		for _, expression := range cfg.expressions {
			name := append([]byte(expression), 0)
			var ptr *byte
			err := prog.LoweredName(&name[0], &ptr)
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
		res.jitOptions = cfg.jitOptions
		res.dependencies = deps
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
