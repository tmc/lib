package cudacheck

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const traceCacheRecipe = "cudacheck/generated-ptx-cache/v1"

// ErrArtifactCacheUnsupported reports a request outside the generated graph
// cache recipe. Uncached Compare remains available for other configurations.
var ErrArtifactCacheUnsupported = errors.New("cudacheck: unsupported artifact cache request")

type perturbationCacheConfig struct {
	adapter *cudacache.NVRTCAdapter
	target  string
}

// WithArtifactCache enables a generated graph PTX recipe with an explicit
// compute target, for example compute_80. It refuses custom kernel sources and
// caller CompilerOpts rather than replaying arbitrary option callbacks. The
// recipe uses C++17, no line information or external headers, and the requested
// JIT options and perturbation vectorization bit. Both inner compilation caches
// are disabled; this option explicitly enables the outer artifact cache.
//
// NVRTC, its builtins and the driver must already be mapped and identifiable.
// This capability currently requires Linux mapped-library inspection. It loads
// no libraries to qualify a missing environment. Tool files must remain stable
// through compilation and lookup. A verified editable manifest is a metadata
// binding, not authentication or device qualification. Failed module cleanup is
// retained by an io.Closer reachable through the returned result's Err.
// Keep the CUDA device open until that cleanup succeeds.
func WithArtifactCache(adapter *cudacache.NVRTCAdapter, target string) CompareOption {
	return func(c *CompareConfig) {
		c.artifactCache = &perturbationCacheConfig{adapter: adapter, target: target}
	}
}

// The test seam supplies declared host identities; production always probes
// real mapped tools. Neither branch treats synthetic identity as a compiler.
type traceCacheEnvironment func(string, []string, cudakernel.JITOptions) (cudacache.CompilerIdentity, error)

type traceCacheRequest struct {
	source, entry, target string
	schema                cudakernel.Schema
	flags                 []string
	options               []nvrtcx.Option
	jit                   cudakernel.JITOptions
}

func makeTraceCacheRequest(cfg *CompareConfig, source, entry string, snap *cudair.Snapshot, generated bool) (traceCacheRequest, error) {
	var r traceCacheRequest
	cache := cfg.artifactCache
	if cache == nil || cache.adapter == nil || !generated || len(cfg.CompilerOpts) != 0 || !strings.HasPrefix(cache.target, "compute_") {
		return r, ErrArtifactCacheUnsupported
	}
	if err := cudakernel.ValidateTargetArch(cache.target); err != nil {
		return r, fmt.Errorf("%w: %w", ErrArtifactCacheUnsupported, err)
	}
	if err := cfg.JITOptions.Validate(); err != nil {
		return r, err
	}
	params := []cudakernel.Param{{Name: "num_elements", Type: "s32"}, {Name: "trace_fold_limit", Type: "u32"}}
	for _, arg := range kernelArgumentLayout(snap) {
		name := "in_" + arg.Name
		if arg.Output {
			name = "out_" + arg.Name
		}
		params = append(params, cudakernel.Param{Name: name, Type: "ptr", IsPointer: true})
	}
	schema, err := cudakernel.NewSchema(params...)
	if err != nil {
		return r, err
	}
	r = traceCacheRequest{source: source, entry: entry, target: cache.target, schema: schema, jit: cfg.JITOptions}
	r.options = []nvrtcx.Option{nvrtcx.WithName("cudacheck_trace.cu"), nvrtcx.WithStandard("c++17"), nvrtcx.WithFlag("--gpu-architecture=" + r.target), nvrtcx.WithOutput(nvrtcx.OutputPTX), nvrtcx.WithNoToolkitIncludes(), nvrtcx.WithNoLineInfo(), nvrtcx.WithNoCache(), nvrtcx.WithJITOptions(r.jit)}
	if cfg.perturbation != nil && cfg.perturbation.ExtraDeviceVectorize {
		r.options = append(r.options, nvrtcx.WithExtraDeviceVectorization())
	}
	r.flags, err = nvrtcx.CompileFlags(r.options...)
	return r, err
}

func (r traceCacheRequest) inputKey() string {
	// JIT and effective flags are in CompilerIdentity, whose key normalizes
	// positive OptLevel spellings. Do not hash their raw JSON a second time.
	value := struct {
		Recipe, Source, Program, Entry, Policy string
		Schema                                 cudakernel.Schema
	}{traceCacheRecipe, r.source, "cudacheck_trace.cu", r.entry, "raw-bits/generated-fold-v1", r.schema}
	data, _ := json.Marshal(value)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func traceCacheFlags(flags []string, version string) ([]string, error) {
	majorText, minorText, ok := strings.Cut(version, ".")
	major, err1 := strconv.Atoi(majorText)
	minor, err2 := strconv.Atoi(minorText)
	if !ok || err1 != nil || err2 != nil || major < 1 || minor < 0 {
		return nil, fmt.Errorf("%w: unknown NVRTC version", ErrArtifactCacheUnsupported)
	}
	flags = slices.Clone(flags)
	if major < 12 || major == 12 && minor < 9 {
		flags = slices.DeleteFunc(flags, func(f string) bool { return f == "--no-cache" })
	}
	return flags, nil
}

func observeTraceCacheEnvironment(target string, flags []string, jit cudakernel.JITOptions) (cudacache.CompilerIdentity, error) {
	if !nvrtc.Loaded() || !driver.Loaded() {
		return cudacache.CompilerIdentity{}, fmt.Errorf("%w: NVRTC and driver must already be loaded", ErrArtifactCacheUnsupported)
	}
	id, err := cudacache.CurrentEnvironmentIdentity(target, flags, nvrtcx.OutputPTX)
	if err != nil {
		return id, err
	}
	if len(id.Provenance.Tools) != 2 || id.Provenance.DriverVersion <= 0 {
		return id, fmt.Errorf("%w: incomplete tool provenance", ErrArtifactCacheUnsupported)
	}
	// ProbeToolchain covers NVRTC and driver. Bind NVRTC's already mapped
	// builtins too, without loading a guessed path or copying another DSO.
	path, err := cudacache.LoadedLibraryPath("libnvrtc-builtins.so")
	if err != nil {
		return id, fmt.Errorf("%w: builtins: %w", ErrArtifactCacheUnsupported, err)
	}
	path, err = filepath.EvalSymlinks(path)
	if err != nil {
		return id, err
	}
	builtins, err := cudacache.DigestTool("nvrtc-builtins", path)
	if err != nil {
		return id, err
	}
	id.Provenance.Tools = append(id.Provenance.Tools, builtins)
	id.Provenance.Recipe = traceCacheRecipe
	id.JITOptions = jit
	return id, nil
}

func (r traceCacheRequest) environment(probe traceCacheEnvironment) (cudacache.CompilerIdentity, error) {
	id, err := probe(r.target, r.flags, r.jit)
	if err != nil {
		return id, err
	}
	flags, err := traceCacheFlags(r.flags, id.NVRTCVersion)
	if err != nil {
		return id, err
	}
	if id.TargetArch != r.target || id.OutputKind != nvrtcx.OutputPTX || !sameTraceJIT(id.JITOptions, r.jit) || !slices.Equal(id.Flags, r.flags) {
		return id, fmt.Errorf("%w: environment request differs", ErrPerturbationArtifactMismatch)
	}
	id.Flags = flags
	return id, nil
}

func sameTraceJIT(a, b cudakernel.JITOptions) bool {
	x, y := a.DriverOptions()
	u, v := b.DriverOptions()
	return slices.Equal(x, u) && slices.Equal(y, v)
}

func (r traceCacheRequest) check(id cudacache.CompilerIdentity, art *cudakernel.Artifact) error {
	if err := art.Validate(); err != nil {
		return err
	}
	if art.Kind != cudakernel.KindPTX || art.EntryPoint != r.entry || art.Target != r.target || !reflect.DeepEqual(art.Schema, r.schema) || !slices.Equal(art.CompilerOpts, id.Flags) || !sameTraceJIT(art.JITOptions, r.jit) || len(art.Dependencies) != 0 {
		return fmt.Errorf("%w: cached source recipe, ABI or options differ", ErrPerturbationArtifactMismatch)
	}
	return nil
}

func compileCachedTrace(ctx context.Context, cfg *CompareConfig, source, entry string, snap *cudair.Snapshot, generated bool) (*cudakernel.Artifact, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r, err := makeTraceCacheRequest(cfg, source, entry, snap, generated)
	if err != nil {
		return nil, err
	}
	probe := cfg.cacheEnvironment
	if probe == nil {
		probe = observeTraceCacheEnvironment
	}
	id, err := r.environment(probe)
	if err != nil {
		return nil, err
	}
	var compileErr error
	art, err := cfg.artifactCache.adapter.GetOrCompile(ctx, r.inputKey(), id, func(ctx context.Context) (*cudakernel.Artifact, cudacache.ToolProvenance, error) {
		opts := append(slices.Clone(r.options), nvrtcx.WithContext(ctx))
		compile := cfg.compileKernel
		if compile == nil {
			compile = func(src string, opts []nvrtcx.Option, _ cudakernel.JITOptions) (*nvrtcx.CompiledModule, error) {
				return nvrtcx.Compile(src, opts...)
			}
		}
		cm, err := compile(source, opts, r.jit)
		compileErr = err // Keep cleanup custody if later revalidation also fails.
		if err != nil {
			return nil, cudacache.ToolProvenance{}, err
		}
		if err := ctx.Err(); err != nil {
			return nil, cudacache.ToolProvenance{}, err
		}
		if cm == nil {
			return nil, cudacache.ToolProvenance{}, fmt.Errorf("%w: nil compiler result", ErrPerturbationArtifactMismatch)
		}
		actual, known := cm.CompilerOptions()
		if !known || !slices.Equal(actual, id.Flags) || !sameTraceJIT(cm.JITOptions(), r.jit) {
			return nil, cudacache.ToolProvenance{}, fmt.Errorf("%w: compiler options differ or were not observed", ErrPerturbationArtifactMismatch)
		}
		produced, err := cm.ToArtifact(entry, r.schema, r.target)
		if err != nil {
			return nil, cudacache.ToolProvenance{}, err
		}
		if err := r.check(id, produced); err != nil {
			return nil, cudacache.ToolProvenance{}, err
		}
		post, err := r.environment(probe)
		if err != nil {
			return nil, cudacache.ToolProvenance{}, err
		}
		if cudacache.CompilerCacheKey(r.inputKey(), id) != cudacache.CompilerCacheKey(r.inputKey(), post) {
			return nil, cudacache.ToolProvenance{}, cudacache.ErrToolIdentityChanged
		}
		return produced, post.Provenance, ctx.Err()
	})
	if err != nil {
		return nil, errors.Join(err, compileErr)
	}
	if err := r.check(id, art); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	post, err := r.environment(probe)
	if err != nil {
		return nil, err
	}
	if cudacache.CompilerCacheKey(r.inputKey(), id) != cudacache.CompilerCacheKey(r.inputKey(), post) {
		return nil, cudacache.ErrToolIdentityChanged
	}
	if cfg.perturbation != nil {
		if err := ValidatePerturbationArtifact(*cfg.perturbation, art); err != nil {
			return nil, err
		}
	}
	return art, ctx.Err()
}

type traceCleanupError struct {
	mu     sync.Mutex
	err    error
	module traceModule
}

func (e *traceCleanupError) Error() string { return e.err.Error() }
func (e *traceCleanupError) Unwrap() error { return e.err }
func (e *traceCleanupError) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.module == nil {
		return nil
	}
	if err := e.module.Close(); err != nil {
		return err
	}
	e.module = nil
	return nil
}

func loadCachedTrace(art *cudakernel.Artifact, dev *cuda.Device) (traceModule, error) {
	raw, err := art.Load(dev)
	if raw == 0 {
		return nil, err
	}
	loaded := &loadedTraceModule{device: dev, module: raw}
	// Reuse the CUDA module/context owner. Its Close tracks successful unload
	// separately from context restoration/release and retains unfinished work.
	owner, ownErr := cuda.OwnModule(dev, raw)
	if ownErr == nil {
		loaded.owner = owner
	} else {
		err = errors.Join(err, ownErr)
	}
	if err != nil {
		if closeErr := loaded.Close(); closeErr != nil {
			return nil, &traceCleanupError{err: errors.Join(err, closeErr), module: loaded}
		}
		return nil, err
	}
	return loaded, nil
}
