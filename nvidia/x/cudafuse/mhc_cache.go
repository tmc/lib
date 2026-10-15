package cudafuse

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ErrMHCCompilerCacheUnavailable reports an unsupported or incomplete compiler
// closure. It is a cache admission error, not an NVRTC compilation failure.
// CompileMHC remains available for uncached compilation.
var ErrMHCCompilerCacheUnavailable = errors.New("cudafuse: mhc compiler cache unavailable")

const mhcCacheRecipe = "cudafuse/mhc-nvrtc-cache/v1"

// CompileMHCCached compiles MHCSource through cache with explicit source, ABI,
// layout, numerical and compiler identities. The cached recipe selects C++17,
// PTX, no line information, no external headers, no contraction or flushing,
// and precise division and square root. It disables nvrtcx's internal cache.
// CompileMHC retains its original compilation recipe.
//
// Only Linux with NVRTC and its builtins already mapped is supported. tools.Tools
// must contain exactly two previously digested files named "nvrtc" and
// "nvrtc-builtins", matching those mappings. Other tools fields are not used:
// this consumer supplies its recipe/header identity and probes a driver only if
// already loaded. Missing or ambiguous mappings refuse caching. This function
// does not load vendor libraries or initialize a device. A caller can first use
// CompileMHC to establish the compiler mappings, then DigestTool on those files.
// Keep the libraries loaded and their files stable throughout this call.
//
// On error, completed descriptors remain in the returned array for inspection;
// they own no native modules and the error does not authorize their execution.
// A cleanup-bearing compiler error remains reachable with errors.As and must be
// closed before unloading the compiler. No evaluation receipt is created here.
func CompileMHCCached(ctx context.Context, c MHCConfig, major, minor int, cache *cudacache.NVRTCAdapter, tools cudacache.ToolProvenance) ([2]*cudakernel.Artifact, error) {
	var out [2]*cudakernel.Artifact
	tools.Tools = slices.Clone(tools.Tools)
	if ctx == nil {
		return out, errors.New("cudafuse: nil mhc cache context")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if _, err := MHCSource(c); err != nil {
		return out, err
	}
	if major <= 0 || minor < 0 {
		return out, errors.New("cudafuse: invalid mhc compute architecture")
	}
	if cache == nil {
		return out, fmt.Errorf("%w: nil adapter", ErrMHCCompilerCacheUnavailable)
	}
	if err := checkMHCCacheTools(tools.Tools); err != nil {
		return out, err
	}
	prov, nvMajor, nvMinor, driverName, err := mhcCacheEnvironment(tools.Tools)
	if err != nil {
		return out, err
	}
	requests, err := mhcCacheRequests(c, major, minor, nvMajor, nvMinor)
	if err != nil {
		return out, err
	}
	for i, req := range requests {
		id := req.identity(nvMajor, nvMinor, driverName, prov)
		out[i], err = compileMHCCachedEntry(ctx, cache, req, id)
		if err != nil {
			return out, fmt.Errorf("cudafuse: cached mhc %s: %w", req.Source.Name, err)
		}
	}
	return out, nil
}

type mhcCacheRequest struct {
	Source           FusedKernel
	Schema           cudakernel.Schema
	Target, InputKey string
	Layout           string
	Flags            []string
	Options          []nvrtcx.Option
	Major, Minor     int
}

func mhcCacheRequests(c MHCConfig, major, minor, nvMajor, nvMinor int) ([2]mhcCacheRequest, error) {
	var out [2]mhcCacheRequest
	sources, err := MHCSource(c)
	if err != nil {
		return out, err
	}
	if nvMajor != 12 && nvMajor != 13 || nvMinor < 0 {
		return out, fmt.Errorf("%w: unsupported nvrtc version %d.%d", ErrMHCCompilerCacheUnavailable, nvMajor, nvMinor)
	}
	for i, s := range sources {
		schema, err := mhcSchema(append(append([]string(nil), s.Inputs...), s.Outputs...))
		if err != nil {
			return out, err
		}
		opts := []nvrtcx.Option{nvrtcx.WithArch(major, minor), nvrtcx.WithStandard("c++17"), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithFlag("--ftz=false"), nvrtcx.WithFlag("--prec-div=true"), nvrtcx.WithFlag("--prec-sqrt=true"), nvrtcx.WithNoLineInfo(), nvrtcx.WithNoToolkitIncludes(), nvrtcx.WithName(s.Name + ".cu"), nvrtcx.WithOutput(nvrtcx.OutputPTX), nvrtcx.WithNoCache()}
		flags, err := nvrtcx.CompileFlags(opts...)
		if err != nil {
			return out, err
		}
		// Compile removes this flag before 12.9 while still disabling its own
		// cache. Bind effective flags, not an unsupported command-line option.
		if nvMajor == 12 && nvMinor < 9 {
			flags = slices.DeleteFunc(flags, func(f string) bool { return f == "--no-cache" })
		}
		r := mhcCacheRequest{Source: s, Schema: schema, Target: fmt.Sprintf("compute_%d%d", major, minor), Flags: flags, Options: opts, Major: major, Minor: minor,
			Layout: "fp32/contiguous/batch-major/logits[B,4,4]/x[B,W]/post[B,4]/residual,out[B,4,W]/mix[B,4,4]/disjoint/fp32-align4/abi-ptr8/count-s32"}
		r.InputKey = mhcCacheInputKey(c, r)
		out[i] = r
	}
	return out, nil
}

func mhcCacheInputKey(c MHCConfig, r mhcCacheRequest) string {
	// Fixed struct fields and exact float bits avoid map order, float text,
	// ambiguous concatenation and a caller's arbitrary workload label.
	return mhcCacheDigest(struct {
		Recipe                 string
		Batch, Width, Repeat   int
		EpsilonBits            uint32
		Source, Program, Entry string
		Inputs, Outputs        []string
		Schema                 cudakernel.Schema
		Layout, Target         string
		Flags                  []string
		Output                 nvrtcx.OutputKind
	}{mhcCacheRecipe, c.Batch, c.Width, c.Repeat, math.Float32bits(c.Epsilon), r.Source.Source, r.Source.Name + ".cu", r.Source.Name, r.Source.Inputs, r.Source.Outputs, r.Schema, r.Layout, r.Target, r.Flags, nvrtcx.OutputPTX})
}

func mhcCacheDigest(v any) string {
	// All callers pass finite fixed structs, strings or ABI integer metadata.
	b, _ := json.Marshal(v)
	h := sha256.Sum256(append([]byte(mhcCacheRecipe+"\x00"), b...))
	return hex.EncodeToString(h[:])
}

func (r mhcCacheRequest) identity(major, minor int, driverName string, prov cudacache.ToolProvenance) cudacache.CompilerIdentity {
	return cudacache.CompilerIdentity{NVRTCVersion: fmt.Sprintf("%d.%d", major, minor), DriverSONAME: driverName, TargetArch: r.Target, Flags: slices.Clone(r.Flags), OutputKind: nvrtcx.OutputPTX, Provenance: prov}
}

func checkMHCCacheTools(tools []cudacache.ToolDigest) error {
	if len(tools) != 2 {
		return fmt.Errorf("%w: nvrtc and nvrtc-builtins digests required", ErrMHCCompilerCacheUnavailable)
	}
	seen := make(map[string]bool)
	for _, d := range tools {
		if d.Name != "nvrtc" && d.Name != "nvrtc-builtins" || seen[d.Name] || !filepath.IsAbs(d.File.Path) {
			return fmt.Errorf("%w: invalid or duplicate compiler tool %q", ErrMHCCompilerCacheUnavailable, d.Name)
		}
		seen[d.Name] = true
		if err := cudacache.VerifyTool(d); err != nil {
			return fmt.Errorf("%w: %w", ErrMHCCompilerCacheUnavailable, err)
		}
	}
	return nil
}

func checkMHCMappedTools(tools []cudacache.ToolDigest) error {
	for _, d := range tools {
		var name string
		switch d.Name {
		case "nvrtc":
			name = nvrtc.Path()
		case "nvrtc-builtins":
			name = "libnvrtc-builtins.so"
		case "driver":
			name = driver.Path()
		default:
			return fmt.Errorf("%w: unsupported mapped tool %q", ErrMHCCompilerCacheUnavailable, d.Name)
		}
		mapped, err := cudacache.LoadedLibraryPath(name)
		if err != nil {
			return fmt.Errorf("%w: %w", ErrMHCCompilerCacheUnavailable, err)
		}
		declared, err := filepath.EvalSymlinks(d.File.Path)
		if err != nil || mapped != declared {
			return fmt.Errorf("%w: mapped %s differs from its declared file", ErrMHCCompilerCacheUnavailable, d.Name)
		}
		if err := cudacache.VerifyTool(d); err != nil {
			return fmt.Errorf("%w: %w", ErrMHCCompilerCacheUnavailable, err)
		}
	}
	return nil
}

func mhcCacheEnvironment(tools []cudacache.ToolDigest) (cudacache.ToolProvenance, int, int, string, error) {
	if runtime.GOOS != "linux" || !nvrtc.Loaded() {
		return cudacache.ToolProvenance{}, 0, 0, "", fmt.Errorf("%w: mapped Linux nvrtc and builtins required", ErrMHCCompilerCacheUnavailable)
	}
	if err := checkMHCMappedTools(tools); err != nil {
		return cudacache.ToolProvenance{}, 0, 0, "", err
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		return cudacache.ToolProvenance{}, 0, 0, "", err
	}
	prov, err := cudacache.ProbeToolchain()
	if err != nil {
		return prov, 0, 0, "", fmt.Errorf("%w: %w", ErrMHCCompilerCacheUnavailable, err)
	}
	if err := checkMHCMappedTools(tools); err != nil {
		return prov, 0, 0, "", err
	}
	// ProbeToolchain records the already loaded main library and optional
	// driver. Builtins are a separate, explicitly verified compiler input.
	for _, d := range tools {
		if d.Name == "nvrtc-builtins" {
			prov.Tools = append(prov.Tools, d)
		}
	}
	prov.Recipe = mhcCacheRecipe
	prov.HeadersSHA256 = mhcCacheDigest("no-external-headers")
	driverName := "libcuda.so.1"
	if driver.Loaded() {
		driverName = driver.Path()
	}
	return prov, int(major), int(minor), driverName, nil
}

type mhcCacheBuild struct {
	Recipe, CompilerKey, Provenance string
}

func mhcCacheDependencies(r mhcCacheRequest, id cudacache.CompilerIdentity) []cudakernel.Dependency {
	return []cudakernel.Dependency{
		{Path: "mhc/source", Digest: mhcCacheDigest(r.Source.Source), Tracked: true},
		{Path: "mhc/request", Digest: r.InputKey, Tracked: true},
		{Path: "mhc/abi", Digest: mhcCacheDigest(r.Schema), Tracked: true},
		{Path: "mhc/layout", Digest: mhcCacheDigest(r.Layout), Tracked: true},
		{Path: "mhc/tools", Digest: id.Provenance.Fingerprint(), Tracked: true},
	}
}

func checkMHCCacheArtifact(r mhcCacheRequest, id cudacache.CompilerIdentity, art *cudakernel.Artifact) error {
	if err := art.Validate(); err != nil {
		return fmt.Errorf("%w: %w", cudacache.ErrCorruptArtifact, err)
	}
	var build mhcCacheBuild
	if err := json.Unmarshal([]byte(art.Toolchain), &build); err != nil {
		return fmt.Errorf("%w: invalid mhc compiler receipt", cudacache.ErrCorruptArtifact)
	}
	canonical, _ := json.Marshal(build)
	key, err := hex.DecodeString(build.CompilerKey)
	if art.EntryPoint != r.Source.Name || art.Target != r.Target || art.Kind != cudakernel.KindPTX || !reflect.DeepEqual(art.Schema, r.Schema) || !slices.Equal(art.CompilerOpts, r.Flags) || !reflect.DeepEqual(art.Dependencies, mhcCacheDependencies(r, id)) || build.Recipe != mhcCacheRecipe || build.Provenance != id.Provenance.Fingerprint() || err != nil || len(key) != sha256.Size || string(canonical) != art.Toolchain {
		return fmt.Errorf("%w: mhc source, abi, layout, flags or tool binding differs", cudacache.ErrCorruptArtifact)
	}
	return nil
}

func compileMHCCachedEntry(ctx context.Context, cache *cudacache.NVRTCAdapter, r mhcCacheRequest, id cudacache.CompilerIdentity) (*cudakernel.Artifact, error) {
	var completed *cudakernel.Artifact
	var compileErr error
	art, err := cache.GetOrCompile(ctx, r.InputKey, id, func(ctx context.Context) (*cudakernel.Artifact, cudacache.ToolProvenance, error) {
		var key string
		produced, key, err := compileMHCArtifact(ctx, r.Source, r.Major, r.Minor, r.Options...)
		if err != nil {
			compileErr = err
			return nil, cudacache.ToolProvenance{}, err
		}
		if len(produced.Dependencies) != 0 {
			return nil, cudacache.ToolProvenance{}, fmt.Errorf("%w: external mhc dependencies", ErrMHCCompilerCacheUnavailable)
		}
		actual, major, minor, driverName, err := mhcCacheEnvironment(id.Provenance.Tools)
		if err != nil {
			return nil, cudacache.ToolProvenance{}, err
		}
		if actual.Fingerprint() != id.Provenance.Fingerprint() || fmt.Sprintf("%d.%d", major, minor) != id.NVRTCVersion || driverName != id.DriverSONAME {
			return nil, cudacache.ToolProvenance{}, fmt.Errorf("%w: compiler environment changed during compilation", ErrMHCCompilerCacheUnavailable)
		}
		produced.CompilerOpts = slices.Clone(r.Flags)
		produced.Dependencies = mhcCacheDependencies(r, id)
		receipt, _ := json.Marshal(mhcCacheBuild{mhcCacheRecipe, key, id.Provenance.Fingerprint()})
		produced.Toolchain = string(receipt)
		if err := checkMHCCacheArtifact(r, id, produced); err != nil {
			return nil, cudacache.ToolProvenance{}, err
		}
		completed = produced
		return produced, actual, nil
	})
	if err != nil {
		// The adapter's post-verification error can replace its callback's
		// error. Keep any native program cleanup owner reachable either way.
		return completed, errors.Join(err, compileErr)
	}
	if err := checkMHCCacheArtifact(r, id, art); err != nil {
		return nil, err
	}
	return art, nil
}
