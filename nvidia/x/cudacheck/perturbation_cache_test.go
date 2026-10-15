package cudacheck

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// This fixture declares synthetic metadata and image bytes. It does not model
// compiler normalization, authenticate output or execute a device instruction.
func cachedFixtureModule(t *testing.T, jit cudakernel.JITOptions, flags []string) *nvrtcx.CompiledModule {
	t.Helper()
	cm := newKnownFakeCompiled(t, jit, flags)
	for _, item := range []struct {
		name  string
		value any
	}{{"ptx", "host descriptor fixture"}, {"output", nvrtcx.OutputPTX}} {
		field := reflect.ValueOf(cm).Elem().FieldByName(item.name)
		value := reflect.ValueOf(item.value)
		if !field.IsValid() || !field.CanAddr() || field.Type() != value.Type() {
			t.Fatal("fixture field changed", item.name)
		}
		reflect.NewAt(field.Type(), field.Addr().UnsafePointer()).Elem().Set(value)
	}
	return cm
}

type cachedTraceFixture struct {
	dir                     string
	adapter                 *cudacache.NVRTCAdapter
	compiles, loads, probes int
	hash                    uint64
	mutate                  func(*nvrtcx.CompiledModule)
	afterCompile            func()
	load                    func() (traceModule, error)
	changePost              bool
}

func newCachedTraceFixture(t *testing.T) *cachedTraceFixture {
	t.Helper()
	dir := t.TempDir()
	c, err := cudacache.New(cudacache.Config{RootDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	return &cachedTraceFixture{dir: dir, adapter: cudacache.NewNVRTCAdapter(c)}
}

func (f *cachedTraceFixture) options(t *testing.T) []CompareOption {
	return []CompareOption{WithArtifactCache(f.adapter, "compute_80"), func(c *CompareConfig) {
		c.cacheEnvironment = func(target string, flags []string, jit cudakernel.JITOptions) (cudacache.CompilerIdentity, error) {
			f.probes++
			driver := "declared-host-fixture"
			if f.changePost && f.compiles > 0 {
				driver += "-changed"
			}
			return cudacache.CompilerIdentity{NVRTCVersion: "13.0", DriverSONAME: driver, TargetArch: target, Flags: append([]string(nil), flags...), OutputKind: nvrtcx.OutputPTX, JITOptions: jit, Provenance: cudacache.ToolProvenance{Recipe: "synthetic-host-fixture"}}, nil
		}
		c.compileKernel = func(_ string, opts []nvrtcx.Option, jit cudakernel.JITOptions) (*nvrtcx.CompiledModule, error) {
			f.compiles++
			flags, err := nvrtcx.CompileFlags(opts...)
			if err != nil {
				return nil, err
			}
			cm := cachedFixtureModule(t, jit, flags)
			if f.mutate != nil {
				f.mutate(cm)
			}
			if f.afterCompile != nil {
				f.afterCompile()
			}
			return cm, nil
		}
		c.loadArtifact = func(art *cudakernel.Artifact, _ *cuda.Device) (traceModule, error) {
			f.loads++
			if err := art.Validate(); err != nil {
				return nil, err
			}
			if f.load != nil {
				return f.load()
			}
			return &fakeTraceModule{hash: f.hash, stateHash: FNVOffsetBasis}, nil
		}
	}}
}

func TestJITCachedPublicPerturbationConsumer(t *testing.T) {
	g, inputs := fakeGraphWithLength(2)
	acc, err := EvaluateCPUGraph(g, inputs, PolicyRawBits, true)
	if err != nil {
		t.Fatal(err)
	}
	f := newCachedTraceFixture(t)
	f.hash = acc.Hash()
	options := append(f.options(t), WithInputs(inputs))
	perts := []Perturbation{{Name: "baseline", JITOptLevel: 4}, {Name: "zero", JITOptLevel: 0}, {Name: "vector", JITOptLevel: 0, ExtraDeviceVectorize: true}}
	for pass := 0; pass < 2; pass++ {
		report, err := RunPerturbations(context.Background(), g, perts, options...)
		if err != nil || len(report.Results) != 3 {
			t.Fatal(report, err)
		}
		for _, row := range report.Results {
			if row.Status != StatusAgreed || row.CompileErr != nil || row.RunErr != nil || row.ArtifactErr != nil {
				t.Fatal(row)
			}
		}
		if f.compiles != 3 || f.loads != (pass+1)*3 {
			t.Fatal("cache dispatch or identities lost", f.compiles, f.loads)
		}
		// A fresh adapter forces the second pass through disk manifests.
		c, err := cudacache.New(cudacache.Config{RootDir: f.dir})
		if err != nil {
			t.Fatal(err)
		}
		f.adapter = cudacache.NewNVRTCAdapter(c)
		options = append(f.options(t), WithInputs(inputs))
	}
}

func TestJITCachedCanonicalPositiveAndUnset(t *testing.T) {
	g, inputs := fakeGraphWithLength(1)
	acc, err := EvaluateCPUGraph(g, inputs, PolicyRawBits, true)
	if err != nil {
		t.Fatal(err)
	}
	f := newCachedTraceFixture(t)
	f.hash = acc.Hash()
	for i, jit := range []cudakernel.JITOptions{{}, {OptLevelSet: true}, {OptLevel: 4}, {OptLevel: 4, OptLevelSet: true}} {
		opts := append(f.options(t), WithInputs(inputs), WithJITOpts(jit))
		res, err := Compare(context.Background(), g, opts...)
		if err != nil || res.Status != StatusAgreed || f.compiles != min(i+1, 3) || f.loads != i+1 {
			t.Fatal(i, res, err, f.compiles, f.loads)
		}
		fresh, err := cudacache.New(cudacache.Config{RootDir: f.dir})
		if err != nil {
			t.Fatal(err)
		}
		f.adapter = cudacache.NewNVRTCAdapter(fresh)
	}
}

func TestJITCachedArtifactBindingBeforeLoad(t *testing.T) {
	g, inputs := fakeGraphWithLength(1)
	f := newCachedTraceFixture(t)
	acc, err := EvaluateCPUGraph(g, inputs, PolicyRawBits, true)
	if err != nil {
		t.Fatal(err)
	}
	f.hash = acc.Hash()
	opts := append(f.options(t), WithInputs(inputs))
	res, err := Compare(context.Background(), g, opts...)
	if err != nil || res.Status != StatusAgreed {
		t.Fatal(res, err)
	}
	files, err := filepath.Glob(filepath.Join(f.dir, "artifacts", "*.artifact"))
	if err != nil || len(files) != 1 {
		t.Fatal(files, err)
	}
	data, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	var art cudakernel.Artifact
	if err := json.Unmarshal(data, &art); err != nil {
		t.Fatal(err)
	}
	art.Schema.Params[0].Name = "wrong_abi_name"
	data, err = json.Marshal(art)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(files[0], data, 0600); err != nil {
		t.Fatal(err)
	}
	fresh, err := cudacache.New(cudacache.Config{RootDir: f.dir})
	if err != nil {
		t.Fatal(err)
	}
	f.adapter = cudacache.NewNVRTCAdapter(fresh)
	opts = append(f.options(t), WithInputs(inputs))
	res, err = Compare(context.Background(), g, opts...)
	if err != nil || res.Status != StatusUnmeasured || res.Stage != StageArtifactValidation || !errors.Is(res.Err, ErrPerturbationArtifactMismatch) || f.loads != 1 || f.compiles != 1 {
		t.Fatal(res, err, f.loads, f.compiles)
	}
}

func TestJITCachedRefusalBeforeLoad(t *testing.T) {
	for _, stage := range []string{"unset", "wrong flags", "unknown", "post identity", "cancel"} {
		t.Run(stage, func(t *testing.T) {
			g, inputs := fakeGraphWithLength(1)
			f := newCachedTraceFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if stage == "unset" {
				f.mutate = func(cm *nvrtcx.CompiledModule) { cm.SetJITOptions(cudakernel.JITOptions{}) }
			}
			if stage == "wrong flags" {
				f.mutate = func(cm *nvrtcx.CompiledModule) { *cm = *cachedFixtureModule(t, cm.JITOptions(), nil) }
			}
			if stage == "unknown" {
				f.mutate = func(cm *nvrtcx.CompiledModule) { *cm = *newFakeCompiled(cm.JITOptions()) }
			}
			if stage == "post identity" {
				f.changePost = true
			}
			if stage == "cancel" {
				f.afterCompile = cancel
			}
			options := append(f.options(t), WithInputs(inputs))
			report, err := RunPerturbations(ctx, g, []Perturbation{{Name: "zero", JITOptLevel: 0}}, options...)
			if err != nil || len(report.Results) != 1 || f.compiles != 1 || f.loads != 0 {
				t.Fatal(report, err, f.compiles, f.loads)
			}
			row := report.Results[0]
			if row.Status != StatusUnmeasured {
				t.Fatal(row)
			}
			switch stage {
			case "cancel":
				if !errors.Is(row.CompileErr, context.Canceled) {
					t.Fatal(row)
				}
			case "post identity":
				if !errors.Is(row.CompileErr, cudacache.ErrToolIdentityChanged) {
					t.Fatal(row)
				}
			default:
				if !errors.Is(row.ArtifactErr, ErrPerturbationArtifactMismatch) {
					t.Fatal(row)
				}
			}
			entries, _ := filepath.Glob(filepath.Join(f.dir, "manifests", "*.manifest"))
			if len(entries) != 0 {
				t.Fatal("refused result cached", entries)
			}
		})
	}
}

func TestJITCachedRejectsOpaqueOptionsWithoutReplay(t *testing.T) {
	g, inputs := fakeGraphWithLength(1)
	f := newCachedTraceFixture(t)
	calls := 0
	// Option's parameter is private. Reflection supplies a test-only callback
	// with an observable call count; production must never evaluate it here.
	callback := reflect.MakeFunc(reflect.TypeFor[nvrtcx.Option](), func([]reflect.Value) []reflect.Value { calls++; return nil }).Interface().(nvrtcx.Option)
	opts := append(f.options(t), WithInputs(inputs), WithCompilerOptions(callback))
	res, err := Compare(context.Background(), g, opts...)
	if err != nil || !errors.Is(res.Err, ErrArtifactCacheUnsupported) || calls != 0 || f.probes != 0 || f.compiles != 0 || f.loads != 0 {
		t.Fatal(res, err, calls, f.probes, f.compiles, f.loads)
	}
	opts = append(f.options(t), WithInputs(inputs), WithKernelSource("custom", "k"))
	res, err = Compare(context.Background(), g, opts...)
	if err != nil || !errors.Is(res.Err, ErrArtifactCacheUnsupported) || f.probes != 0 {
		t.Fatal(res, err, f.probes)
	}
}

func TestJITCacheVersionNormalization(t *testing.T) {
	for _, tc := range []struct {
		version string
		count   int
	}{{"12.8", 1}, {"12.9", 2}, {"13.0", 2}} {
		source := []string{"--std=c++17", "--no-cache"}
		flags, err := traceCacheFlags(source, tc.version)
		if err != nil || len(flags) != tc.count || len(source) != 2 || source[1] != "--no-cache" {
			t.Fatal(tc.version, flags, source, err)
		}
	}
	if _, err := traceCacheFlags(nil, "nvrtc-default"); !errors.Is(err, ErrArtifactCacheUnsupported) {
		t.Fatal(err)
	}
}

type closeFailTrace struct {
	fakeTraceModule
	calls   int
	failure error
}

type cancelTrace struct {
	closeFailTrace
	cancel context.CancelFunc
}

func (x *cancelTrace) Execute(context.Context, string, *CompareConfig, *cudair.Snapshot, uint32, bool, bool) (deviceTrace, error) {
	x.cancel()
	return deviceTrace{hash: x.hash}, nil
}

func TestJITCachedExecutionCancellation(t *testing.T) {
	g, inputs := fakeGraphWithLength(1)
	acc, err := EvaluateCPUGraph(g, inputs, PolicyRawBits, true)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	x := &cancelTrace{closeFailTrace: closeFailTrace{fakeTraceModule: fakeTraceModule{hash: acc.Hash(), stateHash: FNVOffsetBasis}}, cancel: cancel}
	f := newCachedTraceFixture(t)
	f.load = func() (traceModule, error) { return x, nil }
	res, err := Compare(ctx, g, append(f.options(t), WithInputs(inputs))...)
	if err != nil || res.Status != StatusUnmeasured || !errors.Is(res.Err, context.Canceled) || x.calls != 1 {
		t.Fatal(res, err, x.calls)
	}
}

func (x *closeFailTrace) Close() error { x.calls++; return x.failure }

func TestJITCachedCleanupCustody(t *testing.T) {
	for _, loadFailure := range []bool{false, true} {
		t.Run(map[bool]string{false: "close", true: "partial load"}[loadFailure], func(t *testing.T) {
			g, inputs := fakeGraphWithLength(1)
			acc, err := EvaluateCPUGraph(g, inputs, PolicyRawBits, true)
			if err != nil {
				t.Fatal(err)
			}
			failure := errors.New("fixture close failure")
			x := &closeFailTrace{fakeTraceModule: fakeTraceModule{hash: acc.Hash(), stateHash: FNVOffsetBasis}, failure: failure}
			f := newCachedTraceFixture(t)
			f.load = func() (traceModule, error) {
				if loadFailure {
					return x, errors.New("fixture partial load")
				}
				return x, nil
			}
			opts := append(f.options(t), WithInputs(inputs))
			res, err := Compare(context.Background(), g, opts...)
			var owner io.Closer
			if err != nil || res.Status != StatusUnmeasured || !errors.Is(res.Err, failure) || !errors.As(res.Err, &owner) || x.calls != 1 {
				t.Fatal(res, err, x.calls)
			}
			if err := owner.Close(); !errors.Is(err, failure) || x.calls != 2 {
				t.Fatal(err, x.calls)
			}
			x.failure = nil
			if err := owner.Close(); err != nil || x.calls != 3 {
				t.Fatal(err, x.calls)
			}
			if err := owner.Close(); err != nil || x.calls != 3 {
				t.Fatal("double cleanup", err, x.calls)
			}
		})
	}
}

func TestJITCachedCompileCleanupError(t *testing.T) {
	g, inputs := fakeGraphWithLength(1)
	f := newCachedTraceFixture(t)
	x := &closeFailTrace{}
	pending := &traceCleanupError{err: errors.New("synthetic compiler cleanup error"), module: x}
	opts := append(f.options(t), WithInputs(inputs), func(c *CompareConfig) {
		c.compileKernel = func(string, []nvrtcx.Option, cudakernel.JITOptions) (*nvrtcx.CompiledModule, error) {
			f.compiles++
			return nil, pending
		}
	})
	res, err := Compare(context.Background(), g, opts...)
	var owner io.Closer
	if err != nil || res.Status != StatusUnmeasured || res.Stage != StageCompile || !errors.As(res.Err, &owner) || f.compiles != 1 || f.loads != 0 || x.calls != 0 {
		t.Fatal(res, err, f.compiles, f.loads, x.calls)
	}
	if err := owner.Close(); err != nil || x.calls != 1 {
		t.Fatal(err, x.calls)
	}
	if err := owner.Close(); err != nil || x.calls != 1 {
		t.Fatal("double cleanup", err, x.calls)
	}
}
