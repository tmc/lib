//go:build native && linux

package cudasearch

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// This fixture tests a declared scalar-to-float4 affine rewrite, not automatic
// optimization discovery. All tails are checked. No speedup is required.
const memoryNativeSource = `
extern "C" __global__ void scalar(const float *x, float *y, unsigned n) {
 unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < n) y[i] = 2.0f*x[i] + 1.0f;
}
extern "C" __global__ void vector(const float *x, float *y, unsigned n) {
 unsigned i = (blockIdx.x * blockDim.x + threadIdx.x)*4;
 if (i+3 < n) {
  float4 v = reinterpret_cast<const float4 *>(x)[i/4];
  reinterpret_cast<float4 *>(y)[i/4] = make_float4(2.0f*v.x+1.0f,
   2.0f*v.y+1.0f, 2.0f*v.z+1.0f, 2.0f*v.w+1.0f);
 } else {
  for (unsigned j=i; j<n; j++) y[j] = 2.0f*x[j]+1.0f;
 }
}
extern "C" __global__ void wrong(const float *x, float *y, unsigned n) {
 unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < n) y[i] = 2.0f*x[i] + 2.0f;
}
`

func memoryNativeDigest(b []byte) string { return fmt.Sprintf("%x", sha256.Sum256(b)) }

func memoryNativeFP32(x []float32) []byte {
	b := make([]byte, 4*len(x))
	for i, v := range x {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}

func memoryNativeWrite(t *testing.T, dir, name string, value any) {
	t.Helper()
	var data []byte
	if b, ok := value.([]byte); ok {
		data = b
	} else {
		var err error
		data, err = json.MarshalIndent(value, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
		t.Fatal(err)
	}
}

func memoryNativeSourceManifest(t *testing.T) []byte {
	t.Helper()
	path, expected := os.Getenv("CUDASEARCH_MEMORY_SOURCE_MANIFEST"), os.Getenv("CUDASEARCH_MEMORY_MANIFEST_SHA256")
	if !filepath.IsAbs(path) || len(expected) != 64 {
		t.Fatal("set source manifest path and exact SHA256")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, (2<<20)+1))
	if err != nil || len(b) > 2<<20 || memoryNativeDigest(b) != expected {
		t.Fatal("source manifest bytes not sealed")
	}
	var manifest struct {
		BaseRevision string `json:"base_revision"`
		Files        []struct {
			Path   string `json:"path"`
			SHA256 string `json:"sha256"`
		} `json:"files"`
	}
	if err := json.Unmarshal(b, &manifest); err != nil || len(manifest.BaseRevision) != 40 || len(manifest.Files) == 0 || len(manifest.Files) > 10000 {
		t.Fatal("invalid source manifest")
	}
	seen := make(map[string]bool)
	for _, file := range manifest.Files {
		if seen[file.Path] || !strings.HasPrefix(file.Path, "nvidia/") || filepath.Clean(file.Path) != file.Path || strings.Contains(file.Path, "..") {
			t.Fatal("invalid manifest source path")
		}
		seen[file.Path] = true
		data, err := os.ReadFile(filepath.Join("..", file.Path))
		if err != nil || memoryNativeDigest(data) != file.SHA256 {
			t.Fatalf("deployed source differs: %s (%v)", file.Path, err)
		}
	}
	for _, required := range []string{"nvidia/go.mod", "nvidia/go.sum", "nvidia/x/cudasearch/search_memory_native_test.go", "nvidia/x/cudasearch/bottleneck_profile.go", "nvidia/x/cudatrial/trajectory.go", "nvidia/x/cudacache/trajectory.go"} {
		if !seen[required] {
			t.Fatalf("missing source binding: %s", required)
		}
	}
	return b
}

type memoryNativeExecutable struct {
	dev                                                        *cuda.Device
	stream                                                     *cuda.Stream
	start, stop                                                *cuda.Event
	x, y                                                       cudamem.Slice[float32]
	function                                                   driver.Function
	entry, id, target, input, artifact, compiler, policy, uuid string
	reference                                                  []float32
	grid                                                       uint32
	verified                                                   bool
	outputs                                                    []float32
	timed                                                      int
}

func newMemoryNativeExecutable(dev *cuda.Device, module driver.Module, entry, target, artifact, compiler, policy string, input, reference []float32) (x *memoryNativeExecutable, err error) {
	x = &memoryNativeExecutable{dev: dev, entry: entry, target: target, artifact: artifact, compiler: compiler, policy: policy, input: memoryNativeDigest(memoryNativeFP32(input)), reference: append([]float32(nil), reference...)}
	defer func() {
		if err != nil {
			err = errors.Join(err, x.Close())
		}
	}()
	n := len(input)
	if n <= 0 || n > 1<<20 || n != len(reference) {
		return x, fmt.Errorf("invalid native memory fixture extent")
	}
	width := 1
	if entry == "vector" {
		width = 4
	}
	x.grid = uint32((n + width*256 - 1) / (width * 256))
	x.id = memoryNativeDigest([]byte(fmt.Sprintf("%s/%s/%s/%s/%d/256/%d", artifact, compiler, target, entry, x.grid, n)))
	err = dev.WithContext(func() error {
		var err error
		x.function, err = module.Function(entry)
		if err != nil {
			return err
		}
		uuid, err := dev.Raw().Uuid()
		if err == nil {
			x.uuid = fmt.Sprintf("%x", uuid.Bytes)
		}
		return err
	})
	if err != nil {
		return x, err
	}
	x.stream, err = dev.NewStreamWithFlags(driver.StreamNonBlocking)
	if err != nil {
		return x, err
	}
	x.start, err = dev.NewEvent()
	if err != nil {
		return x, err
	}
	x.stop, err = dev.NewEvent()
	if err != nil {
		return x, err
	}
	x.x, err = cudamem.Alloc[float32](dev, n)
	if err != nil {
		return x, err
	}
	x.y, err = cudamem.Alloc[float32](dev, n)
	if err != nil {
		return x, err
	}
	return x, x.x.CopyFromHost(input)
}

func (x *memoryNativeExecutable) Identity() string { return x.id }

func (x *memoryNativeExecutable) prepare() error {
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		return err
	}
	if err := x.stream.Synchronize(nil); err != nil {
		return err
	}
	if err := x.y.MemsetAsync(0xff, x.stream); err != nil {
		return err
	}
	return x.stream.Synchronize(nil)
}

func (x *memoryNativeExecutable) launch() error {
	return x.dev.WithContext(func() error {
		return cudalaunch.Kernel(x.function).Grid(x.grid, 1, 1).Block(256, 1, 1).On(x.stream).Launch(x.x, x.y, uint32(x.x.Len()))
	})
}

func (x *memoryNativeExecutable) Outputs(ctx context.Context) (map[string][]float32, error) {
	x.verified = false
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := x.prepare(); err != nil {
		return nil, err
	}
	if err := x.launch(); err != nil {
		return nil, err
	}
	if err := x.stream.Synchronize(nil); err != nil {
		return nil, err
	}
	x.outputs = make([]float32, len(x.reference))
	if err := x.y.CopyToHost(x.outputs); err != nil {
		return nil, err
	}
	p := cudatest.CUDAOutputPolicy{Target: x.target, DType: "fp32", MathMode: "fmad=false; exact binary affine", SourceDigest: memoryNativeDigest([]byte(memoryNativeSource)), InputDigest: x.input, Elements: len(x.reference), NaN: "reject", Infinity: "reject", Zero: "bits"}
	x.verified = cudatest.CompareCUDAOutputFP32(x.outputs, x.reference, p) == nil
	return map[string][]float32{"out": append([]float32(nil), x.outputs...)}, nil
}

func (x *memoryNativeExecutable) Run(ctx context.Context) (r cudatrial.Receipt, err error) {
	if err := ctx.Err(); err != nil {
		return r, err
	}
	if !x.verified {
		return r, cudatrial.ErrUnqualifiedOutcome
	}
	if err := x.prepare(); err != nil {
		return r, err
	}
	if err := x.start.Record(x.stream); err != nil {
		return r, err
	}
	// Always join completion before exposing buffers or cleanup, including a
	// launch failure. Context cancellation never substitutes for device drain.
	err = x.launch()
	err = errors.Join(err, x.stop.Record(x.stream), x.stream.Synchronize(nil))
	if err != nil {
		return r, err
	}
	d, err := x.stop.ElapsedSince(x.start)
	if err != nil || d <= 0 {
		return r, errors.Join(err, fmt.Errorf("unresolved CUDA event interval"))
	}
	x.timed++
	r.Complete = true
	r.Provenance = map[string]string{"timer": "cuda_event", "timing_scope": "single kernel start/stop on private stream; includes event boundary", "input_digest": x.input, "artifact_digest": x.artifact, "compiler_key": x.compiler, "policy_id": x.policy, "verified_correct": "true", "device_uuid": x.uuid, "entry": x.entry, "grid": fmt.Sprint(x.grid), "block": "256", "cache": "warm; no eviction"}
	return r, r.RecordTiming(cudatrial.TimingSummedKernel, d)
}

func (x *memoryNativeExecutable) Close() error {
	if x.stream != nil {
		if err := x.stream.Synchronize(nil); err != nil {
			return err
		}
	}
	for _, e := range []**cuda.Event{&x.start, &x.stop} {
		if *e != nil {
			if err := (*e).Close(); err != nil {
				return err
			}
			*e = nil
		}
	}
	if x.x.Len() > 0 {
		if err := x.x.Free(); err != nil {
			return err
		}
		x.x = cudamem.Slice[float32]{}
	}
	if x.y.Len() > 0 {
		if err := x.y.Free(); err != nil {
			return err
		}
		x.y = cudamem.Slice[float32]{}
	}
	if x.stream != nil {
		if err := x.stream.Close(); err != nil {
			return err
		}
		x.stream = nil
	}
	return nil
}

type memoryNativeCost struct{}

// Retain failed owners for process lifetime. A failed drain must keep the
// module and device alive even after the testing cleanup callback returns.
var memoryNativeRetainedOwners []*memoryFixtureOwner

func (memoryNativeCost) NodeCost(n egraph.ENode, _ []float64) float64 {
	if n.Op == "scalar" {
		return 10
	}
	return 1
}

func TestNativeSearchMemoryProducerConsumer(t *testing.T) {
	if os.Getenv("CUDASEARCH_MEMORY_NATIVE") != "1" {
		t.Skip("PendingHardware: set CUDASEARCH_MEMORY_NATIVE=1 under gpu-lock")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	sourceManifest := memoryNativeSourceManifest(t)
	root := os.Getenv("CUDASEARCH_MEMORY_RECEIPTS")
	if root == "" || !filepath.IsAbs(root) {
		t.Fatal("set absolute CUDASEARCH_MEMORY_RECEIPTS to a persistent task directory")
	}
	if err := os.Mkdir(root, 0700); err != nil {
		t.Fatal("receipt directory must be new: ", err)
	}
	memoryNativeWrite(t, root, "source-manifest.json", sourceManifest)
	owner := new(memoryFixtureOwner)
	t.Cleanup(func() {
		if err := owner.Close(); err != nil {
			memoryNativeRetainedOwners = append(memoryNativeRetainedOwners, owner)
			t.Error("native fixture retained until process exit: ", err)
		}
	})
	dev := cudatest.RequireDevice(t)
	owner.add(memoryDeviceStage, "device", dev.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	target := fmt.Sprintf("sm_%d%d", major, minor)
	opts := []nvrtcx.Option{nvrtcx.WithArch(major, minor), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithNoCache()}
	flags, err := nvrtcx.CompileFlags(opts...)
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := nvrtcx.CompileContext(ctx, memoryNativeSource, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if compiled.PTX() == "" {
		t.Fatal("native producer requires retained PTX")
	}
	image, artifact, selected, err := memoryNativeSelectPTX(compiled)
	if err != nil {
		t.Fatal(err)
	}
	module, err := compiled.LoadOutput(dev, selected)
	if err != nil {
		var failed io.Closer
		if errors.As(err, &failed) {
			owner.add(memoryModuleStage, "failed module load", failed.Close)
		}
		t.Fatal(err)
	}
	owner.add(memoryModuleStage, "module", func() error { return dev.WithContext(module.Close) })
	tools, err := cudacache.ProbeToolchain()
	if err != nil || len(tools.Tools) != 2 {
		t.Fatalf("exact tool provenance unavailable: %+v %v", tools, err)
	}
	nvMajor, nvMinor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	// Artifact identity is the exact selected PTX input to LoadOutput.
	compiler := memoryNativeDigest([]byte(fmt.Sprintf("%s/%v/%s", tools.Fingerprint(), flags, target)))
	policy := memoryNativeDigest([]byte("fp32 exact finite affine; fmad=false; signed-zero bits; all tails; no tolerance"))
	domain := memoryNativeDigest([]byte("lane46 scalar-to-float4/fp32/contiguous/exact/v1/" + target))
	memoryNativeWrite(t, root, "fixture.cu", []byte(memoryNativeSource))
	memoryNativeWrite(t, root, "fixture.ptx", image)
	memoryNativeWrite(t, root, "compiler.log", []byte(compiled.Log()))
	memoryNativeWrite(t, root, "toolchain.json", map[string]any{"tools": tools, "nvrtc_version": []int32{nvMajor, nvMinor}, "flags": flags, "target": target, "artifact_digest": artifact, "loaded_output": "ptx", "loaded_output_bytes": len(image), "compiler_key": compiler, "policy_id": policy, "domain": domain})
	cache, err := cudacache.New(cudacache.Config{RootDir: filepath.Join(root, "cache")})
	if err != nil {
		t.Fatal(err)
	}
	memory, err := cudatrial.NewTrajectoryMemory(16)
	if err != nil {
		t.Fatal(err)
	}
	var catalog cudacache.StrategyKnowledgeBase
	var previous *cudatrial.TrajectoryReceipt
	for task, count := range []int{262147, 65537} {
		t.Run(fmt.Sprintf("task%d", task), func(t *testing.T) {
			dir := filepath.Join(root, fmt.Sprintf("task%d", task))
			if err := os.Mkdir(dir, 0700); err != nil {
				t.Fatal(err)
			}
			input, want := make([]float32, count), make([]float32, count)
			for i := range input {
				input[i] = float32(i%256-128) / 16
				want[i] = float32(i%256-128)/8 + 1
			}
			inputID := memoryNativeDigest(memoryNativeFP32(input))
			binding := WorkloadBindings{Name: "lane46-fp32-affine", InputNames: []string{"x"}, InputIdentities: map[string]string{"x": inputID}, NumericalPolicy: NumericalPolicyExact, Objective: ObjectiveMinLatency, Layout: "contiguous"}
			base, err := newMemoryNativeExecutable(dev, module, "scalar", target, artifact, compiler, policy, input, want)
			if base != nil {
				owner.add(memoryExecutableStage, fmt.Sprintf("task%d baseline", task), base.Close)
			}
			if err != nil {
				t.Fatal(err)
			}
			memoryNativeWrite(t, dir, "input.fp32", memoryNativeFP32(input))
			memoryNativeWrite(t, dir, "reference.fp32", memoryNativeFP32(want))
			if _, err := base.Outputs(ctx); err != nil || !base.verified {
				t.Fatalf("baseline correctness: %v", err)
			}
			profile, raw, err := CollectBottleneckProfile(ctx, dev, BottleneckProfile{WorkloadKey: binding.Key(), InputDigest: inputID, ArtifactDigest: artifact, CompilerKey: compiler}, cuptibench.Workload{Streams: []*cuda.Stream{base.stream}, Prepare: base.prepare, Run: func(*cuptibench.Scope) error { return base.launch() }, Synchronize: func() error { return base.stream.Synchronize(nil) }, Preparation: "input immutable; output poisoned each pass; warm cache; scalar FP32 affine"})
			var failed io.Closer
			if errors.As(err, &failed) {
				owner.add(memoryCollectorStage, fmt.Sprintf("task%d pending CUPTI collector", task), failed.Close)
			}
			memoryNativeWrite(t, dir, "profile-raw.json", raw)
			if err != nil {
				t.Fatal("profile qualification pending: ", err)
			}
			memoryNativeWrite(t, dir, "profile.json", profile)
			if !profile.HardwareObserved() || profile.State.PendingHardware {
				t.Fatal("profile not admitted")
			}
			// Run the real constructor against mutated copies of the native
			// capture. Neither a lossy capture nor another device can qualify.
			badCapture := *raw
			badCapture.Quality.DroppedRanges = 1
			if _, err := observeBottleneckProfile(dev, profile, &badCapture); !errors.Is(err, cudatrial.ErrUnqualifiedOutcome) {
				t.Fatal("lossy native capture admitted")
			}
			badCapture = *raw
			badCapture.Metadata = make(map[string]string)
			for k, v := range raw.Metadata {
				badCapture.Metadata[k] = v
			}
			badCapture.Metadata["device_uuid"] = "wrong-device"
			if _, err := observeBottleneckProfile(dev, profile, &badCapture); !errors.Is(err, cudatrial.ErrUnqualifiedOutcome) {
				t.Fatal("mixed-device native capture admitted")
			}
			memoryNativeWrite(t, dir, "profile-refusal-controls.json", map[string]bool{"dropped_ranges_refused": true, "mixed_device_refused": true})
			if err := catalog.Update(cudacache.Strategy{Family: profile.State.StateName, TargetArch: target, Technique: "vectorized_memory_access", Confidence: 1, PredictedSpeedup: 1}); err != nil {
				t.Fatal(err)
			}
			if err := catalog.Update(cudacache.Strategy{Family: profile.State.StateName, TargetArch: target, Technique: "identity_control", Confidence: 0, PredictedSpeedup: 1}); err != nil {
				t.Fatal(err)
			}
			mapping := map[string]string{"vectorize": "vectorized_memory_access", "a-null": "identity_control"}
			config := &SearchMemory{Profile: &profile, WorkloadKey: binding.Key(), InputDigest: inputID, TargetArch: target, Domain: domain, Knowledge: &catalog, Trajectories: memory, RuleTechniques: mapping}
			compileCount := 0
			eval := &ExecutableEvaluator{Generation: 1, Workload: binding, CostModel: memoryNativeCost{}, Baseline: base, References: map[string][]float32{"out": want}, Rounds: 4, TimingKind: cudatrial.TimingSummedKernel, HardwareLockChecker: func() bool { _, err := cudalock.Inherited(os.Getenv); return err == nil }}
			eval.Compile = func(ctx context.Context, expr *egraph.Expr, _ WorkloadBindings) (Executable, error) {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				entry := expr.Op
				if entry == "scalar_copy" {
					entry = "scalar"
				}
				x, err := newMemoryNativeExecutable(dev, module, entry, target, artifact, compiler, policy, input, want)
				serial := compileCount
				compileCount++
				if x != nil {
					owner.add(memoryExecutableStage, fmt.Sprintf("task%d executable%d", task, serial), func() error {
						path := filepath.Join(dir, fmt.Sprintf("output-%02d-%s.fp32", serial, expr.Op))
						if err := os.WriteFile(path, memoryNativeFP32(x.outputs), 0600); err != nil {
							return fmt.Errorf("retain readback: %w", err)
						}
						return x.Close()
					})
				}
				return x, err
			}
			owner.add(memoryEvaluatorStage, fmt.Sprintf("task%d evaluator", task), eval.Close)
			newState := func() *egraph.SearchState {
				g := egraph.New()
				id := g.Add(egraph.ENode{Op: "scalar"})
				return egraph.NewSearchState(g, map[string]egraph.Id{"root": id})
			}
			budget := SearchBudget{MaxStates: 2, MaxDepth: 1, MaxActions: 2, MaxTotalActions: 2, MaxNodesPerState: 16, MaxClassesPerState: 16, MaxMemoryBytes: 1 << 20, MaxEvaluations: 2, MaxCompilations: 2, MaxExtractions: 2, MaxExtractionWork: 1000, Timeout: 20 * time.Second}
			rules := []*egraph.Rewrite{egraph.MustRewrite("a-null", "scalar", "scalar_copy"), egraph.MustRewrite("vectorize", "scalar", "vector")}
			if task > 0 && previous != nil {
				if previous.WorkloadKey == binding.Key() {
					t.Fatal("cross-task identity did not change")
				}
				// Replay compatibility is exact category/domain/target. An explicit
				// transfer-policy experiment consults the old category as a prior;
				// it does not relabel this task's independently measured profile.
				transfer := *config
				transfer.Profile = nil
				transfer.Family = previous.BottleneckState
				r, err := NewBeamSearch(BeamConfig{Budget: budget, Memory: &transfer, EvaluatorID: "lane46-native-affine-v1"}).Search(ctx, newState(), rules, eval)
				if err != nil {
					t.Fatal(err)
				}
				found := false
				for _, id := range r.MemoryDecision.TrajectoryIDs {
					if id == previous.SHA256 {
						found = true
					}
				}
				if !found || !r.MemoryDecision.State.PendingHardware {
					t.Fatal("cross-task native history not consumed honestly")
				}
				memoryNativeWrite(t, dir, "cross-task-transfer-search.json", r)
			}
			for _, strategy := range []string{"beam", "mcts"} {
				var result *SearchResult
				if strategy == "beam" {
					result, err = NewBeamSearch(BeamConfig{Budget: budget, Memory: config, EvaluatorID: "lane46-native-affine-v1"}).Search(ctx, newState(), rules, eval)
				} else {
					result, err = NewMCTSSearch(MCTSConfig{Budget: budget, Memory: config, Seed: 2, EvaluatorID: "lane46-native-affine-v1"}).Search(ctx, newState(), rules, eval)
				}
				if err != nil {
					t.Fatal(err)
				}
				memoryNativeWrite(t, dir, strategy+"-search.json", result)
				decision := result.FrozenMemory()
				if decision == nil || decision.ProfileID != profile.ID || decision.State.PendingHardware {
					t.Fatal("real consumer missed profile")
				}
				if previous != nil {
					found := false
					for _, id := range decision.TrajectoryIDs {
						if id == previous.SHA256 {
							found = true
						}
					}
					if found != (previous.BottleneckState == profile.State.StateName) {
						t.Fatal("trajectory category compatibility violated")
					}
				}
				var candidate *CandidatePlan
				for _, c := range result.AllCandidates {
					if len(c.Derivation) > 0 {
						candidate = c
					}
				}
				if candidate == nil {
					t.Fatal("native transition not evaluated")
				}
				trajectory, err := ObserveCandidateTrajectory(dev, cudatrial.TrajectoryReceipt{Domain: domain, TargetArch: target, BottleneckState: profile.State.StateName}, candidate)
				if err != nil || !trajectory.HardwareObserved() {
					t.Fatalf("trajectory admission: %v", err)
				}
				memoryNativeWrite(t, dir, strategy+"-trajectory.json", trajectory)
				if err := memory.Record(trajectory); err != nil {
					t.Fatal(err)
				}
				if err := cache.PutTrajectory(ctx, trajectory); err != nil {
					t.Fatal(err)
				}
				persisted, err := cache.GetTrajectory(ctx, trajectory.SHA256)
				if err != nil || persisted.HardwareObserved() || persisted.SHA256 != trajectory.SHA256 {
					t.Fatalf("cache roundtrip: %v", err)
				}
				if err := catalog.RecordTrajectory(trajectory, mapping); err != nil {
					t.Fatal(err)
				}
				if err := catalog.SaveFile(filepath.Join(dir, strategy+"-catalog.json")); err != nil {
					t.Fatal(err)
				}
				previous = &trajectory
			}
			// Independent null trial: same implementation and workload on both
			// arms, retained without a speedup assertion or reward admission.
			state := newState()
			null, err := eval.Evaluate(ctx, state, state.Roots["root"])
			if err != nil {
				t.Fatal(err)
			}
			memoryNativeWrite(t, dir, "null-abba.json", null.Trials)
			// Wrong fast candidate must fail before any timing is admitted.
			g := egraph.New()
			id := g.Add(egraph.ENode{Op: "wrong"})
			badState := egraph.NewSearchState(g, map[string]egraph.Id{"root": id})
			bad, err := eval.Evaluate(ctx, badState, id)
			if !errors.Is(err, ErrVerificationFailed) || bad != nil || len(eval.Receipts) != 0 {
				t.Fatalf("wrong candidate reached timing: %v", err)
			}
			memoryNativeWrite(t, dir, "wrong-output-refusal.json", map[string]any{"error": err.Error(), "comparison": eval.Comparison, "timed_receipts": len(eval.Receipts)})
			t.Logf("target=%s category=%s profile=%s previous=%v; no speedup or promotion assertion", target, profile.State.StateName, profile.ID, previous.SHA256)
		})
		if t.Failed() {
			break
		}
	}
	if previous != nil && !t.Failed() {
		path := filepath.Join(root, "cache", "receipts", "trajectories", previous.SHA256+".json")
		bytes, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var corrupt cudatrial.TrajectoryReceipt
		if err := json.Unmarshal(bytes, &corrupt); err != nil {
			t.Fatal(err)
		}
		corrupt.SpeedupRatio++
		badBytes, err := json.Marshal(corrupt)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, badBytes, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := cache.GetTrajectory(ctx, previous.SHA256); !errors.Is(err, cudacache.ErrCorruptReceipt) {
			t.Fatal("corrupt actual trajectory was admitted")
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatal("corrupt receipt not quarantined")
		}
		memoryNativeWrite(t, root, "trajectory-quarantine-control.json", map[string]any{"checksum": previous.SHA256, "corrupt_receipt_refused": true})
	}
	cuptiPath, err := cudacache.LoadedLibraryPath(cupti.Path())
	if err != nil {
		t.Fatal(err)
	}
	cuptiBytes, err := os.ReadFile(cuptiPath)
	if err != nil {
		t.Fatal(err)
	}
	memoryNativeWrite(t, root, "cupti-library.json", map[string]string{"path": cuptiPath, "sha256": memoryNativeDigest(cuptiBytes)})
}
