//go:build native

package cudaattn

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudablock"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func indexedNativeSources() map[string]string {
	sources := map[string]string{"baseline": IndexedSource}
	for _, m := range indexedMutants() {
		s := strings.Replace(IndexedSource, m.old, m.new, 1)
		if m.name == "sentinel-aliases-zero" {
			s = strings.Replace(s, "int physical=pages[seq*max_blocks+logical];", "logical=logical<0?0:logical; int physical=pages[seq*max_blocks+logical];", 1)
		}
		sources[m.name] = s
	}
	return sources
}
func indexedPacketOptions(topk bool) []string {
	if topk {
		return []string{"--gpu-architecture=sm_121", "--std=c++17", "--generate-line-info"}
	}
	return []string{"--gpu-architecture=sm_121", "--std=c++17", "--fmad=false", "--ftz=false"}
}

// TestBuildIndexedPacket invokes only NVRTC and host file IO. It never loads a
// driver or queries a device. The eight images are compiled outside gpu-lock.
func TestBuildIndexedPacket(t *testing.T) {
	if os.Getenv("INDEXED_BUILD_ONLY") != "1" {
		t.Skip("offline build not requested")
	}
	dir, path, builtins := os.Getenv("INDEXED_PACKET_DIR"), os.Getenv("INDEXED_NVRTC_PATH"), os.Getenv("INDEXED_BUILTINS_PATH")
	if !filepath.IsAbs(dir) || !filepath.IsAbs(path) || !filepath.IsAbs(builtins) {
		t.Fatal("absolute explicit paths required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := nvrtc.Load(path); err != nil {
		t.Fatal(err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil || major != 13 || minor != 4 {
		t.Fatalf("compiler version %d.%d %v", major, minor, err)
	}
	compiler, err := indexedReadBounded(path, 512<<20)
	if err != nil {
		t.Fatal(err)
	}
	builtin, err := indexedReadBounded(builtins, 512<<20)
	if err != nil {
		t.Fatal(err)
	}
	sources := indexedNativeSources()
	sources["topk"] = cudablock.TopKSource
	for name, source := range sources {
		var a *cudakernel.Artifact
		var err error
		if name == "topk" {
			cm, e := nvrtcx.Compile(source, nvrtcx.WithName("cudablock_topk.cu"), nvrtcx.WithPhysicalArchTarget("sm_121"), nvrtcx.WithFlag("--std=c++17"), nvrtcx.WithFlag("--generate-line-info"), nvrtcx.WithNoCache())
			err = e
			if err == nil {
				s, _ := cudablock.TopKSchema()
				a, err = cm.ToArtifact("topk_fp32", s, "sm_121")
				if err == nil {
					a.Toolchain = "nvrtc 13.4"
					a.CompilerOpts = indexedPacketOptions(true)
					a.Dependencies = append(a.Dependencies, cudakernel.Dependency{Path: "cudablock_topk.cu", Digest: indexedDigest([]byte(source)), Tracked: true})
				}
			}
		} else {
			a, err = indexedCompileSource("sm_121", source)
		}
		if err != nil {
			t.Fatalf("compile %s: %v", name, err)
		}
		p := indexedPacket{Version: 1, SourceDigest: indexedDigest([]byte(source)), ImageDigest: a.Digest, CompilerPath: nvrtc.Path(), CompilerVersion: a.Toolchain, CompilerDigest: indexedDigest(compiler), BuiltinsPath: builtins, BuiltinsDigest: indexedDigest(builtin), Artifact: *a}
		b, err := json.MarshalIndent(p, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		for suffix, data := range map[string][]byte{".json": b, ".cu": []byte(source), ".cubin": a.Raw} {
			if err := os.WriteFile(filepath.Join(dir, name+suffix), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("compiled %s source=%s image=%s compiler=%s", name, p.SourceDigest, p.ImageDigest, p.CompilerDigest)
	}
}

func TestIndexedPreparedPacketMetadata(t *testing.T) {
	dir := os.Getenv("INDEXED_HOST_PACKET_DIR")
	if dir == "" {
		t.Skip("retained offline packet not requested")
	}
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute retained packet path required")
	}
	schema, _ := IndexedSchema()
	for name, source := range indexedNativeSources() {
		if _, err := indexedReadPacket(dir, name, source, "indexed_attention", schema, indexedPacketOptions(false)); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	schema, _ = cudablock.TopKSchema()
	if _, err := indexedReadPacket(dir, "topk", cudablock.TopKSource, "topk_fp32", schema, indexedPacketOptions(true)); err != nil {
		t.Fatal(err)
	}
}

func indexedNativeDevice(t *testing.T) *cuda.Device {
	t.Helper()
	if os.Getenv("GPU_LOCK_HELD") == "" || os.Getenv("GPU_LOCK_DISABLED") != "" || os.Getenv("CUDATEST_NO_LOCK") != "" {
		t.Fatal("native indexed packet requires coordinator gpu-lock lease")
	}
	if err := driver.Load(""); err != nil {
		t.Fatal(err)
	}
	if err := cuda.Init(); err != nil {
		t.Fatal(err)
	}
	count, err := cuda.DeviceCount()
	if err != nil || count < 1 {
		t.Fatalf("native indexed packet has no admitted device: %d %v", count, err)
	}
	dev := cudatest.Require(t)
	major, minor, err := dev.ComputeCapability()
	if err != nil || major != 12 || minor != 1 {
		t.Fatalf("packet target mismatch %d.%d %v", major, minor, err)
	}
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	binaryPath, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binaryBytes, err := indexedReadBounded(binaryPath, 32<<20)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("producer go=%s platform=%s/%s driver=%s API=%d device=%s uuid=%x sm=%d%d executing_binary=%s", runtime.Version(), runtime.GOOS, runtime.GOARCH, driver.Path(), version, name, uuid, major, minor, indexedDigest(binaryBytes))
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	return dev
}
func indexedNativeAllocation[T any](t *testing.T, dev *cuda.Device, values []T) cudamem.Slice[T] {
	t.Helper()
	var guard T
	switch any(guard).(type) {
	case float32:
		guard = any(float32(-555.5)).(T)
	case int32:
		guard = any(int32(0x55aa12)).(T)
	default:
		t.Fatal("unsupported native fixture guard type")
	}
	whole := make([]T, len(values)+2)
	whole[0], whole[len(whole)-1] = guard, guard
	copy(whole[1:], values)
	s, err := cudamem.Alloc[T](dev, len(whole))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if s.Ptr() != 0 {
			read := make([]T, len(whole))
			if err := s.CopyToHost(read); err != nil {
				t.Error(err)
			} else if !reflect.DeepEqual(read[0], guard) || !reflect.DeepEqual(read[len(read)-1], guard) {
				t.Error("native indexed allocation redzone damaged")
			}
		}
		if err := s.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := s.CopyFromHost(whole); err != nil {
		t.Fatal(err)
	}
	return s.Slice(1, len(values)+1)
}
func indexedNativeBuffers(t *testing.T, dev *cuda.Device, f indexedFixture) (IndexedBuffers, cudablock.TopKBuffers) {
	t.Helper()
	c, r := f.Config, f.Request
	e, _ := c.Extents()
	floats := func(n int) cudamem.Slice[float32] {
		v := make([]float32, n)
		for i := range v {
			v[i] = -987
		}
		return indexedNativeAllocation(t, dev, v)
	}
	ints := func(n int) cudamem.Slice[int32] { return indexedNativeAllocation(t, dev, make([]int32, n)) }
	b := IndexedBuffers{Q: indexedNativeAllocation(t, dev, f.Q), K: indexedNativeAllocation(t, dev, f.K), V: indexedNativeAllocation(t, dev, f.V), Sequences: indexedNativeAllocation(t, dev, r.Sequences), Positions: indexedNativeAllocation(t, dev, r.Positions), Lengths: indexedNativeAllocation(t, dev, r.Lengths), Pages: indexedNativeAllocation(t, dev, r.Pages), IDs: indexedNativeAllocation(t, dev, r.IDs), Counts: indexedNativeAllocation(t, dev, r.Counts), Status: ints(e.Rows), ScratchOutput: floats(e.Query), ScratchLSE: floats(e.Rows), Output: floats(e.Query), LSE: floats(e.Rows)}
	if !f.Pipeline {
		return b, cudablock.TopKBuffers{}
	}
	b.Scores = floats(e.Scores)
	b.ScoreLengths = ints(e.Rows)
	b.ScoreOffsets = ints(e.Rows)
	return b, cudablock.TopKBuffers{Input: b.Scores, Lengths: b.ScoreLengths, Offsets: b.ScoreOffsets, ScratchIndices: ints(e.IDs), ScratchCounts: ints(e.Rows), Status: ints(e.Rows), Indices: b.IDs, Counts: b.Counts}
}

func indexedNativeOwner(t *testing.T, dev *cuda.Device, c IndexedConfig, name, source string) *IndexedAttention {
	t.Helper()
	dir := os.Getenv("INDEXED_PACKET_DIR")
	schema, _ := IndexedSchema()
	p, err := indexedReadPacket(dir, name, source, "indexed_attention", schema, indexedPacketOptions(false))
	if err != nil {
		t.Fatal(err)
	}
	owner, err := indexedLoadSource(dev, c, &p.Artifact, source)
	if owner != nil {
		t.Cleanup(func() {
			if err := owner.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range []driver.FunctionAttribute{driver.FuncAttributeNumRegs, driver.FuncAttributeLocalSizeBytes, driver.FuncAttributeSharedSizeBytes, driver.FuncAttributeMaxThreadsPerBlock} {
		v, err := owner.kernel.Attribute(a)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("loaded attribute %d=%d", a, v)
	}
	return owner
}
func indexedNativeRead[T any](t *testing.T, s cudamem.Slice[T]) []T {
	t.Helper()
	v := make([]T, s.Len())
	if err := s.CopyToHost(v); err != nil {
		t.Fatal(err)
	}
	return v
}
func indexedNativeReceipt(t *testing.T, f indexedFixture, name string, b IndexedBuffers, err error) {
	t.Helper()
	dir := os.Getenv("INDEXED_RESULT_DIR")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute result directory required")
	}
	if e := os.MkdirAll(dir, 0700); e != nil {
		t.Fatal(e)
	}
	out, lse := indexedNativeRead(t, b.Output), indexedNativeRead(t, b.LSE)
	var want, wl []float32
	if !strings.HasPrefix(name, "refuse-") {
		want, wl = indexedOracle(f)
	}
	bits := func(v []float32) []uint32 {
		r := make([]uint32, len(v))
		for i, x := range v {
			r[i] = math.Float32bits(x)
		}
		return r
	}
	result := map[string]any{"name": name, "input": indexedInputDigest(f), "output_bits": bits(out), "lse_bits": bits(lse), "reference_bits": bits(want), "reference_lse_bits": bits(wl), "status": indexedNativeRead(t, b.Status), "ids": indexedNativeRead(t, b.IDs), "counts": indexedNativeRead(t, b.Counts), "error": fmt.Sprint(err), "score_gpu_duration": "unmeasured", "selection_gpu_duration": "unmeasured", "attention_gpu_duration": "unmeasured", "composed_gpu_duration": "unmeasured", "cupti_activity_duration": "unmeasured", "wall_duration": "unmeasured"}
	data, e := json.MarshalIndent(result, "", "  ")
	if e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(filepath.Join(dir, name+".json"), data, 0600); e != nil {
		t.Fatal(e)
	}
}

func TestNativeIndexedAttentionReference(t *testing.T) {
	dev := indexedNativeDevice(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	for i, f := range indexedFixtures()[:5] {
		t.Run(f.Name, func(t *testing.T) {
			b, _ := indexedNativeBuffers(t, dev, f)
			// One actual contiguous cudakv owner supplies the physical pool;
			// logical pages retain the same noncontiguous mapping as the oracle.
			if i == 1 {
				c := f.Config
				cache, err := cudakv.New(dev, 1, c.KVHeads, c.HeadDim, c.PhysicalBlocks*c.BlockSize)
				if cache != nil {
					t.Cleanup(func() {
						if err := cache.Close(); err != nil {
							t.Error(err)
						}
					})
				}
				if err != nil {
					t.Fatal(err)
				}
				layer, err := cache.Layer(0)
				if err != nil {
					t.Fatal(err)
				}
				if err := layer.K.CopyFromHost(f.K); err != nil {
					t.Fatal(err)
				}
				if err := layer.V.CopyFromHost(f.V); err != nil {
					t.Fatal(err)
				}
				b.K, b.V = layer.K, layer.V
			}
			stream := mustIndexedStream(t, dev)
			owner := indexedNativeOwner(t, dev, f.Config, "baseline", IndexedSource)
			err := owner.Execute(ctx, stream, b)
			indexedNativeReceipt(t, f, f.Name, b, err)
			if err != nil {
				t.Fatal(err)
			}
			indexedCheckResult(t, f, IndexedSource, "sm_121", indexedNativeRead(t, b.Output), indexedNativeRead(t, b.LSE))
		})
	}
	for _, bad := range indexedBadRequests() {
		t.Run("refuse-"+bad.name, func(t *testing.T) {
			f := indexedFixtures()[0]
			bad.edit(&f)
			b, _ := indexedNativeBuffers(t, dev, f)
			stream := mustIndexedStream(t, dev)
			owner := indexedNativeOwner(t, dev, f.Config, "baseline", IndexedSource)
			err := owner.Execute(ctx, stream, b)
			indexedNativeReceipt(t, f, "refuse-"+bad.name, b, err)
			if err == nil {
				t.Fatal("bad device metadata accepted")
			}
			for _, v := range indexedNativeRead(t, b.Output) {
				if v != -987 {
					t.Fatal("refusal published output")
				}
			}
		})
	}
	for _, m := range indexedMutants() {
		t.Run("mutant-"+m.name, func(t *testing.T) {
			f := indexedFixtures()[m.fixture]
			if m.name == "duplicate-double-count" {
				f.Request.IDs[1] = f.Request.IDs[0]
			}
			src := indexedNativeSources()[m.name]
			b, _ := indexedNativeBuffers(t, dev, f)
			stream := mustIndexedStream(t, dev)
			owner := indexedNativeOwner(t, dev, f.Config, m.name, src)
			err := owner.Execute(ctx, stream, b)
			indexedNativeReceipt(t, f, "mutant-"+m.name, b, err)
			if err != nil {
				t.Fatalf("mutant execution failed instead of numerical disagreement: %v", err)
			}
			want, lse := indexedOracle(f)
			if indexedCompare(f, src, "sm_121", indexedNativeRead(t, b.Output), want, false) == nil && indexedCompare(f, src, "sm_121", indexedNativeRead(t, b.LSE), lse, true) == nil {
				t.Fatal("device source mutant survived")
			}
		})
	}
}
func mustIndexedStream(t *testing.T, dev *cuda.Device) *cuda.Stream {
	t.Helper()
	s, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	return s
}

func indexedEventPhases(t *testing.T, dev *cuda.Device, stream *cuda.Stream) (func(string, bool) error, func() map[string]any) {
	t.Helper()
	starts, ends := map[string]*cuda.Event{}, map[string]*cuda.Event{}
	for _, name := range []string{"score", "selection", "attention", "composed"} {
		for _, table := range []map[string]*cuda.Event{starts, ends} {
			e, err := dev.NewEvent()
			if err != nil {
				t.Fatal(err)
			}
			table[name] = e
			t.Cleanup(func() {
				if err := e.Close(); err != nil {
					t.Error(err)
				}
			})
		}
	}
	record := func(name string, start bool) error {
		table := ends
		if start {
			table = starts
		}
		e := table[name]
		if e == nil {
			return fmt.Errorf("unknown indexed event phase")
		}
		return e.Record(stream)
	}
	read := func() map[string]any {
		t.Helper()
		if err := ends["composed"].Synchronize(nil); err != nil {
			t.Fatal(err)
		}
		m := map[string]any{"method": "CUDA event stream intervals, including host submission/drain gaps; non-additive", "cupti_activity_duration": "unmeasured", "setup_transfer_duration": "unmeasured"}
		for name, end := range ends {
			duration, err := end.ElapsedSince(starts[name])
			if err != nil {
				t.Fatal(err)
			}
			m[name+"_stream_interval_ns"] = duration.Nanoseconds()
		}
		return m
	}
	return record, read
}

func TestNativeIndexedSelectionPipeline(t *testing.T) {
	dev := indexedNativeDevice(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	for _, empty := range []bool{false, true} {
		f := indexedFixtures()[5]
		if empty {
			f.Name = "proxy-topk-zero-length"
			f.Request.Lengths[0] = 0
			f.Request.Positions[0] = -1
			for i := 0; i < 3; i++ {
				f.Request.Pages[i] = -1
			}
		}
		t.Run(f.Name, func(t *testing.T) {
			b, selection := indexedNativeBuffers(t, dev, f)
			stream := mustIndexedStream(t, dev)
			record, readTimes := indexedEventPhases(t, dev, stream)
			owner := indexedNativeOwner(t, dev, f.Config, "baseline", IndexedSource)
			owner.observe = record
			schema, _ := cudablock.TopKSchema()
			packet, err := indexedReadPacket(os.Getenv("INDEXED_PACKET_DIR"), "topk", cudablock.TopKSource, "topk_fp32", schema, indexedPacketOptions(true))
			if err != nil {
				t.Fatal(err)
			}
			config, err := IndexedTopKConfig(f.Config)
			if err != nil {
				t.Fatal(err)
			}
			selector, err := cudablock.LoadTopK(dev, config, &packet.Artifact)
			if selector != nil {
				t.Cleanup(func() {
					if err := selector.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := record("composed", true); err != nil {
				t.Fatal(err)
			}
			wallStart := time.Now()
			err = owner.ExecuteSelection(ctx, stream, b, selector, selection)
			wallDuration := time.Since(wallStart)
			if err != nil {
				t.Fatal(err)
			}
			if err := record("composed", false); err != nil {
				t.Fatal(err)
			}
			times := readTimes()
			times["execute_wall_ns"] = wallDuration.Nanoseconds()
			data, err := json.MarshalIndent(times, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			dir := os.Getenv("INDEXED_RESULT_DIR")
			if err := os.MkdirAll(dir, 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, f.Name+".timing.json"), data, 0600); err != nil {
				t.Fatal(err)
			}
			scores, ids, counts := indexedProxyOracle(f)
			actualIDs, actualCounts := indexedNativeRead(t, b.IDs), indexedNativeRead(t, b.Counts)
			if !reflect.DeepEqual(ids, actualIDs) || !reflect.DeepEqual(counts, actualCounts) {
				t.Fatalf("device TopK selection differs: %v/%v want %v/%v", actualIDs, actualCounts, ids, counts)
			}
			if err := indexedCompare(f, IndexedSource, "sm_121", indexedNativeRead(t, b.Scores), scores, true); err != nil {
				t.Fatal(err)
			}
			f.Request.IDs, f.Request.Counts = actualIDs, actualCounts
			indexedNativeReceipt(t, f, f.Name, b, nil)
			indexedCheckResult(t, f, IndexedSource, "sm_121", indexedNativeRead(t, b.Output), indexedNativeRead(t, b.LSE))
		})
	}
	t.Run("cancel-retains-queued-borrows", func(t *testing.T) {
		f := indexedFixtures()[0]
		b, _ := indexedNativeBuffers(t, dev, f)
		stream := mustIndexedStream(t, dev)
		owner := indexedNativeOwner(t, dev, f.Config, "baseline", IndexedSource)
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()
		err := owner.Execute(cancelled, stream, b)
		if err == nil || !indexedOnlyCancelled(err) {
			t.Fatal("cancelled wait accepted")
		}
		if owner.pending == nil || owner.pending.Done() {
			t.Fatal("cancellation claimed successful device drain")
		}
		if err := b.Q.Free(); !errors.Is(err, cudamem.ErrInUse) {
			t.Fatalf("queued Q was not pinned: %v", err)
		}
		if err := b.K.Free(); !errors.Is(err, cudamem.ErrInUse) {
			t.Fatalf("queued K was not pinned: %v", err)
		}
		if err := owner.Close(); err != nil {
			t.Fatal(err)
		}
		if err := CheckIndexedStatus(indexedNativeRead(t, b.Status), f.Config.Queries*f.Config.QueryHeads); err != nil {
			t.Fatalf("queued poisoned validation status was not written: %v", err)
		}
		for _, v := range indexedNativeRead(t, b.Output) {
			if v != -987 {
				t.Fatal("cancellation/Close published output")
			}
		}
		if err := b.Q.Free(); err != nil {
			t.Fatalf("successful Close retained Q: %v", err)
		}
		t.Log("actual queued validation cancellation retained Q/K until successful Close")
	})
}

func indexedOnlyCancelled(err error) bool {
	if err == context.Canceled {
		return true
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		children := joined.Unwrap()
		if len(children) == 0 {
			return false
		}
		for _, child := range children {
			if !indexedOnlyCancelled(child) {
				return false
			}
		}
		return true
	}
	if wrapped, ok := err.(interface{ Unwrap() error }); ok {
		return indexedOnlyCancelled(wrapped.Unwrap())
	}
	return false
}
