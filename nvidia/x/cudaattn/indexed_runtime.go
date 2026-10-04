package cudaattn

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudablock"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// IndexedBuffers contains caller-owned exact-sized device allocations. All
// active ranges must be disjoint, including read-only ranges. IDs and Counts
// use the explicitly configured selection heads. Status and scratch are never
// admitted output. Scores and its lengths/offsets are needed only by the pipeline.
// K/V may be slices from an actual cudakv contiguous cache or an externally
// leased physical pool. Allocation borrows prevent Free, not serving-page reuse:
// the caller excludes writes, remapping/recycling and all concurrent buffer,
// stream and device use until success or a successful retry Close.
type IndexedBuffers struct {
	Q, K, V                                                   cudamem.Slice[float32]
	Sequences, Positions, Lengths, Pages, IDs, Counts, Status cudamem.Slice[int32]
	ScratchOutput, ScratchLSE, Output, LSE                    cudamem.Slice[float32]
	Scores                                                    cudamem.Slice[float32]
	ScoreLengths, ScoreOffsets                                cudamem.Slice[int32]
}

// IndexedAttention owns the ordinary artifact and loaded module. The zero value
// is closed. Methods serialize; do not copy an IndexedAttention. A returned
// nonnil error-result owner must be closed before closing its device.
type IndexedAttention struct {
	mu       sync.Mutex
	config   IndexedConfig
	device   *cuda.Device
	artifact *cudakernel.Artifact
	module   *cuda.Module
	kernel   *cuda.Kernel
	raw      driver.Module
	cleanup  io.Closer
	pending  *cudaplan.Completion
	stream   *cuda.Stream
	releases []func()
	// Optional native fixture observer records stream event intervals. It does
	// not replace submission/completion or provide host timing fallbacks.
	observe func(string, bool) error
}

// CompileIndexedArtifact compiles the exact baseline into a direct CUBIN, without
// a device or a driver JIT. target is an explicit physical architecture, such as
// sm_121. Compiler selection is supplied through nvrtc's existing loader API.
func CompileIndexedArtifact(target string) (*cudakernel.Artifact, error) {
	return indexedCompileSource(target, IndexedSource)
}

func indexedCompileSource(target, source string) (*cudakernel.Artifact, error) {
	if !regexp.MustCompile(`^sm_[1-9][0-9]{1,2}$`).MatchString(target) {
		return nil, fmt.Errorf("cudaattn: invalid indexed physical target")
	}
	cm, err := nvrtcx.Compile(source, nvrtcx.WithName("cudaattn_indexed.cu"), nvrtcx.WithPhysicalArchTarget(target), nvrtcx.WithFlag("--std=c++17"), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithFlag("--ftz=false"), nvrtcx.WithNoCache())
	if err != nil {
		return nil, err
	}
	s, err := IndexedSchema()
	if err != nil {
		return nil, err
	}
	a, err := cm.ToArtifact("indexed_attention", s, target)
	if err != nil {
		return nil, err
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256([]byte(source))
	a.Toolchain = fmt.Sprintf("nvrtc %d.%d", major, minor)
	a.CompilerOpts = []string{"--gpu-architecture=" + target, "--std=c++17", "--fmad=false", "--ftz=false"}
	a.Dependencies = append(a.Dependencies, cudakernel.Dependency{Path: "cudaattn_indexed.cu", Digest: hex.EncodeToString(sum[:]), Tracked: true})
	return a, nil
}

// CompileIndexedAttention compiles for the actual device and loads the baseline.
func CompileIndexedAttention(dev *cuda.Device, c IndexedConfig) (*IndexedAttention, error) {
	if _, err := c.Extents(); err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, fmt.Errorf("cudaattn: nil indexed device")
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, err
	}
	a, err := CompileIndexedArtifact(fmt.Sprintf("sm_%d%d", major, minor))
	if err != nil {
		t := &IndexedAttention{device: dev, config: c}
		if errors.As(err, &t.cleanup) {
			return t, err
		}
		return nil, err
	}
	return LoadIndexedAttention(dev, c, a)
}

// LoadIndexedAttention admits an independently retained baseline image and ABI.
// It verifies the exact schema, source identity and entry before loading. A
// digest binds bytes; it does not attest compiler provenance or correctness.
func LoadIndexedAttention(dev *cuda.Device, c IndexedConfig, a *cudakernel.Artifact) (*IndexedAttention, error) {
	return indexedLoadSource(dev, c, a, IndexedSource)
}

func indexedLoadSource(dev *cuda.Device, c IndexedConfig, a *cudakernel.Artifact, source string) (*IndexedAttention, error) {
	if _, err := c.Extents(); err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, fmt.Errorf("cudaattn: nil indexed device")
	}
	if err := indexedAdmitSource(a, source); err != nil {
		return nil, err
	}
	t := &IndexedAttention{config: c, device: dev, artifact: indexedClone(a)}
	var err error
	t.raw, err = t.artifact.Load(dev)
	if err != nil {
		return t, err
	}
	t.module, err = cuda.OwnModule(dev, t.raw)
	if err != nil {
		return t, err
	}
	t.raw = 0
	t.kernel, err = t.module.Function(a.EntryPoint)
	if err == nil {
		limit, attributeErr := t.kernel.Attribute(driver.FuncAttributeMaxThreadsPerBlock)
		if attributeErr != nil {
			return t, fmt.Errorf("cudaattn: indexed thread admission: %w", attributeErr)
		}
		if limit < 1 {
			return t, fmt.Errorf("cudaattn: indexed kernel admits no block threads")
		}
	}
	return t, err
}

func indexedAdmitArtifact(a *cudakernel.Artifact) error {
	return indexedAdmitSource(a, IndexedSource)
}

func indexedAdmitSource(a *cudakernel.Artifact, source string) error {
	if a == nil {
		return fmt.Errorf("cudaattn: missing indexed artifact")
	}
	if err := a.Validate(); err != nil {
		return err
	}
	if a.Kind != cudakernel.KindCUBIN {
		return fmt.Errorf("cudaattn: indexed baseline requires direct cubin")
	}
	options := []string{"--gpu-architecture=" + a.Target, "--std=c++17", "--fmad=false", "--ftz=false"}
	if !reflect.DeepEqual(a.CompilerOpts, options) || a.Toolchain == "" || a.JITOptions != (cudakernel.JITOptions{}) {
		return fmt.Errorf("cudaattn: indexed numerical/compiler mode disagreement")
	}
	s, err := IndexedSchema()
	if err != nil {
		return err
	}
	if a.EntryPoint != "indexed_attention" || !reflect.DeepEqual(a.Schema, s) {
		return fmt.Errorf("cudaattn: indexed artifact ABI mismatch")
	}
	sum := sha256.Sum256([]byte(source))
	found := false
	for _, d := range a.Dependencies {
		if d.Path != "cudaattn_indexed.cu" {
			continue
		}
		if found || !d.Tracked || d.Digest != hex.EncodeToString(sum[:]) || d.ErrorMsg != "" {
			return fmt.Errorf("cudaattn: indexed artifact source disagreement")
		}
		found = true
	}
	if !found {
		return fmt.Errorf("cudaattn: indexed artifact source mismatch")
	}
	return nil
}

func indexedClone(a *cudakernel.Artifact) *cudakernel.Artifact {
	if a == nil {
		return nil
	}
	b := *a
	b.Raw = append([]byte(nil), a.Raw...)
	b.Schema.Params = append([]cudakernel.Param(nil), a.Schema.Params...)
	b.CompilerOpts = append([]string(nil), a.CompilerOpts...)
	b.Dependencies = append([]cudakernel.Dependency(nil), a.Dependencies...)
	return &b
}

// Artifact returns a detached descriptor of the loaded image.
func (t *IndexedAttention) Artifact() *cudakernel.Artifact {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return indexedClone(t.artifact)
}

// IndexedTopKConfig returns the exact public TopK configuration for the proxy
// pipeline. Shared KV-head selection is unsupported by this per-query-head proxy.
func IndexedTopKConfig(c IndexedConfig) (cudablock.TopKConfig, error) {
	e, err := c.Extents()
	if err != nil {
		return cudablock.TopKConfig{}, err
	}
	if c.Selection != IndexedQueryHeads {
		return cudablock.TopKConfig{}, fmt.Errorf("cudaattn: proxy requires query-head selection")
	}
	return cudablock.TopKConfig{Rows: e.Rows, Width: e.MaxBlocks, RowStride: e.MaxBlocks, K: c.SelectedBlocks}, nil
}

// Execute validates the device metadata/IDs in a separate completed dispatch,
// computes into scratch, validates status, then publishes output and LSE. A
// pre-publication refusal leaves output untouched. Publication failure leaves
// outputs undefined. Cancellation/drain failure retains every allocation borrow
// for retry Close; cancellation does not stop the GPU or release its resources.
func (t *IndexedAttention) Execute(ctx context.Context, stream *cuda.Stream, b IndexedBuffers) error {
	return t.execute(ctx, stream, b, nil, cudablock.TopKBuffers{})
}

// ExecuteSelection runs the real proxy-score -> public TopK.Execute -> indexed
// attention pipeline. selector must be compiled with IndexedTopKConfig. TopK's
// Input/Lengths/Offsets and published Indices/Counts must be the corresponding
// IndexedBuffers allocations; all its other buffers must be disjoint. No CPU
// score/sort or inferred head sharing participates. The caller retains selector
// and closes it after any failure, before freeing its allocations or device.
// TopK.Execute is synchronous and has no cancellation API; ctx is checked at
// each indexed stream drain. No timing channel is substituted for GPU duration.
func (t *IndexedAttention) ExecuteSelection(ctx context.Context, stream *cuda.Stream, b IndexedBuffers, selector *cudablock.TopK, selection cudablock.TopKBuffers) error {
	if selector == nil {
		return fmt.Errorf("cudaattn: missing indexed topk selector")
	}
	return t.execute(ctx, stream, b, selector, selection)
}

func (t *IndexedAttention) execute(ctx context.Context, stream *cuda.Stream, b IndexedBuffers, selector *cudablock.TopK, s cudablock.TopKBuffers) (err error) {
	if t == nil {
		return fmt.Errorf("cudaattn: closed indexed attention")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.kernel == nil || t.stream != nil {
		return fmt.Errorf("cudaattn: indexed closed or awaiting drain")
	}
	if stream == nil || stream.Device() != t.device {
		return fmt.Errorf("cudaattn: indexed stream device mismatch")
	}
	pipeline := selector != nil
	if pipeline {
		if _, err := IndexedTopKConfig(t.config); err != nil {
			return err
		}
		if s.Input.Ptr() != b.Scores.Ptr() || s.Lengths.Ptr() != b.ScoreLengths.Ptr() || s.Offsets.Ptr() != b.ScoreOffsets.Ptr() || s.Indices.Ptr() != b.IDs.Ptr() || s.Counts.Ptr() != b.Counts.Ptr() {
			return fmt.Errorf("cudaattn: indexed topk pipeline binding mismatch")
		}
	}
	bufs := indexedBufferList(t.config, b, pipeline)
	if pipeline {
		e, _ := t.config.Extents()
		bufs = append(bufs, indexedView("topk scratch indices", s.ScratchIndices, e.IDs), indexedView("topk scratch counts", s.ScratchCounts, e.Rows), indexedView("topk status", s.Status, e.Rows))
		if s.ScratchValues.Len() != 0 || s.Values.Len() != 0 {
			return fmt.Errorf("cudaattn: indexed topk must be indices-only")
		}
	}
	releases, err := indexedBorrow(bufs, t.device)
	if err != nil {
		return err
	}
	t.releases, t.stream = releases, stream
	defer func() {
		if t.pending != nil && t.pending.Done() {
			t.release()
		}
	}()
	mode := int32(1)
	if pipeline {
		mode = 0
	}
	if err = t.stage(ctx, b, mode); err != nil {
		return err
	}
	if err = t.status(b); err != nil {
		return err
	}
	if pipeline {
		if err = t.phase("score", func() error { return t.stage(ctx, b, 3) }); err != nil {
			return err
		}
		if err = t.status(b); err != nil {
			return err
		}
		if err = t.phase("selection", func() error { return selector.Execute(stream, s) }); err != nil { // TopK may retain a failed drain; establish our own drain before releasing.
			t.pending = nil
			return fmt.Errorf("cudaattn: indexed topk: %w", err)
		}
		if err = t.stage(ctx, b, 1); err != nil {
			return err
		}
		if err = t.status(b); err != nil {
			return err
		}
	}
	if err = t.phase("attention", func() error { return t.stage(ctx, b, 2) }); err != nil {
		return err
	}
	if err = t.status(b); err != nil {
		return err
	}
	copyErr := b.Output.CopyDeviceToDeviceAsync(b.ScratchOutput, stream)
	if copyErr == nil {
		copyErr = b.LSE.CopyDeviceToDeviceAsync(b.ScratchLSE, stream)
	}
	return errors.Join(copyErr, t.drain(ctx))
}

func (t *IndexedAttention) phase(name string, action func() error) error {
	if t.observe != nil {
		if err := t.observe(name, true); err != nil {
			return err
		}
	}
	err := action()
	if t.observe != nil {
		err = errors.Join(err, t.observe(name, false))
	}
	return err
}

func (t *IndexedAttention) stage(ctx context.Context, b IndexedBuffers, mode int32) error {
	t.pending = nil
	e, _ := t.config.Extents()
	status := make([]int32, e.Rows)
	for i := range status {
		status[i] = -1
	}
	if err := b.Status.CopyFromHost(status); err != nil {
		return err
	}
	err := t.launch(b, mode)
	return errors.Join(err, t.drain(ctx))
}
func (t *IndexedAttention) drain(ctx context.Context) error {
	var err error
	t.pending, err = cudaplan.NewStreamCompletion(t.stream)
	if err != nil {
		return err
	}
	return t.pending.Wait(ctx)
}
func (t *IndexedAttention) status(b IndexedBuffers) error {
	e, _ := t.config.Extents()
	s := make([]int32, e.Rows)
	if err := b.Status.CopyToHost(s); err != nil {
		return err
	}
	return CheckIndexedStatus(s, e.Rows)
}
func (t *IndexedAttention) release() {
	for _, f := range t.releases {
		f()
	}
	t.releases = nil
	t.stream = nil
	t.pending = nil
}
func (t *IndexedAttention) launch(b IndexedBuffers, mode int32) error {
	ptrs := [15]driver.DevicePtr{b.Q.Ptr(), b.K.Ptr(), b.V.Ptr(), b.Sequences.Ptr(), b.Positions.Ptr(), b.Lengths.Ptr(), b.Pages.Ptr(), b.IDs.Ptr(), b.Counts.Ptr(), b.Status.Ptr(), b.ScratchOutput.Ptr(), b.ScratchLSE.Ptr(), b.Scores.Ptr(), b.ScoreLengths.Ptr(), b.ScoreOffsets.Ptr()}
	args, err := indexedLaunchArguments(t.artifact.Schema, t.config, ptrs, mode)
	if err != nil {
		return err
	}
	return t.stream.WithHandle(func(h driver.Stream) error {
		return t.kernel.Launch([3]uint32{uint32(t.config.Queries * t.config.QueryHeads), 1, 1}, [3]uint32{1, 1, 1}, 0, h, args)
	})
}

func indexedLaunchArguments(schema cudakernel.Schema, c IndexedConfig, ptrs [15]driver.DevicePtr, mode int32) ([]unsafe.Pointer, error) {
	want, err := IndexedSchema()
	if err != nil {
		return nil, err
	}
	if !reflect.DeepEqual(schema, want) {
		return nil, fmt.Errorf("cudaattn: indexed launch ABI mismatch")
	}
	if _, err := c.Extents(); err != nil {
		return nil, err
	}
	if mode < 0 || mode > 3 {
		return nil, fmt.Errorf("cudaattn: invalid indexed launch mode")
	}
	shared, causal := int32(0), int32(0)
	if c.Selection == IndexedKVHeads {
		shared = 1
	}
	if c.Causal {
		causal = 1
	}
	ints := []int32{int32(c.Batch), int32(c.Queries), int32(c.QueryHeads), int32(c.KVHeads), int32(c.HeadDim), int32(c.BlockSize), int32(c.MaxSequence), int32(c.PhysicalBlocks), int32(c.SelectedBlocks), shared, causal}
	var args []unsafe.Pointer
	for i := range ptrs {
		if uint64(ptrs[i])%4 != 0 {
			return nil, fmt.Errorf("cudaattn: indexed pointer %d is not four-byte aligned", i)
		}
		args = append(args, unsafe.Pointer(&ptrs[i]))
	}
	for i := range ints {
		args = append(args, unsafe.Pointer(&ints[i]))
	}
	scale := c.Scale
	args = append(args, unsafe.Pointer(&scale), unsafe.Pointer(&mode))
	return args, nil
}

// Close drains a failed invocation and then unloads the module. A failed drain
// or cleanup retains ownership for a later retry. It never publishes scratch.
func (t *IndexedAttention) Close() error {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.stream != nil {
		if err := t.drain(nil); err != nil {
			return err
		}
		t.release()
	}
	if t.module != nil {
		if err := t.module.Close(); err != nil {
			return err
		}
		t.module = nil
		t.kernel = nil
	}
	if t.raw != 0 {
		if err := t.device.WithContext(func() error { return t.raw.Close() }); err != nil {
			return err
		}
		t.raw = 0
	}
	if t.cleanup != nil {
		if err := t.cleanup.Close(); err != nil {
			return err
		}
		t.cleanup = nil
	}
	return nil
}

type indexedBuffer struct {
	name         string
	ptr          uint64
	length, need int
	owner        *cuda.Device
	borrow       func(*cuda.Device) (func(), error)
}

func indexedView[T any](name string, s cudamem.Slice[T], need int) indexedBuffer {
	return indexedBuffer{name, uint64(s.Ptr()), s.Len(), need, s.Device(), s.Borrow}
}
func indexedBufferList(c IndexedConfig, b IndexedBuffers, pipeline bool) []indexedBuffer {
	e, _ := c.Extents()
	scores, metadata := 0, 0
	if pipeline {
		scores, metadata = e.Scores, e.Rows
	}
	return []indexedBuffer{indexedView("q", b.Q, e.Query), indexedView("k", b.K, e.KV), indexedView("v", b.V, e.KV), indexedView("sequences", b.Sequences, c.Queries), indexedView("positions", b.Positions, c.Queries), indexedView("lengths", b.Lengths, c.Batch), indexedView("pages", b.Pages, e.Pages), indexedView("ids", b.IDs, e.IDs), indexedView("counts", b.Counts, e.Sets), indexedView("status", b.Status, e.Rows), indexedView("scratch output", b.ScratchOutput, e.Query), indexedView("scratch lse", b.ScratchLSE, e.Rows), indexedView("output", b.Output, e.Query), indexedView("lse", b.LSE, e.Rows), indexedView("scores", b.Scores, scores), indexedView("score lengths", b.ScoreLengths, metadata), indexedView("score offsets", b.ScoreOffsets, metadata)}
}
func indexedCheckBuffers(b []indexedBuffer, dev *cuda.Device) error {
	for i, v := range b {
		if v.need == 0 {
			if v.length != 0 {
				return fmt.Errorf("cudaattn: indexed %s must be empty", v.name)
			}
			continue
		}
		if v.length != v.need || v.ptr == 0 || v.ptr%4 != 0 || v.owner != dev || v.ptr > ^uint64(0)-uint64(v.need)*4 {
			return fmt.Errorf("cudaattn: invalid indexed %s buffer", v.name)
		}
		for _, p := range b[:i] {
			if p.need > 0 && p.ptr < v.ptr+uint64(v.need)*4 && v.ptr < p.ptr+uint64(p.need)*4 {
				return fmt.Errorf("cudaattn: indexed buffers %s and %s alias", p.name, v.name)
			}
		}
	}
	return nil
}
func indexedBorrow(b []indexedBuffer, dev *cuda.Device) ([]func(), error) {
	if err := indexedCheckBuffers(b, dev); err != nil {
		return nil, err
	}
	var releases []func()
	fail := func(err error) ([]func(), error) {
		for _, f := range releases {
			f()
		}
		return nil, err
	}
	for _, v := range b {
		if v.need == 0 {
			continue
		}
		f, err := v.borrow(dev)
		if err != nil {
			return fail(err)
		}
		releases = append(releases, f)
	}
	if err := indexedCheckBuffers(b, dev); err != nil {
		return fail(err)
	}
	return releases, nil
}
