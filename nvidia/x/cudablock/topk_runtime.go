package cudablock

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// TopKBuffers holds caller-owned allocations. Lengths and Offsets contain one
// int32 per row; offsets must be nonnegative and every live output index must
// fit int32. ScratchIndices has Rows*K elements, ScratchCounts and Status have
// Rows elements. Indices and Counts are separate publication destinations.
// ScratchValues and Values have Rows*K elements when configured, otherwise
// both must be empty. No two active buffer ranges may overlap.
//
// The caller must not write, read, free, close the device or stream, or submit
// concurrent work using these buffers during Execute. Scratch and status are
// diagnostic data, never admitted output. Outputs become usable only after a
// successful Execute; a publication or completion failure leaves them undefined.
type TopKBuffers struct {
	Input          cudamem.Slice[float32]
	Lengths        cudamem.Slice[int32]
	Offsets        cudamem.Slice[int32]
	ScratchIndices cudamem.Slice[int32]
	ScratchValues  cudamem.Slice[float32]
	ScratchCounts  cudamem.Slice[int32]
	Status         cudamem.Slice[int32]
	Indices        cudamem.Slice[int32]
	Values         cudamem.Slice[float32]
	Counts         cudamem.Slice[int32]
}

// TopK owns a loaded FP32 selection artifact. Its zero value is closed. Execute
// and Close are serialized; the device must remain live until Close succeeds.
type TopK struct {
	mu             sync.Mutex
	config         TopKConfig
	device         *cuda.Device
	artifact       *cudakernel.Artifact
	module         *cuda.Module
	kernel         *cuda.Kernel
	raw            driver.Module
	cleanup        io.Closer
	pendingStream  *cuda.Stream
	pendingRelease []func()
}

// CompileTopK compiles and loads the ordinary FP32 baseline. Geometry is one
// block per row with one thread and zero dynamic shared bytes. Compiler-allocated
// registers/local memory must be observed separately from the returned artifact;
// the native fixture records loaded attributes without treating missing facts as zero.
// A nonnil TopK returned with an error retains cleanup and must be closed.
func CompileTopK(dev *cuda.Device, config TopKConfig) (*TopK, error) {
	return compileTopKSource(dev, config, TopKSource)
}

func compileTopKSource(dev *cuda.Device, config TopKConfig, source string) (*TopK, error) {
	if _, err := config.Extents(); err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, fmt.Errorf("cudablock: nil topk device")
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("cudablock: topk capability: %w", err)
	}
	artifact, err := topkCompileArtifact(source, fmt.Sprintf("sm_%d%d", major, minor))
	if err != nil {
		t := &TopK{config: config, device: dev}
		if errors.As(err, &t.cleanup) {
			return t, fmt.Errorf("cudablock: compile topk: %w", err)
		}
		return nil, fmt.Errorf("cudablock: compile topk: %w", err)
	}
	return topkLoadArtifact(dev, config, artifact)
}

func topkCompileArtifact(source, target string) (*cudakernel.Artifact, error) {
	cm, err := nvrtcx.Compile(source, nvrtcx.WithName("cudablock_topk.cu"), nvrtcx.WithPhysicalArchTarget(target), nvrtcx.WithFlag("--std=c++17"), nvrtcx.WithFlag("--generate-line-info"), nvrtcx.WithNoCache())
	if err != nil {
		return nil, err
	}
	schema, err := TopKSchema()
	if err != nil {
		return nil, err
	}
	a, err := cm.ToArtifact("topk_fp32", schema, target)
	if err != nil {
		return nil, err
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256([]byte(source))
	a.Toolchain = fmt.Sprintf("nvrtc %d.%d", major, minor)
	a.CompilerOpts = []string{"--gpu-architecture=" + target, "--std=c++17", "--generate-line-info"}
	a.Dependencies = append(a.Dependencies, cudakernel.Dependency{Path: "cudablock_topk.cu", Digest: hex.EncodeToString(sum[:]), Tracked: true})
	return a, nil
}

func topkLoadArtifact(dev *cuda.Device, config TopKConfig, artifact *cudakernel.Artifact) (*TopK, error) {
	if err := artifact.Validate(); err != nil {
		return nil, err
	}
	t := &TopK{config: config, device: dev, artifact: artifact}
	var err error
	t.raw, err = t.artifact.Load(dev)
	if err != nil {
		return t, fmt.Errorf("cudablock: load topk: %w", err)
	}
	t.module, err = cuda.OwnModule(dev, t.raw)
	if err != nil {
		return t, fmt.Errorf("cudablock: own topk module: %w", err)
	}
	t.raw = 0
	t.kernel, err = t.module.Function("topk_fp32")
	if err != nil {
		return t, fmt.Errorf("cudablock: resolve topk: %w", err)
	}
	return t, nil
}

// Artifact returns a detached descriptor for the exact loaded image and ABI.
func (t *TopK) Artifact() *cudakernel.Artifact {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.artifact == nil {
		return nil
	}
	a := *t.artifact
	a.Raw = append([]byte(nil), a.Raw...)
	a.Schema.Params = append([]cudakernel.Param(nil), a.Schema.Params...)
	a.CompilerOpts = append([]string(nil), a.CompilerOpts...)
	a.Dependencies = append([]cudakernel.Dependency(nil), a.Dependencies...)
	return &a
}

// Execute completes selection, validates every row's device status and count,
// then publishes indices/counts and optional values on the same stream. It never
// uses a CPU sorting path. An error before publication leaves output untouched.
// After any device completion failure the buffers remain borrowed; retry Close
// to drain them. Cancellation is not offered by this synchronous initial API.
func (t *TopK) Execute(stream *cuda.Stream, b TopKBuffers) error {
	if t == nil {
		return fmt.Errorf("cudablock: closed topk")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.kernel == nil || t.pendingStream != nil {
		return fmt.Errorf("cudablock: topk closed or awaiting drain")
	}
	if stream == nil || stream.Device() != t.device {
		return fmt.Errorf("cudablock: topk stream device mismatch")
	}
	releases, err := t.borrow(b)
	if err != nil {
		return err
	}
	t.pendingRelease, t.pendingStream = releases, stream
	completed, err := topkInvoke(t.config, func() error { return t.launch(stream, b) }, func() error { return stream.Synchronize(nil) }, func() ([]int32, []int32, error) {
		status, counts := make([]int32, t.config.Rows), make([]int32, t.config.Rows)
		if err := b.Status.CopyToHost(status); err != nil {
			return nil, nil, err
		}
		if err := b.ScratchCounts.CopyToHost(counts); err != nil {
			return nil, nil, err
		}
		return status, counts, nil
	}, func() error {
		if err := b.Indices.CopyDeviceToDeviceAsync(b.ScratchIndices, stream); err != nil {
			return err
		}
		if err := b.Counts.CopyDeviceToDeviceAsync(b.ScratchCounts, stream); err != nil {
			return err
		}
		if t.config.Values {
			return b.Values.CopyDeviceToDeviceAsync(b.ScratchValues, stream)
		}
		return nil
	})
	if completed {
		t.release()
	}
	if err != nil {
		return fmt.Errorf("cudablock: execute topk: %w", err)
	}
	return nil
}

func topkInvoke(c TopKConfig, launch, wait func() error, read func() ([]int32, []int32, error), publish func() error) (completed bool, err error) {
	launchErr := launch()
	if err := wait(); err != nil {
		return false, errors.Join(launchErr, err)
	}
	if launchErr != nil {
		return true, launchErr
	}
	status, counts, err := read()
	if err != nil {
		return true, err
	}
	if len(status) != c.Rows || len(counts) != c.Rows {
		return true, fmt.Errorf("missing topk row evidence")
	}
	if err := CheckTopKStatus(status); err != nil {
		return true, err
	}
	for row, n := range counts {
		if n < 0 || int(n) > c.K {
			return true, fmt.Errorf("invalid topk row %d count %d", row, n)
		}
	}
	publishErr := publish()
	if err := wait(); err != nil {
		return false, errors.Join(publishErr, err)
	}
	return true, publishErr
}

func (t *TopK) launch(stream *cuda.Stream, b TopKBuffers) error {
	ptrs := []driver.DevicePtr{b.Input.Ptr(), b.Lengths.Ptr(), b.Offsets.Ptr(), b.ScratchIndices.Ptr(), b.ScratchValues.Ptr(), b.ScratchCounts.Ptr(), b.Status.Ptr()}
	stride := uint64(t.config.RowStride)
	width, k, rows, values := int32(t.config.Width), int32(t.config.K), int32(t.config.Rows), int32(0)
	if t.config.Values {
		values = 1
	}
	args := make([]unsafe.Pointer, 0, 12)
	for i := range ptrs {
		args = append(args, unsafe.Pointer(&ptrs[i]))
	}
	args = append(args, unsafe.Pointer(&stride), unsafe.Pointer(&width), unsafe.Pointer(&k), unsafe.Pointer(&rows), unsafe.Pointer(&values))
	return stream.WithHandle(func(handle driver.Stream) error {
		return t.kernel.Launch([3]uint32{uint32(rows), 1, 1}, [3]uint32{1, 1, 1}, 0, handle, args)
	})
}

type topkBuffer struct {
	name         string
	ptr          uint64
	length, need int
	owner        *cuda.Device
	borrow       func(*cuda.Device) (func(), error)
}

func (t *TopK) borrow(b TopKBuffers) ([]func(), error) {
	e, err := t.config.Extents()
	if err != nil {
		return nil, err
	}
	floatBuf := func(name string, s cudamem.Slice[float32], n int) topkBuffer {
		return topkBuffer{name, uint64(s.Ptr()), s.Len(), n, s.Device(), s.Borrow}
	}
	intBuf := func(name string, s cudamem.Slice[int32], n int) topkBuffer {
		return topkBuffer{name, uint64(s.Ptr()), s.Len(), n, s.Device(), s.Borrow}
	}
	v := 0
	if t.config.Values {
		v = e.Results
	}
	bufs := []topkBuffer{floatBuf("input", b.Input, e.Input), intBuf("lengths", b.Lengths, e.RowMetadata), intBuf("offsets", b.Offsets, e.RowMetadata), intBuf("scratch indices", b.ScratchIndices, e.Results), floatBuf("scratch values", b.ScratchValues, v), intBuf("scratch counts", b.ScratchCounts, e.RowMetadata), intBuf("status", b.Status, e.RowMetadata), intBuf("indices", b.Indices, e.Results), floatBuf("values", b.Values, v), intBuf("counts", b.Counts, e.RowMetadata)}
	if err := topkCheckBuffers(bufs, t.device); err != nil {
		return nil, err
	}
	var releases []func()
	for _, b := range bufs {
		if b.need == 0 {
			continue
		}
		release, err := b.borrow(t.device)
		if err != nil {
			for _, f := range releases {
				f()
			}
			return nil, err
		}
		releases = append(releases, release)
	}
	// Recheck after borrowing: Free may have raced with the initial admission.
	if err := topkCheckBuffers(bufs, t.device); err != nil {
		for _, f := range releases {
			f()
		}
		return nil, err
	}
	return releases, nil
}

func topkCheckBuffers(bufs []topkBuffer, dev *cuda.Device) error {
	for i, b := range bufs {
		if b.need == 0 {
			if b.length != 0 {
				return fmt.Errorf("cudablock: topk %s must be empty", b.name)
			}
			continue
		}
		if b.length != b.need || b.ptr == 0 || b.ptr%4 != 0 || b.owner != dev || b.ptr > ^uint64(0)-uint64(b.need)*4 {
			return fmt.Errorf("cudablock: invalid topk %s buffer", b.name)
		}
		for _, a := range bufs[:i] {
			if a.need > 0 && a.ptr < b.ptr+uint64(b.need)*4 && b.ptr < a.ptr+uint64(a.need)*4 {
				return fmt.Errorf("cudablock: topk buffers %s and %s alias", a.name, b.name)
			}
		}
	}
	return nil
}

func (t *TopK) release() {
	for _, f := range t.pendingRelease {
		f()
	}
	t.pendingRelease, t.pendingStream = nil, nil
}

// Close drains a failed invocation before releasing its borrows and unloading
// the module. Failed drains or cleanup retain ownership for a later retry.
func (t *TopK) Close() error {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.pendingStream != nil {
		if err := t.pendingStream.Synchronize(nil); err != nil {
			return fmt.Errorf("cudablock: drain topk: %w", err)
		}
		t.release()
	}
	if t.module != nil {
		if err := t.module.Close(); err != nil {
			return fmt.Errorf("cudablock: close topk: %w", err)
		}
		t.module, t.kernel = nil, nil
	}
	if t.raw != 0 {
		if err := t.device.WithContext(func() error {
			if err := t.raw.Close(); err != nil {
				return err
			}
			t.raw = 0
			return nil
		}); err != nil {
			return fmt.Errorf("cudablock: close raw topk: %w", err)
		}
	}
	if t.cleanup != nil {
		if err := t.cleanup.Close(); err != nil {
			return fmt.Errorf("cudablock: close topk compiler: %w", err)
		}
		t.cleanup = nil
	}
	return nil
}
