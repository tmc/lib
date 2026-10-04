package cudareduce

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"regexp"
	"strings"
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

// SparseTeacherObjective declares the omitted teacher probability policy.
type SparseTeacherObjective int

const (
	// NormalizedSupport normalizes nonnegative retained weights to total one.
	// Student normalization and gradients still span the full vocabulary.
	NormalizedSupport SparseTeacherObjective = iota
	// PreservedTail preserves supplied probabilities and models the omitted
	// vocabulary as one coarse bucket. Only temperature one is supported.
	PreservedTail
)

// SparseTeacherConfig admits contiguous FP32 logits and targets. Losses are
// per row and Total is their sum; there is no averaging, T squared scaling,
// generation-temperature correction, or gradient accumulation. Zero weights
// use the convention 0*log(0)=0. All inputs, including masked rows, are checked.
// The finite domain rejects every nonfinite logit, including negative infinity,
// scaled-logit overflow and nonfinite loss/gradient. Underflowed probabilities
// are allowed; log probabilities are evaluated from LSE without clamping.
type SparseTeacherConfig struct {
	Rows, Vocabulary, Retained int
	Objective                  SparseTeacherObjective
	Temperature                float32
}

// SparseTeacherExtents are exact element counts. No dense teacher is allocated.
type SparseTeacherExtents struct{ Logits, Targets, Rows, Stats int }

// Extents checks shapes, bytes and bounded launch geometry before native calls.
func (c SparseTeacherConfig) Extents() (SparseTeacherExtents, error) {
	if c.Rows < 1 || c.Rows > 65535 || c.Vocabulary < 1 || c.Vocabulary > 1<<20 || c.Retained < 1 || c.Retained > min(c.Vocabulary, 1024) {
		return SparseTeacherExtents{}, fmt.Errorf("cudareduce: invalid sparse teacher shape")
	}
	if !distillFinite(c.Temperature) || c.Temperature <= 0 || (c.Objective != NormalizedSupport && c.Objective != PreservedTail) || (c.Objective == PreservedTail && c.Temperature != 1) {
		return SparseTeacherExtents{}, fmt.Errorf("cudareduce: invalid sparse teacher objective or temperature")
	}
	// Bound every count to signed32 indexing and every four-byte extent to int.
	limit := min(int64(math.MaxInt32), int64(^uint(0)>>1)/4)
	n, k := int64(c.Rows)*int64(c.Vocabulary), int64(c.Rows)*int64(c.Retained)
	if n > limit || k > limit || int64(c.Rows)*4 > limit {
		return SparseTeacherExtents{}, fmt.Errorf("cudareduce: sparse teacher extent overflow")
	}
	return SparseTeacherExtents{int(n), int(k), c.Rows, c.Rows * 4}, nil
}
func distillFinite(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }

// ValidateSparseTeacher checks host input arrays. Resident device arrays are
// independently checked by a completed validation dispatch in Execute.
func ValidateSparseTeacher(c SparseTeacherConfig, z []float32, ids []int32, w []float32, masks []int32) error {
	e, err := c.Extents()
	if err != nil {
		return err
	}
	if len(z) != e.Logits || len(ids) != e.Targets || len(w) != e.Targets || len(masks) != e.Rows {
		return fmt.Errorf("cudareduce: sparse teacher host extents disagree")
	}
	for row := 0; row < c.Rows; row++ {
		if masks[row] != 0 && masks[row] != 1 {
			return fmt.Errorf("cudareduce: invalid sparse teacher mask at row %d", row)
		}
		for _, x := range z[row*c.Vocabulary : (row+1)*c.Vocabulary] {
			if !distillFinite(x) || !distillFinite(x/c.Temperature) {
				return fmt.Errorf("cudareduce: nonfinite sparse teacher student at row %d", row)
			}
		}
		mass := 0.0
		seen := make(map[int32]bool, c.Retained)
		for i := row * c.Retained; i < (row+1)*c.Retained; i++ {
			if ids[i] < 0 || int(ids[i]) >= c.Vocabulary || seen[ids[i]] || !distillFinite(w[i]) || w[i] < 0 {
				return fmt.Errorf("cudareduce: invalid sparse teacher target at row %d", row)
			}
			seen[ids[i]] = true
			mass += float64(w[i])
		}
		if c.Objective == NormalizedSupport && mass <= 0 || c.Objective == PreservedTail && (mass > 1 || c.Retained == c.Vocabulary && mass != 1) {
			return fmt.Errorf("cudareduce: invalid sparse teacher mass at row %d", row)
		}
	}
	return nil
}

// SparseTeacherBuffers contains exact-sized caller allocations. Every range
// must be disjoint, owned by the stream's device, and four-byte aligned.
// Stats has four values per row; scratch losses/gradients/total are not admitted
// output. Inputs, outputs, scratch, stream and device remain live and unchanged
// until Execute succeeds or Close successfully drains a failed invocation.
// Borrowing prevents Free; callers exclude concurrent writes, recycling and
// stream/device Close. Outputs are overwritten only after validation succeeds.
// A partial publication failure leaves outputs undefined.
type SparseTeacherBuffers struct {
	Logits, Weights                                      cudamem.Slice[float32]
	IDs, Masks, Status                                   cudamem.Slice[int32]
	Stats, ScratchLosses, ScratchGradients, ScratchTotal cudamem.Slice[float32]
	Losses, Gradients, Total                             cudamem.Slice[float32]
}

// SparseTeacher owns the loaded ordinary CUDA baseline. The zero value is
// closed. Methods serialize; do not copy an owner. A nonnil error-result owner
// must be closed before its device. There is no CPU execution fallback.
type SparseTeacher struct {
	mu         sync.Mutex
	config     SparseTeacherConfig
	device     *cuda.Device
	artifact   *cudakernel.Artifact
	module     *cuda.Module
	raw        driver.Module
	kernel     *cuda.Kernel
	cleanup    io.Closer
	pending    *cudaplan.Completion
	stream     *cuda.Stream
	releases   []func()
	beforeWait func() // Native control may cancel only the host wait after submit.
}

// CompileSparseTeacherArtifact compiles a direct CUBIN without initializing a
// device. target is an explicit physical architecture such as sm_121. NVRTC's
// existing loader and toolkit include discovery select the compiler/headers.
// An error exposing io.Closer retains compiler cleanup ownership; the caller
// must close it. CompileSparseTeacher adopts that owner in its error result.
func CompileSparseTeacherArtifact(target string) (*cudakernel.Artifact, error) {
	return distillCompileSource(target, SparseTeacherSource)
}
func distillCompileSource(target, source string) (*cudakernel.Artifact, error) {
	if !regexp.MustCompile(`^sm_[1-9][0-9]{1,2}$`).MatchString(target) {
		return nil, fmt.Errorf("cudareduce: invalid sparse teacher physical target")
	}
	header, err := distillBlockHeader()
	if err != nil {
		return nil, err
	}
	cm, err := nvrtcx.Compile(source, nvrtcx.WithName("cudareduce_distill.cu"), nvrtcx.WithPhysicalArchTarget(target), nvrtcx.WithHeader("cudablock.cuh", header), nvrtcx.WithFlag("--std=c++17"), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithFlag("--ftz=false"), nvrtcx.WithNoCache())
	if err != nil {
		return nil, err
	}
	schema, err := SparseTeacherSchema()
	if err != nil {
		return nil, err
	}
	a, err := cm.ToArtifact("sparse_teacher", schema, target)
	if err != nil {
		return nil, err
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		return nil, err
	}
	a.Toolchain = fmt.Sprintf("nvrtc %d.%d", major, minor)
	a.CompilerOpts = []string{"--gpu-architecture=" + target, "--std=c++17", "--fmad=false", "--ftz=false"}
	a.Dependencies = append(a.Dependencies, cudakernel.Dependency{Path: "cudareduce_distill.cu", Digest: distillDigest([]byte(source)), Tracked: true})
	a.Dependencies = append(a.Dependencies, cudakernel.Dependency{Path: "cudablock.cuh", Digest: distillDigest([]byte(header)), Tracked: true}, cudakernel.Dependency{Path: "cudablock.owner.cuh", Digest: distillDigest([]byte(cudablock.Header)), Tracked: true})
	return a, nil
}

// Include the owner's unchanged FP32 collectives without its unrelated half,
// tile and vector helpers. Both the used bytes and the full owner header are
// bound in the artifact; an upstream layout change refuses rather than guesses.
func distillBlockHeader() (string, error) {
	return distillTrimBlockHeader(cudablock.Header)
}
func distillTrimBlockHeader(full string) (string, error) {
	const boundary = "// Block-level cooperative 2D tile loader"
	for _, form := range []string{boundary, "#include <cuda_fp16.h>", "#include <cuda_bf16.h>", "#ifndef CUDABLOCK_CUH", "#define CUDABLOCK_CUH", "T block_reduce_sum(", "T block_reduce_max(", "int warp_lane_count("} {
		if strings.Count(full, form) != 1 {
			return "", fmt.Errorf("cudareduce: sparse teacher collective header form changed")
		}
	}
	if strings.Count(full, boundary) != 1 {
		return "", fmt.Errorf("cudareduce: sparse teacher collective header boundary changed")
	}
	header := strings.Split(full, boundary)[0]
	header = strings.ReplaceAll(header, "#include <cuda_fp16.h>", "")
	header = strings.ReplaceAll(header, "#include <cuda_bf16.h>", "")
	if strings.Contains(header, "#include") || strings.Contains(header, "#endif") {
		return "", fmt.Errorf("cudareduce: unexpected sparse teacher collective include or guard")
	}
	return header + "\n#endif\n", nil
}
func distillDigest(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

// CompileSparseTeacher compiles for dev and loads the ordinary baseline.
func CompileSparseTeacher(dev *cuda.Device, c SparseTeacherConfig) (*SparseTeacher, error) {
	if _, err := c.Extents(); err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, fmt.Errorf("cudareduce: nil sparse teacher device")
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, err
	}
	a, err := CompileSparseTeacherArtifact(fmt.Sprintf("sm_%d%d", major, minor))
	if err != nil {
		owner := &SparseTeacher{device: dev}
		if errors.As(err, &owner.cleanup) {
			return owner, err
		}
		return nil, err
	}
	return LoadSparseTeacher(dev, c, a)
}

// LoadSparseTeacher checks retained image, schema, source and header bindings
// before loading. These byte/metadata checks do not authenticate a compiler or
// prove the semantics of arbitrary externally supplied machine code.
func LoadSparseTeacher(dev *cuda.Device, c SparseTeacherConfig, a *cudakernel.Artifact) (*SparseTeacher, error) {
	return distillLoadSource(dev, c, a, SparseTeacherSource)
}
func distillAdmitArtifact(a *cudakernel.Artifact, source string) error {
	if a == nil {
		return fmt.Errorf("cudareduce: missing sparse teacher artifact")
	}
	if err := a.Validate(); err != nil {
		return err
	}
	schema, _ := SparseTeacherSchema()
	if a.Kind != cudakernel.KindCUBIN || a.EntryPoint != "sparse_teacher" || !reflect.DeepEqual(a.Schema, schema) || a.Toolchain == "" || a.JITOptions != (cudakernel.JITOptions{}) || !regexp.MustCompile(`^sm_[1-9][0-9]{1,2}$`).MatchString(a.Target) {
		return fmt.Errorf("cudareduce: sparse teacher artifact contract disagrees")
	}
	header, err := distillBlockHeader()
	if err != nil {
		return err
	}
	required := map[string]string{"cudareduce_distill.cu": distillDigest([]byte(source)), "cudablock.cuh": distillDigest([]byte(header)), "cudablock.owner.cuh": distillDigest([]byte(cudablock.Header))}
	for _, dep := range a.Dependencies {
		if want, ok := required[dep.Path]; ok {
			if !dep.Tracked || dep.ErrorMsg != "" || dep.Digest != want {
				return fmt.Errorf("cudareduce: sparse teacher dependency disagrees")
			}
			delete(required, dep.Path)
		} else {
			return fmt.Errorf("cudareduce: duplicate or unexpected sparse teacher dependency")
		}
	}
	if len(required) != 0 {
		return fmt.Errorf("cudareduce: missing sparse teacher source binding")
	}
	if !reflect.DeepEqual(a.CompilerOpts, []string{"--gpu-architecture=" + a.Target, "--std=c++17", "--fmad=false", "--ftz=false"}) {
		return fmt.Errorf("cudareduce: sparse teacher compiler mode disagrees")
	}
	return nil
}
func distillClone(a *cudakernel.Artifact) *cudakernel.Artifact {
	b := *a
	b.Raw = append([]byte(nil), a.Raw...)
	b.Schema.Params = append([]cudakernel.Param(nil), a.Schema.Params...)
	b.CompilerOpts = append([]string(nil), a.CompilerOpts...)
	b.Dependencies = append([]cudakernel.Dependency(nil), a.Dependencies...)
	return &b
}
func distillLoadSource(dev *cuda.Device, c SparseTeacherConfig, a *cudakernel.Artifact, source string) (*SparseTeacher, error) {
	if _, err := c.Extents(); err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, fmt.Errorf("cudareduce: nil sparse teacher device")
	}
	if err := distillAdmitArtifact(a, source); err != nil {
		return nil, err
	}
	o := &SparseTeacher{device: dev, config: c, artifact: distillClone(a)}
	var err error
	o.raw, err = o.artifact.Load(dev)
	if err != nil {
		return o, err
	}
	o.module, err = cuda.OwnModule(dev, o.raw)
	if err != nil {
		return o, err
	}
	o.raw = 0
	o.kernel, err = o.module.Function(a.EntryPoint)
	if err != nil {
		return o, err
	}
	limit, err := o.kernel.Attribute(driver.FuncAttributeMaxThreadsPerBlock)
	if err != nil {
		return o, err
	}
	if limit < 32 {
		return o, fmt.Errorf("cudareduce: sparse teacher kernel admits fewer than 32 threads")
	}
	return o, nil
}

// Artifact returns a detached descriptor, or nil for a closed owner.
func (o *SparseTeacher) Artifact() *cudakernel.Artifact {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.artifact == nil {
		return nil
	}
	return distillClone(o.artifact)
}

type distillView struct {
	name         string
	ptr          driver.DevicePtr
	length, want int
	device       *cuda.Device
	borrow       func(*cuda.Device) (func(), error)
}

func distillBuffer[T any](name string, s cudamem.Slice[T], want int) distillView {
	return distillView{name, s.Ptr(), s.Len(), want, s.Device(), s.Borrow}
}
func distillViews(c SparseTeacherConfig, b SparseTeacherBuffers) []distillView {
	e, _ := c.Extents()
	return []distillView{
		distillBuffer("logits", b.Logits, e.Logits), distillBuffer("weights", b.Weights, e.Targets), distillBuffer("ids", b.IDs, e.Targets), distillBuffer("masks", b.Masks, e.Rows), distillBuffer("status", b.Status, e.Rows), distillBuffer("stats", b.Stats, e.Stats), distillBuffer("scratch losses", b.ScratchLosses, e.Rows), distillBuffer("scratch gradients", b.ScratchGradients, e.Logits), distillBuffer("scratch total", b.ScratchTotal, 1), distillBuffer("losses", b.Losses, e.Rows), distillBuffer("gradients", b.Gradients, e.Logits), distillBuffer("total", b.Total, 1)}
}
func distillCheckViews(v []distillView, dev *cuda.Device) error {
	if dev == nil {
		return fmt.Errorf("cudareduce: nil sparse teacher buffer device")
	}
	for i, x := range v {
		if x.length != x.want || x.ptr == 0 || uint64(x.ptr)%4 != 0 || x.device != dev || x.want < 1 || uint64(x.ptr) > math.MaxUint64-uint64(x.want)*4 {
			return fmt.Errorf("cudareduce: invalid sparse teacher %s", x.name)
		}
		end := uint64(x.ptr) + uint64(x.want)*4
		for _, y := range v[:i] {
			if uint64(x.ptr) < uint64(y.ptr)+uint64(y.want)*4 && uint64(y.ptr) < end {
				return fmt.Errorf("cudareduce: aliased sparse teacher buffers")
			}
		}
	}
	return nil
}
func distillBorrow(v []distillView, dev *cuda.Device) ([]func(), error) {
	if err := distillCheckViews(v, dev); err != nil {
		return nil, err
	}
	var releases []func()
	unwind := func() {
		for _, f := range releases {
			f()
		}
	}
	for _, x := range v {
		release, err := x.borrow(dev)
		if err != nil {
			unwind()
			return nil, err
		}
		releases = append(releases, release)
	}
	// Callers exclude concurrent mutation; retain rechecks catch owner changes.
	for _, x := range v {
		if x.device != dev {
			unwind()
			return nil, fmt.Errorf("cudareduce: sparse teacher owner changed")
		}
	}
	return releases, nil
}

// CheckSparseTeacherStatus requires one successful device write per row.
var errDistillStatus = errors.New("cudareduce: sparse teacher device status refusal")

func CheckSparseTeacherStatus(status []int32, rows int) error {
	if rows < 1 || len(status) != rows {
		return fmt.Errorf("cudareduce: sparse teacher status extent disagrees")
	}
	for row, s := range status {
		if s != 0 {
			return fmt.Errorf("%w: row %d status %d", errDistillStatus, row, s)
		}
	}
	return nil
}

// Execute validates resident input, computes scratch, sums per-row losses, and
// publishes loss/full gradient/total after successful checks. It synchronizes
// each dispatch through the existing stream completion contract. Cancellation
// or failed drain retains every borrow for Close; scratch cannot be reused.
func (o *SparseTeacher) Execute(ctx context.Context, stream *cuda.Stream, b SparseTeacherBuffers) error {
	if o == nil {
		return fmt.Errorf("cudareduce: closed sparse teacher")
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.kernel == nil || o.device == nil {
		return fmt.Errorf("cudareduce: closed sparse teacher")
	}
	if len(o.releases) != 0 {
		return fmt.Errorf("cudareduce: incomplete sparse teacher invocation")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if stream == nil || stream.Device() != o.device {
		return fmt.Errorf("cudareduce: sparse teacher stream owner disagrees")
	}
	views := distillViews(o.config, b)
	releases, err := distillBorrow(views, o.device)
	if err != nil {
		return err
	}
	if err := distillCheckViews(distillViews(o.config, b), o.device); err != nil {
		for _, f := range releases {
			f()
		}
		return err
	}
	o.releases, o.stream = releases, stream
	defer func() {
		if o.pending != nil && o.pending.Done() {
			o.release()
		}
	}()
	for stage := int32(0); stage < 3; stage++ {
		o.pending = nil
		poison := make([]int32, o.config.Rows)
		for i := range poison {
			poison[i] = -1
		}
		if err := b.Status.CopyFromHost(poison); err != nil {
			return err
		}
		launchErr := o.launch(b, stage)
		drainErr := o.drain(ctx)
		if err := errors.Join(launchErr, drainErr); err != nil {
			return err
		}
		s := make([]int32, o.config.Rows)
		if err := b.Status.CopyToHost(s); err != nil {
			return err
		}
		if err := CheckSparseTeacherStatus(s, o.config.Rows); err != nil {
			return err
		}
	}
	o.pending = nil
	err = b.Losses.CopyDeviceToDeviceAsync(b.ScratchLosses, stream)
	if err == nil {
		err = b.Gradients.CopyDeviceToDeviceAsync(b.ScratchGradients, stream)
	}
	if err == nil {
		err = b.Total.CopyDeviceToDeviceAsync(b.ScratchTotal, stream)
	}
	return errors.Join(err, o.drain(ctx))
}
func (o *SparseTeacher) release() {
	for _, f := range o.releases {
		f()
	}
	o.releases = nil
	o.stream = nil
	o.pending = nil
}
func (o *SparseTeacher) drain(ctx context.Context) error {
	var err error
	o.pending, err = cudaplan.NewStreamCompletion(o.stream)
	if err != nil {
		return err
	}
	if o.beforeWait != nil {
		o.beforeWait()
	}
	return o.pending.Wait(ctx)
}
func distillArguments(schema cudakernel.Schema, c SparseTeacherConfig, p [9]driver.DevicePtr, stage int32) ([]unsafe.Pointer, error) {
	want, _ := SparseTeacherSchema()
	if !reflect.DeepEqual(schema, want) {
		return nil, fmt.Errorf("cudareduce: sparse teacher launch ABI disagrees")
	}
	if _, err := c.Extents(); err != nil {
		return nil, err
	}
	if stage < 0 || stage > 2 {
		return nil, fmt.Errorf("cudareduce: invalid sparse teacher stage")
	}
	var args []unsafe.Pointer
	for i := range p {
		if p[i] == 0 || uint64(p[i])%4 != 0 {
			return nil, fmt.Errorf("cudareduce: invalid sparse teacher pointer")
		}
		args = append(args, unsafe.Pointer(&p[i]))
	}
	ints := []int32{int32(c.Rows), int32(c.Vocabulary), int32(c.Retained), int32(c.Objective)}
	for i := range ints {
		args = append(args, unsafe.Pointer(&ints[i]))
	}
	temperature := c.Temperature
	args = append(args, unsafe.Pointer(&temperature), unsafe.Pointer(&stage))
	return args, nil
}
func (o *SparseTeacher) launch(b SparseTeacherBuffers, stage int32) error {
	args, err := distillArguments(o.artifact.Schema, o.config, [9]driver.DevicePtr{b.Logits.Ptr(), b.IDs.Ptr(), b.Weights.Ptr(), b.Masks.Ptr(), b.Status.Ptr(), b.Stats.Ptr(), b.ScratchLosses.Ptr(), b.ScratchGradients.Ptr(), b.ScratchTotal.Ptr()}, stage)
	if err != nil {
		return err
	}
	rows := o.config.Rows
	if stage == 2 {
		rows = 1
	}
	return o.stream.WithHandle(func(h driver.Stream) error {
		return o.kernel.Launch([3]uint32{uint32(rows), 1, 1}, [3]uint32{32, 1, 1}, 0, h, args)
	})
}

// Close drains incomplete work without publishing scratch, then unloads the
// module. Failed drains or cleanup retain ownership and may be retried.
func (o *SparseTeacher) Close() error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(o.releases) != 0 {
		if err := o.drain(context.Background()); err != nil {
			return err
		}
		o.release()
	}
	if o.cleanup != nil {
		if err := o.cleanup.Close(); err != nil {
			return err
		}
		o.cleanup = nil
	}
	if o.module != nil {
		if err := o.module.Close(); err != nil {
			return err
		}
		o.module = nil
	}
	if o.raw != 0 {
		if err := o.device.WithContext(func() error { return o.raw.Close() }); err != nil {
			return err
		}
		o.raw = 0
	}
	o.kernel = nil
	o.artifact = nil
	o.device = nil
	return nil
}
