package cudaattn

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// KDAFP32 names the preactivated, per-key-channel, separately rounded FP32
// numerical mode. It does not identify upstream FlashKDA's BF16 chunk mode.
const KDAFP32 = "kda-d128-fp32-rn-separate-vk-v1"

// KDAConfig bounds one ordinary recurrence invocation. Dimension must be 128,
// Heads and Sequences 1 or 2, Tokens 0 through 66, and each sequence at most 33
// tokens. Scale must be positive and finite. Empty segments preserve their state.
type KDAConfig struct {
	Dimension, Heads, Sequences, Tokens int
	Scale                               float32
}

// KDAExtents contains exact scalar counts, including separate scratch results.
type KDAExtents struct{ Vectors, Betas, State, Offsets, WorkspaceBytes int }

// Extents validates the bounded shape and returns checked allocation counts.
func (c KDAConfig) Extents() (KDAExtents, error) {
	if c.Dimension != 128 || c.Heads < 1 || c.Heads > 2 || c.Sequences < 1 || c.Sequences > 2 || c.Tokens < 0 || c.Tokens > 66 || !kdaFinite(c.Scale) || c.Scale <= 0 {
		return KDAExtents{}, fmt.Errorf("cudaattn: unsupported kda configuration")
	}
	e := KDAExtents{Vectors: c.Tokens * c.Heads * 128, Betas: c.Tokens * c.Heads, State: c.Sequences * c.Heads * 128 * 128, Offsets: c.Sequences + 1}
	e.WorkspaceBytes = 4 * (e.State + e.Vectors + e.Offsets + 1)
	return e, nil
}

func (c KDAConfig) checkOffsets(offsets []int32) error {
	if _, err := c.Extents(); err != nil {
		return err
	}
	if len(offsets) != c.Sequences+1 || offsets[0] != 0 || offsets[len(offsets)-1] != int32(c.Tokens) {
		return fmt.Errorf("cudaattn: invalid kda terminal offsets")
	}
	for i := 0; i < c.Sequences; i++ {
		if offsets[i] < 0 || offsets[i+1] < offsets[i] || offsets[i+1] > int32(c.Tokens) || offsets[i+1]-offsets[i] > 33 {
			return fmt.Errorf("cudaattn: invalid kda sequence %d offsets", i)
		}
	}
	return nil
}

func kdaFinite(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }

func kdaCheckNumerics(c KDAConfig, q, key, value, decay, beta, state []float32) error {
	e, err := c.Extents()
	if err != nil {
		return err
	}
	for _, b := range []struct {
		name string
		data []float32
		size int
	}{{"q", q, e.Vectors}, {"key", key, e.Vectors}, {"value", value, e.Vectors}, {"decay", decay, e.Vectors}, {"beta", beta, e.Betas}, {"state", state, e.State}} {
		if len(b.data) != b.size {
			return fmt.Errorf("cudaattn: kda %s size mismatch", b.name)
		}
		for _, x := range b.data {
			if !kdaFinite(x) {
				return fmt.Errorf("cudaattn: nonfinite kda %s", b.name)
			}
		}
	}
	for _, a := range [][]float32{q, key} {
		for i := 0; i < len(a); i += 128 {
			norm := 0.0
			for _, x := range a[i : i+128] {
				norm += float64(x) * float64(x)
			}
			if math.Abs(norm-1) > 2e-5 {
				return fmt.Errorf("cudaattn: kda q/key must have unit squared norm within 2e-5")
			}
		}
	}
	for _, x := range decay {
		if x < 0 || x > 1 {
			return fmt.Errorf("cudaattn: kda decay outside [0,1]")
		}
	}
	for _, x := range beta {
		if x < 0 || x > 1 {
			return fmt.Errorf("cudaattn: kda beta outside [0,1]")
		}
	}
	return nil
}

// KDAIdentity binds a consumer state to its sequence, consumed prefix and
// numerical mode. Scale and head count are additionally bound by KDAState.
type KDAIdentity struct {
	Sequence string
	Prefix   uint64
	Mode     string
}

// KDAState is an immutable imported or successfully published [N,H,V,K] state.
// The caller must preserve its data and keep the device live until all users
// complete. Import establishes a caller assertion, not proof of data provenance.
type KDAState struct {
	data  cudamem.Slice[float32]
	ids   []KDAIdentity
	heads int
	scale float32
}

// NewKDAState imports an explicit initial snapshot without taking allocation
// ownership. Sequence names must be nonempty and distinct, with mode KDAFP32.
func NewKDAState(c KDAConfig, data cudamem.Slice[float32], ids []KDAIdentity) (*KDAState, error) {
	e, err := c.Extents()
	if err != nil {
		return nil, err
	}
	if data.Len() != e.State || data.Device() == nil {
		return nil, fmt.Errorf("cudaattn: invalid kda state allocation")
	}
	if err := kdaCheckIdentity(c, ids); err != nil {
		return nil, err
	}
	return &KDAState{data: data, ids: append([]KDAIdentity(nil), ids...), heads: c.Heads, scale: c.Scale}, nil
}

// Identities returns a detached copy of the published consumer identities.
func (s *KDAState) Identities() []KDAIdentity {
	if s == nil {
		return nil
	}
	return append([]KDAIdentity(nil), s.ids...)
}

// Data returns the immutable state allocation. The caller owns its allocation;
// writing or freeing it while an invocation uses it violates the contract.
func (s *KDAState) Data() cudamem.Slice[float32] {
	if s == nil {
		return cudamem.Slice[float32]{}
	}
	return s.data
}

func kdaCheckIdentity(c KDAConfig, ids []KDAIdentity) error {
	if len(ids) != c.Sequences {
		return fmt.Errorf("cudaattn: kda identity count mismatch")
	}
	for i, id := range ids {
		if id.Sequence == "" || id.Mode != KDAFP32 {
			return fmt.Errorf("cudaattn: invalid kda sequence/mode identity")
		}
		for j := 0; j < i; j++ {
			if ids[j].Sequence == id.Sequence {
				return fmt.Errorf("cudaattn: repeated kda sequence identity")
			}
		}
	}
	return nil
}

func kdaAdvance(c KDAConfig, s *KDAState, expected []KDAIdentity, offsets []int32) ([]KDAIdentity, error) {
	if err := c.checkOffsets(offsets); err != nil {
		return nil, err
	}
	if err := kdaCheckIdentity(c, expected); err != nil {
		return nil, err
	}
	if s == nil || s.heads != c.Heads || s.scale != c.Scale || len(s.ids) != len(expected) {
		return nil, fmt.Errorf("cudaattn: kda state numerical identity mismatch")
	}
	next := append([]KDAIdentity(nil), expected...)
	for i, id := range expected {
		if id != s.ids[i] {
			return nil, fmt.Errorf("cudaattn: kda sequence/prefix/mode mismatch")
		}
		n := uint64(offsets[i+1] - offsets[i])
		if id.Prefix > ^uint64(0)-n {
			return nil, fmt.Errorf("cudaattn: kda prefix overflow")
		}
		next[i].Prefix += n
	}
	return next, nil
}

// KDABuffers contains contiguous caller-owned inputs and separate publication
// destinations. No active ranges may overlap, including read-only inputs.
// Inputs and imported state must remain immutable during execution. Outputs
// may be consumed only after successful KDAResult.Wait. Failed publication
// leaves raw destinations undefined and never returns an admitted state.
type KDABuffers struct {
	Q, Key, Value, Decay, Beta cudamem.Slice[float32]
	FinalState, Output         cudamem.Slice[float32]
}

// KDA owns an ordinary recurrence artifact and bounded scratch. Its zero value
// is closed. Only one invocation may be pending; keep device and stream live
// and exclude other buffer users until its result completes. Close retries
// failed drains before releasing resources.
type KDA struct {
	mu                          sync.Mutex
	config                      KDAConfig
	dev                         *cuda.Device
	artifact                    *cudakernel.Artifact
	module                      *cuda.Module
	raw                         driver.Module
	kernel                      *cuda.Kernel
	scratchState, scratchOutput cudamem.Slice[float32]
	offsets, status             cudamem.Slice[int32]
	pending                     *KDAResult
	closed                      bool
	cleanup                     io.Closer
}

func kdaCompileArtifact(source, target string) (*cudakernel.Artifact, error) {
	cm, err := nvrtcx.Compile(source, nvrtcx.WithName("cudaattn_kda.cu"), nvrtcx.WithPhysicalArchTarget(target), nvrtcx.WithFlag("--std=c++17"), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithFlag("--ftz=false"), nvrtcx.WithNoCache())
	if err != nil {
		return nil, err
	}
	schema, err := KDASchema()
	if err != nil {
		return nil, err
	}
	a, err := cm.ToArtifact("kda_fp32", schema, target)
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
	a.Dependencies = append(a.Dependencies, cudakernel.Dependency{Path: "cudaattn_kda.cu", Digest: hex.EncodeToString(sum[:]), Tracked: true})
	return a, nil
}

// CompileKDA compiles the ordinary kernel for the actual device. A nonnil KDA
// returned with an error owns cleanup and must be closed. Geometry is one block,
// one thread, zero dynamic shared bytes; this baseline makes no speed claim.
func CompileKDA(dev *cuda.Device, c KDAConfig) (*KDA, error) {
	if _, err := c.Extents(); err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, fmt.Errorf("cudaattn: nil kda device")
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, err
	}
	a, err := kdaCompileArtifact(KDASource, fmt.Sprintf("sm_%d%d", major, minor))
	if err != nil {
		k := &KDA{dev: dev, config: c}
		if errors.As(err, &k.cleanup) {
			return k, err
		}
		return nil, err
	}
	return kdaLoadArtifact(dev, c, a)
}

func kdaLoadArtifact(dev *cuda.Device, c KDAConfig, a *cudakernel.Artifact) (*KDA, error) {
	e, err := c.Extents()
	if err != nil {
		return nil, err
	}
	if dev == nil || a == nil {
		return nil, fmt.Errorf("cudaattn: missing kda device/artifact")
	}
	if err := a.Validate(); err != nil {
		return nil, err
	}
	schema, err := KDASchema()
	if err != nil {
		return nil, err
	}
	if !reflect.DeepEqual(a.Schema, schema) || a.EntryPoint != "kda_fp32" {
		return nil, fmt.Errorf("cudaattn: kda artifact ABI mismatch")
	}
	k := &KDA{dev: dev, config: c, artifact: a}
	k.raw, err = a.Load(dev)
	if err != nil {
		return k, err
	}
	k.module, err = cuda.OwnModule(dev, k.raw)
	if err != nil {
		return k, err
	}
	k.raw = 0
	k.kernel, err = k.module.Function(a.EntryPoint)
	if err != nil {
		return k, err
	}
	k.scratchState, err = cudamem.Alloc[float32](dev, e.State)
	if err != nil {
		return k, err
	}
	if e.Vectors > 0 {
		k.scratchOutput, err = cudamem.Alloc[float32](dev, e.Vectors)
		if err != nil {
			return k, err
		}
	}
	k.offsets, err = cudamem.Alloc[int32](dev, e.Offsets)
	if err != nil {
		return k, err
	}
	k.status, err = cudamem.Alloc[int32](dev, 1)
	return k, err
}

// Artifact returns a detached descriptor of the loaded kernel image and ABI.
func (k *KDA) Artifact() *cudakernel.Artifact {
	if k == nil {
		return nil
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.artifact == nil {
		return nil
	}
	a := *k.artifact
	a.Raw = append([]byte(nil), a.Raw...)
	a.Schema.Params = append([]cudakernel.Param(nil), a.Schema.Params...)
	a.CompilerOpts = append([]string(nil), a.CompilerOpts...)
	a.Dependencies = append([]cudakernel.Dependency(nil), a.Dependencies...)
	return &a
}

type kdaBuffer struct {
	name         string
	ptr          uint64
	length, need int
	dev          *cuda.Device
	borrow       func(*cuda.Device) (func(), error)
}

func kdaCheckBuffers(bufs []kdaBuffer, dev *cuda.Device) error {
	if dev == nil {
		return fmt.Errorf("cudaattn: nil kda buffer device")
	}
	for i, b := range bufs {
		if b.length != b.need {
			return fmt.Errorf("cudaattn: kda %s extent mismatch", b.name)
		}
		if b.need == 0 {
			continue
		}
		if b.ptr == 0 || b.ptr%4 != 0 || b.dev != dev || b.ptr > ^uint64(0)-uint64(b.need)*4 {
			return fmt.Errorf("cudaattn: invalid kda %s allocation", b.name)
		}
		for _, a := range bufs[:i] {
			if a.need > 0 && a.ptr < b.ptr+uint64(b.need)*4 && b.ptr < a.ptr+uint64(a.need)*4 {
				return fmt.Errorf("cudaattn: kda %s/%s alias", a.name, b.name)
			}
		}
	}
	return nil
}

func kdaBorrowBuffers(bufs []kdaBuffer, dev *cuda.Device) ([]func(), error) {
	if err := kdaCheckBuffers(bufs, dev); err != nil {
		return nil, err
	}
	var releases []func()
	fail := func(err error) ([]func(), error) {
		for _, f := range releases {
			f()
		}
		return nil, err
	}
	for _, b := range bufs {
		if b.need == 0 {
			continue
		}
		release, err := b.borrow(dev)
		if err != nil {
			return fail(err)
		}
		releases = append(releases, release)
	}
	if err := kdaCheckBuffers(bufs, dev); err != nil {
		return fail(err)
	}
	return releases, nil
}

// Execute validates actual device inputs, identity and offsets before submitting
// the ordinary recurrence. It reads input data back solely for admission; all
// recurrence arithmetic runs in CUDA. Offsets are copied into owned device
// storage. A nonnil result returned with an error retains partial submission
// ownership; call Wait or Close until the stream drains successfully.
func (k *KDA) Execute(ctx context.Context, stream *cuda.Stream, initial *KDAState, expected []KDAIdentity, offsets []int32, b KDABuffers) (*KDAResult, error) {
	if k == nil {
		return nil, fmt.Errorf("cudaattn: closed kda")
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.closed || k.kernel == nil || (k.pending != nil && !k.pending.done.Load()) {
		return nil, fmt.Errorf("cudaattn: kda closed or pending")
	}
	if stream == nil || stream.Device() != k.dev {
		return nil, fmt.Errorf("cudaattn: kda stream device mismatch")
	}
	offsets = append([]int32(nil), offsets...)
	ids, err := kdaAdvance(k.config, initial, expected, offsets)
	if err != nil {
		return nil, err
	}
	e, _ := k.config.Extents()
	var bufs []kdaBuffer
	add := func(name string, s cudamem.Slice[float32], n int) {
		bufs = append(bufs, kdaBuffer{name, uint64(s.Ptr()), s.Len(), n, s.Device(), s.Borrow})
	}
	for _, a := range []struct {
		name string
		s    cudamem.Slice[float32]
		n    int
	}{{"q", b.Q, e.Vectors}, {"key", b.Key, e.Vectors}, {"value", b.Value, e.Vectors}, {"decay", b.Decay, e.Vectors}, {"beta", b.Beta, e.Betas}, {"initial", initial.data, e.State}, {"final", b.FinalState, e.State}, {"output", b.Output, e.Vectors}, {"scratch-state", k.scratchState, e.State}, {"scratch-output", k.scratchOutput, e.Vectors}} {
		add(a.name, a.s, a.n)
	}
	for _, a := range []struct {
		name string
		s    cudamem.Slice[int32]
		n    int
	}{{"offsets", k.offsets, e.Offsets}, {"status", k.status, 1}} {
		bufs = append(bufs, kdaBuffer{a.name, uint64(a.s.Ptr()), a.s.Len(), a.n, a.s.Device(), a.s.Borrow})
	}
	releases, err := kdaBorrowBuffers(bufs, k.dev)
	if err != nil {
		return nil, err
	}
	release := func() {
		for _, f := range releases {
			f()
		}
	}
	if err := stream.Synchronize(ctx); err != nil {
		release()
		return nil, err
	}
	host := make([][]float32, 6)
	for i, s := range []cudamem.Slice[float32]{b.Q, b.Key, b.Value, b.Decay, b.Beta, initial.data} {
		host[i] = make([]float32, s.Len())
		if s.Len() > 0 {
			if err := s.CopyToHost(host[i]); err != nil {
				release()
				return nil, err
			}
		}
	}
	if err := kdaCheckNumerics(k.config, host[0], host[1], host[2], host[3], host[4], host[5]); err != nil {
		release()
		return nil, err
	}
	if err := k.offsets.CopyFromHost(offsets); err != nil {
		release()
		return nil, err
	}
	if err := k.status.CopyFromHost([]int32{-1}); err != nil {
		release()
		return nil, err
	}
	args, err := kdaLaunchArguments(k.artifact.Schema, [10]driver.DevicePtr{b.Q.Ptr(), b.Key.Ptr(), b.Value.Ptr(), b.Decay.Ptr(), b.Beta.Ptr(), initial.data.Ptr(), k.offsets.Ptr(), k.scratchState.Ptr(), k.scratchOutput.Ptr(), k.status.Ptr()}, int32(k.config.Sequences), int32(k.config.Heads), k.config.Scale)
	if err != nil {
		release()
		return nil, err
	}
	r := &KDAResult{owner: k, stream: stream, buffers: b, ids: ids, releases: releases}
	k.pending = r
	r.submitErr = stream.WithHandle(func(handle driver.Stream) error {
		return k.kernel.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, handle, args)
	})
	r.compute, err = cudaplan.NewStreamCompletion(stream)
	return r, errors.Join(r.submitErr, err)
}

// kdaLaunchArguments validates the parameter ABI, then supplies addresses of
// typed host values to cuLaunchKernel. A pointer parameter is eight bytes;
// its pointed-to FP32/int32 allocation needs only four-byte alignment.
func kdaLaunchArguments(schema cudakernel.Schema, pointers [10]driver.DevicePtr, sequences, heads int32, scale float32) ([]unsafe.Pointer, error) {
	want, err := KDASchema()
	if err != nil {
		return nil, err
	}
	if !reflect.DeepEqual(schema, want) {
		return nil, fmt.Errorf("cudaattn: kda launch ABI mismatch")
	}
	args := make([]unsafe.Pointer, 0, 13)
	for i := range pointers {
		if uint64(pointers[i])%4 != 0 {
			return nil, fmt.Errorf("cudaattn: kda pointer %d is not four-byte aligned", i)
		}
		args = append(args, unsafe.Pointer(&pointers[i]))
	}
	return append(args, unsafe.Pointer(&sequences), unsafe.Pointer(&heads), unsafe.Pointer(&scale)), nil
}

// KDAResult retains an invocation and its publication destinations. Its zero
// value is unusable. Wait may be retried after cancellation or a failed drain;
// it never publishes a consumer identity before successful native completion.
type KDAResult struct {
	mu                  sync.Mutex
	owner               *KDA
	stream              *cuda.Stream
	buffers             KDABuffers
	ids                 []KDAIdentity
	releases            []func()
	compute, publish    *cudaplan.Completion
	submitErr, terminal error
	copied              bool
	done                atomic.Bool
	state               *KDAState
	// Controlled host phase hooks exercise this same publication state machine.
	computeWait, publishWait func(context.Context) error
	readStatus, copyResults  func() error
}

// Wait drains compute, checks every device status, submits publication, then
// drains publication. All borrows span both phases. A partially failed copy
// cannot publish state; its queued work is drained before releasing pins.
func (r *KDAResult) Wait(ctx context.Context) error {
	if r == nil || r.owner == nil {
		return fmt.Errorf("cudaattn: invalid kda result")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.done.Load() {
		return r.terminal
	}
	if r.computeWait != nil {
		if err := r.computeWait(ctx); err != nil {
			return err
		}
	} else {
		if r.compute == nil {
			var err error
			r.compute, err = cudaplan.NewStreamCompletion(r.stream)
			if err != nil {
				return err
			}
		}
		if err := r.compute.Wait(ctx); err != nil {
			return err
		}
	}
	if !r.copied {
		if r.submitErr != nil {
			return r.finish(r.submitErr)
		}
		if r.readStatus != nil {
			if err := r.readStatus(); err != nil {
				return r.finish(err)
			}
		} else {
			status := []int32{-1}
			if err := r.owner.status.CopyToHost(status); err != nil {
				return r.finish(err)
			}
			if status[0] != 0 {
				return r.finish(fmt.Errorf("cudaattn: kda nonfinite or missing device result"))
			}
		}
		r.copied = true
		if r.copyResults != nil {
			r.terminal = r.copyResults()
		} else {
			r.terminal = r.buffers.FinalState.CopyDeviceToDeviceAsync(r.owner.scratchState, r.stream)
			if r.terminal == nil && r.buffers.Output.Len() > 0 {
				r.terminal = r.buffers.Output.CopyDeviceToDeviceAsync(r.owner.scratchOutput, r.stream)
			}
		}
	}
	if r.publishWait != nil {
		if err := r.publishWait(ctx); err != nil {
			return err
		}
	} else {
		if r.publish == nil {
			var err error
			r.publish, err = cudaplan.NewStreamCompletion(r.stream, r.releases...)
			if err != nil {
				return err
			}
			r.releases = nil
		}
		if err := r.publish.Wait(ctx); err != nil {
			return err
		}
	}
	if r.terminal == nil {
		r.state = &KDAState{data: r.buffers.FinalState, ids: append([]KDAIdentity(nil), r.ids...), heads: r.owner.config.Heads, scale: r.owner.config.Scale}
	}
	return r.finish(r.terminal)
}
func (r *KDAResult) finish(err error) error {
	r.terminal = err
	for _, f := range r.releases {
		f()
	}
	r.releases = nil
	r.done.Store(true)
	return err
}

// State returns the successfully published state, or an error while pending or
// after failure. It cannot expose a speculative consumer prefix.
func (r *KDAResult) State() (*KDAState, error) {
	if r == nil {
		return nil, fmt.Errorf("cudaattn: invalid kda result")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.done.Load() || r.state == nil {
		return nil, fmt.Errorf("cudaattn: kda state not published")
	}
	return r.state, nil
}

// Close waits without cancellation for this result's compute and publication.
func (r *KDAResult) Close() error { return r.Wait(nil) }

// Close drains outstanding work, then frees scratch and unloads the artifact.
// Cleanup errors retain the remaining resources for another Close call.
func (k *KDA) Close() error {
	if k == nil {
		return nil
	}
	k.mu.Lock()
	k.closed = true
	r := k.pending
	k.mu.Unlock()
	if r != nil && !r.done.Load() {
		if err := r.Close(); err != nil && !r.done.Load() {
			return err
		}
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	for _, free := range []func() error{k.scratchState.Free, k.scratchOutput.Free, k.offsets.Free, k.status.Free} {
		if err := free(); err != nil {
			return err
		}
	}
	if k.module != nil {
		if err := k.module.Close(); err != nil {
			return err
		}
		k.module = nil
	}
	if k.raw != 0 {
		if err := k.dev.WithContext(k.raw.Close); err != nil {
			return err
		}
		k.raw = 0
	}
	if k.cleanup != nil {
		if err := k.cleanup.Close(); err != nil {
			return err
		}
		k.cleanup = nil
	}
	k.kernel = nil
	return nil
}
