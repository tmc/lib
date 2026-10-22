//go:build darwin && arm64

package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

const (
	metalStateMagic   = "DS4MST02"
	metalStateVersion = uint32(3)
	// Each layer records both attention and indexer counts.
	metalStateHeaderSize = 8 + 4 + 4 + 8 + 4 + 8*ds4Layers + 64
	metalStateMaxBytes   = int64(2 << 30)
)

// metalStateAdapter is the checked boundary between the legacy Metal runtime
// and the request-state contract. openSessionBackend uses it to create
// request-local state around immutable model bindings; mutable KV, compressor,
// indexer, and hyper-connection history belongs to each returned state.
type metalStateAdapter struct {
	runtime *metalRuntime
	closed  bool
}

var _ dsv4.Backend = (*metalStateAdapter)(nil)
var _ dsv4.IndependentStates = (*metalStateAdapter)(nil)
var _ dsv4.SpeculativeBackend = (*metalStateAdapter)(nil)

// NewSpeculativeVerifier returns the scalar, rollback-capable verifier. It is
// deliberately scalar: Metal snapshots are host copies of the request-owned
// buffers, so a future batched implementation must preserve this state and
// ordering behavior before it replaces this path.
func (a *metalStateAdapter) NewSpeculativeVerifier(state dsv4.State) (dsv4.SpeculativeVerifier, error) {
	if a == nil || a.closed {
		return nil, dsv4.ErrClosed
	}
	s, ok := state.(*metalRequestState)
	if !ok || s == nil || s.closed || s.owner != a || s.runtime == nil {
		return nil, fmt.Errorf("Metal speculative verifier: state is not owned by backend: %w", dsv4.ErrUnimplemented)
	}
	return &metalSpeculativeVerifier{adapter: a, state: s}, nil
}

func (a *metalStateAdapter) NewState(context int) (dsv4.State, error) {
	if a == nil || a.closed {
		return nil, dsv4.ErrClosed
	}
	if a.runtime == nil || a.runtime.dev == nil || a.runtime.views == nil ||
		a.runtime.ops == nil || a.runtime.arena == nil || a.runtime.model == nil {
		return nil, dsv4.ErrUnimplemented
	}
	if err := validateStateContext(context, a.runtime.ctx); err != nil {
		return nil, err
	}
	r, err := a.runtime.newIndependentState(context)
	if err != nil {
		return nil, err
	}
	return &metalRequestState{runtime: r, owner: a}, nil
}

func (a *metalStateAdapter) Eval(state dsv4.State, token, position int) error {
	if a == nil || a.closed {
		return dsv4.ErrClosed
	}
	s, ok := state.(*metalRequestState)
	if !ok || s == nil || s.owner != a {
		return dsv4.ErrUnimplemented
	}
	return s.eval(token, position)
}

func (a *metalStateAdapter) EvalBatch(state dsv4.State, tokens []int, position int) error {
	if a == nil || a.closed || a.runtime == nil {
		return dsv4.ErrClosed
	}
	s, ok := state.(*metalRequestState)
	if !ok || s == nil || s.owner != a || s.closed || s.runtime == nil {
		return dsv4.ErrUnimplemented
	}
	return s.runtime.EvalBatch(tokens, position)
}

func (a *metalStateAdapter) Logits(state dsv4.State) []float32 {
	if a == nil || a.closed {
		return nil
	}
	s, ok := state.(*metalRequestState)
	if !ok || s == nil || s.owner != a {
		return nil
	}
	return s.logits()
}

func (a *metalStateAdapter) Close() error {
	if a == nil || a.closed {
		return nil
	}
	a.closed = true
	if a.runtime == nil {
		return nil
	}
	return a.runtime.Close()
}

// stateBackend returns the backend adapter used by the session seam.
func (r *metalRuntime) stateBackend() dsv4.Backend {
	return &metalStateAdapter{runtime: r}
}

// newIndependentState shares only immutable model bindings with r. Operations
// (including their HC scratch and command-buffer bookkeeping), the arena, and
// all mutable inference buffers belong to the returned runtime. The model
// mapping and device remain owned by r and are therefore not closed with the
// request.
func (r *metalRuntime) newIndependentState(context int) (*metalRuntime, error) {
	if r == nil || r.dev == nil || r.views == nil || r.ops == nil || r.arena == nil || r.model == nil {
		return nil, dsv4.ErrUnimplemented
	}
	if err := validateStateContext(context, r.ctx); err != nil {
		return nil, err
	}
	ops, err := newMetalPinnedOps(r.dev)
	if err != nil {
		return nil, err
	}
	state := &metalRuntime{
		power:         r.power.newRequest(),
		file:          r.file,
		dev:           r.dev,
		views:         r.views,
		ops:           ops,
		model:         r.model,
		stateIdentity: r.stateIdentity,
		layers:        r.layers,
		output:        r.output,
		ctx:           context,
		borrowed:      true,
		streaming:     r.streaming.request(),
	}
	state.arena = metalmem.NewArena(r.dev)
	state.allocState()
	if err := state.arena.Err(); err != nil {
		_ = state.Close()
		return nil, fmt.Errorf("Metal state: allocate request state: %w", err)
	}
	if err := state.Reset(); err != nil {
		_ = state.Close()
		return nil, fmt.Errorf("Metal state: reset request state: %w", err)
	}
	return state, nil
}

type metalRequestState struct {
	runtime *metalRuntime
	owner   *metalStateAdapter
	closed  bool
}

// metalSnapshot is an immutable host copy of the request-owned buffers. Metal
// uses shared allocations, and every public evaluation waits for its command
// buffer before returning, so the serialized state is a stable snapshot at
// this boundary. The owner check keeps a snapshot from being restored into a
// request backed by a different model or device.
type metalSnapshot struct {
	owner    *metalStateAdapter
	request  *metalRequestState
	data     []byte
	closed   bool
	steering steeringIdentity
	dspark   *metalDSparkFeatureSnapshot
}

func (s *metalSnapshot) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	s.data = nil
	s.dspark = nil
	return nil
}

func (s *metalRequestState) eval(token, position int) error {
	if s == nil || s.closed || s.runtime == nil || (s.owner != nil && s.owner.closed) {
		return dsv4.ErrClosed
	}
	return s.runtime.Eval(token, position)
}

func (s *metalRequestState) logits() []float32 {
	if s == nil || s.closed || s.runtime == nil || (s.owner != nil && s.owner.closed) {
		return nil
	}
	return s.runtime.Logits()
}

func (s *metalRequestState) Reset() error {
	if s == nil || s.closed || s.runtime == nil || (s.owner != nil && s.owner.closed) {
		return dsv4.ErrClosed
	}
	return s.runtime.Reset()
}

func (s *metalRequestState) Snapshot() (dsv4.Snapshot, error) {
	if s == nil || s.runtime == nil {
		return nil, fmt.Errorf("Metal state snapshot: %w", dsv4.ErrUnimplemented)
	}
	if s.closed || (s.owner != nil && s.owner.closed) {
		return nil, dsv4.ErrClosed
	}
	if s.owner == nil {
		return nil, fmt.Errorf("Metal state snapshot: missing owner")
	}
	var data bytes.Buffer
	if err := saveMetalState(s.runtime, &data); err != nil {
		return nil, fmt.Errorf("Metal state snapshot: %w", err)
	}
	v := &metalSnapshot{owner: s.owner, request: s, data: append([]byte(nil), data.Bytes()...), steering: s.runtime.steeringStateIdentity()}
	if s.runtime.dsparkFeatures != nil {
		var err error
		v.dspark, err = s.runtime.dsparkFeatures.snapshot()
		if err != nil {
			return nil, err
		}
	}
	return v, nil
}

func (s *metalRequestState) Restore(snapshot dsv4.Snapshot) error {
	if s == nil || s.runtime == nil {
		return fmt.Errorf("Metal state restore: %w", dsv4.ErrUnimplemented)
	}
	if s.closed || (s.owner != nil && s.owner.closed) {
		return dsv4.ErrClosed
	}
	if err := metalStatePoison(s.runtime); err != nil {
		return err
	}
	if s.runtime.speculativeEpoch == ^uint64(0) {
		return fmt.Errorf("metal state restore: speculative epoch overflow")
	}
	v, ok := snapshot.(*metalSnapshot)
	if !ok || v == nil || v.closed || v.owner != s.owner || v.request != s {
		return fmt.Errorf("Metal state restore: %w", dsv4.ErrUnimplemented)
	}
	if v.steering != s.runtime.steeringStateIdentity() {
		return fmt.Errorf("Metal state restore: steering identity mismatch")
	}
	if err := v.steering.validate(); err != nil {
		return fmt.Errorf("Metal state restore steering: %w", err)
	}
	if (s.runtime.dsparkFeatures == nil) != (v.dspark == nil) {
		return fmt.Errorf("metal state restore: support owner mismatch")
	}
	if s.runtime.dsparkFeatures != nil {
		if len(v.data) < 24 {
			return fmt.Errorf("metal state restore: missing target frontier")
		}
		position := binary.LittleEndian.Uint64(v.data[16:24])
		if position > uint64(s.runtime.ctx) {
			return fmt.Errorf("metal state restore: target frontier exceeds context")
		}
		if err := s.runtime.dsparkFeatures.validateSnapshot(v.dspark, int(position)); err != nil {
			return err
		}
	}
	if err := loadMetalState(s.runtime, bytes.NewReader(v.data)); err != nil {
		return err
	}
	if s.runtime.dsparkFeatures != nil {
		s.runtime.dsparkFeatures.publishSnapshot(v.dspark)
	}
	s.runtime.speculativeEpoch++
	return nil
}

func (s *metalRequestState) Save(w io.Writer) error {
	if s == nil || s.runtime == nil {
		return fmt.Errorf("Metal state save: %w", dsv4.ErrUnimplemented)
	}
	if s.closed || (s.owner != nil && s.owner.closed) {
		return dsv4.ErrClosed
	}
	if s.runtime.dsparkFeatures != nil {
		return s.saveSupport(w)
	}
	return saveMetalState(s.runtime, w)
}

func (s *metalRequestState) Load(r io.Reader) error {
	if s == nil || s.runtime == nil {
		return fmt.Errorf("Metal state load: %w", dsv4.ErrUnimplemented)
	}
	if s.closed || (s.owner != nil && s.owner.closed) {
		return dsv4.ErrClosed
	}
	if s.runtime.dsparkFeatures != nil {
		return s.loadSupport(r)
	}
	if s.runtime.speculativeEpoch == ^uint64(0) {
		return fmt.Errorf("metal state load: speculative epoch overflow")
	}
	if err := loadMetalState(s.runtime, r); err != nil {
		return err
	}
	s.runtime.speculativeEpoch++
	return nil
}

// saveMetalState writes the mutable request buffers in a private, versioned
// format. The model bindings are immutable and deliberately do not appear in
// this payload. All device work is synchronous before this adapter returns
// from Eval, so HostSlice is a stable view of the state here.
func saveMetalState(r *metalRuntime, w io.Writer) error {
	if r == nil || r.arena == nil {
		return dsv4.ErrClosed
	}
	if w == nil {
		return fmt.Errorf("Metal state save: nil writer")
	}
	if r.failed != nil {
		return fmt.Errorf("Metal state save: reset required: %w", r.failed)
	}
	model, quant, err := r.nativeStateIdentity()
	if err != nil {
		return fmt.Errorf("Metal state save: identity: %w", err)
	}
	if err := r.synchronizeState(); err != nil {
		return fmt.Errorf("Metal state save: synchronize: %w", err)
	}
	fields := metalStateFields(&r.state)
	if len(fields) == 0 {
		return fmt.Errorf("Metal state save: empty state")
	}
	var header bytes.Buffer
	header.WriteString(metalStateMagic)
	_ = binary.Write(&header, binary.LittleEndian, metalStateVersion)
	_ = binary.Write(&header, binary.LittleEndian, uint32(r.ctx))
	_ = binary.Write(&header, binary.LittleEndian, uint64(r.pos))
	_ = binary.Write(&header, binary.LittleEndian, uint32(len(fields)))
	for i := range r.state.layers {
		_ = binary.Write(&header, binary.LittleEndian, uint32(r.state.layers[i].attn.n))
		_ = binary.Write(&header, binary.LittleEndian, uint32(r.state.layers[i].indexer.n))
	}
	header.Write(model[:])
	header.Write(quant[:])
	steering, err := r.steeringStateIdentity().encode()
	if err != nil {
		return fmt.Errorf("Metal state save steering: %w", err)
	}
	header.Write(steering[:])
	if _, err := io.Copy(w, &header); err != nil {
		return fmt.Errorf("Metal state save: write header: %w", err)
	}
	for i, field := range fields {
		data, err := encodeMetalStateField(field)
		if err != nil {
			return fmt.Errorf("Metal state save: field %d: %w", i, err)
		}
		if uint64(len(data)) > uint64(metalStateMaxBytes) {
			return fmt.Errorf("Metal state save: field %d is too large", i)
		}
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(data)))
		if err := writeMetalStateFull(w, size[:]); err != nil {
			return fmt.Errorf("Metal state save: field %d size: %w", i, err)
		}
		if err := writeMetalStateFull(w, data); err != nil {
			return fmt.Errorf("Metal state save: field %d: %w", i, err)
		}
	}
	return nil
}

func loadMetalState(r *metalRuntime, src io.Reader) error {
	return loadMetalStateWithHooks(r, src, nil)
}

func loadMetalStateWithHooks(r *metalRuntime, src io.Reader, hooks *metalStateLoadHooks) error {
	if r == nil || r.arena == nil {
		return dsv4.ErrClosed
	}
	if err := metalStatePoison(r); err != nil {
		return err
	}
	if src == nil {
		return fmt.Errorf("Metal state load: nil reader")
	}
	data, err := io.ReadAll(io.LimitReader(src, metalStateMaxBytes+1))
	if err != nil {
		return fmt.Errorf("Metal state load: read: %w", err)
	}
	if int64(len(data)) > metalStateMaxBytes {
		return fmt.Errorf("Metal state load: payload exceeds limit")
	}
	if len(data) < metalStateHeaderSize {
		return fmt.Errorf("Metal state load: short header")
	}
	br := bytes.NewReader(data)
	magic := make([]byte, len(metalStateMagic))
	if _, err := io.ReadFull(br, magic); err != nil || string(magic) != metalStateMagic {
		return fmt.Errorf("Metal state load: invalid magic")
	}
	var version, contextSize, fieldCount uint32
	var pos uint64
	if err := binary.Read(br, binary.LittleEndian, &version); err != nil {
		return fmt.Errorf("Metal state load: version: %w", err)
	}
	if version != metalStateVersion && version != 2 {
		return fmt.Errorf("Metal state load: unsupported version %d", version)
	}
	if err := binary.Read(br, binary.LittleEndian, &contextSize); err != nil {
		return fmt.Errorf("Metal state load: context: %w", err)
	}
	if contextSize != uint32(r.ctx) {
		return fmt.Errorf("Metal state load: context %d does not match %d", contextSize, r.ctx)
	}
	if err := binary.Read(br, binary.LittleEndian, &pos); err != nil {
		return fmt.Errorf("Metal state load: position: %w", err)
	}
	if pos > uint64(r.ctx) {
		return fmt.Errorf("Metal state load: position %d exceeds context %d", pos, r.ctx)
	}
	fields := metalStateFields(&r.state)
	if err := binary.Read(br, binary.LittleEndian, &fieldCount); err != nil {
		return fmt.Errorf("Metal state load: field count: %w", err)
	}
	if fieldCount != uint32(len(fields)) {
		return fmt.Errorf("Metal state load: field count %d does not match %d", fieldCount, len(fields))
	}
	var ns [ds4Layers * 2]uint32
	for i := range ns {
		if err := binary.Read(br, binary.LittleEndian, &ns[i]); err != nil {
			return fmt.Errorf("Metal state load: layer metadata: %w", err)
		}
	}
	for i := range r.state.layers {
		if int(ns[2*i]) > r.state.layers[i].attn.cap || int(ns[2*i+1]) > r.state.layers[i].indexer.cap {
			return fmt.Errorf("Metal state load: layer %d count exceeds capacity", i)
		}
	}
	model, quant, err := r.nativeStateIdentity()
	if err != nil {
		return fmt.Errorf("Metal state load: identity: %w", err)
	}
	var storedModel, storedQuant [32]byte
	if _, err := io.ReadFull(br, storedModel[:]); err != nil {
		return fmt.Errorf("Metal state load: model identity: %w", err)
	}
	if _, err := io.ReadFull(br, storedQuant[:]); err != nil {
		return fmt.Errorf("Metal state load: quantization identity: %w", err)
	}
	if model == ([32]byte{}) || quant == ([32]byte{}) || storedModel != model || storedQuant != quant {
		return fmt.Errorf("Metal state load: model or quantization identity mismatch")
	}
	if err := checkMetalSteeringVersion(br, version, r.steeringStateIdentity()); err != nil {
		return err
	}
	// Keep encoded fields separate until every size and type has been checked.
	// This makes malformed input leave the destination untouched.
	encoded := make([][]byte, len(fields))
	for i, field := range fields {
		var size uint64
		if err := binary.Read(br, binary.LittleEndian, &size); err != nil {
			return fmt.Errorf("Metal state load: field %d size: %w", i, err)
		}
		if size > uint64(br.Len()) || size > uint64(metalStateMaxBytes) {
			return fmt.Errorf("Metal state load: field %d size %d is invalid", i, size)
		}
		want := binary.Size(field)
		if want < 0 || size != uint64(want) {
			return fmt.Errorf("Metal state load: field %d size %d does not match %d", i, size, want)
		}
		encoded[i] = make([]byte, size)
		if _, err := io.ReadFull(br, encoded[i]); err != nil {
			return fmt.Errorf("Metal state load: field %d: %w", i, err)
		}
		if err := validateMetalStateField(field, encoded[i]); err != nil {
			return fmt.Errorf("Metal state load: field %d: %w", i, err)
		}
	}
	if br.Len() != 0 {
		return fmt.Errorf("Metal state load: trailing data")
	}
	if err := r.synchronizeState(); err != nil {
		return fmt.Errorf("Metal state load: synchronize: %w", err)
	}
	if err := replaceMetalStateFields(fields, encoded, hooks, func(err error) { r.failed = err }); err != nil {
		return err
	}
	for i := range r.state.layers {
		r.state.layers[i].attn.n = int(ns[2*i])
		r.state.layers[i].indexer.n = int(ns[2*i+1])
	}
	r.pos = int(pos)
	r.failed = nil
	r.invalidateStateCaches()
	return nil
}

func writeMetalStateFull(w io.Writer, data []byte) error {
	for len(data) != 0 {
		n, err := w.Write(data)
		if err != nil {
			return err
		}
		if n <= 0 || n > len(data) {
			return io.ErrShortWrite
		}
		data = data[n:]
	}
	return nil
}

// metalStateFields lists owning allocations only; attnCur/attnNorm,
// ffnCur/ffnNorm, routedOut, and sharedOut are sub-slices of those owners.
func metalStateFields(s *metalSessionState) []any {
	fields := []any{
		s.hc[0].HostSlice(), s.hc[1].HostSlice(), s.attnPre.HostSlice(),
		s.qLora.HostSlice(), s.qHeads.HostSlice(), s.kvRaw.HostSlice(), s.kvCur.HostSlice(),
		s.compressorKV.HostSlice(), s.compressorGate.HostSlice(), s.indexerKV.HostSlice(), s.indexerGate.HostSlice(),
		s.heads.HostSlice(), s.rawScores.HostSlice(), s.rawOutput.HostSlice(), s.mixedScores.HostSlice(),
		s.mixedOutput.HostSlice(), s.allowed.HostSlice(), s.attnLow.HostSlice(), s.attnOut.HostSlice(),
		s.ffnPre.HostSlice(), s.router.HostSlice(), s.probs.HostSlice(), s.expertIDs.HostSlice(), s.expertW.HostSlice(),
		s.sharedGate.HostSlice(), s.sharedUp.HostSlice(), s.sharedMid.HostSlice(), s.ffnBlocks.HostSlice(),
		s.expertGateRaw.HostSlice(), s.expertUpRaw.HostSlice(), s.expertGate.HostSlice(), s.expertUp.HostSlice(),
		s.expertMid.HostSlice(), s.expertDown.HostSlice(), s.ffnOut.HostSlice(), s.q8.HostSlice(),
		s.outputNorm.HostSlice(), s.logits.HostSlice(),
	}
	for i := range s.layers {
		layer := &s.layers[i]
		fields = append(fields, layer.rawKV.HostSlice())
		if layer.attn.cap != 0 {
			fields = append(fields, layer.attn.cache.HostSlice(), layer.attn.stateKV.HostSlice(), layer.attn.stateScore.HostSlice(), layer.attn.pooled.HostSlice())
		}
		if layer.indexer.cap != 0 {
			fields = append(fields, layer.indexer.cache.HostSlice(), layer.indexer.stateKV.HostSlice(), layer.indexer.stateScore.HostSlice(), layer.indexer.pooled.HostSlice())
		}
	}
	return fields
}

func encodeMetalStateField(field any) ([]byte, error) {
	var b bytes.Buffer
	if err := binary.Write(&b, binary.LittleEndian, field); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

func validateMetalStateField(field any, data []byte) error {
	want := 0
	switch v := field.(type) {
	case []byte:
		want = len(v)
	case []float32:
		want = 4 * len(v)
	case []uint32:
		want = 4 * len(v)
	case []int32:
		want = 4 * len(v)
	default:
		return fmt.Errorf("unsupported field type %T", field)
	}
	if len(data) != want {
		return fmt.Errorf("payload has %d bytes, want %d", len(data), want)
	}
	return nil
}

func decodeMetalStateField(field any, data []byte) error {
	if b, ok := field.([]byte); ok {
		copy(b, data)
		return nil
	}
	return binary.Read(bytes.NewReader(data), binary.LittleEndian, field)
}

func (s *metalRequestState) Close() error {
	if s == nil || s.closed {
		return nil
	}
	if s.runtime == nil {
		s.closed = true
		return nil
	}
	err := s.runtime.Close()
	return s.finishClose(err)
}

func (s *metalRequestState) finishClose(err error) error {
	if err != nil && s.runtime.steering != nil {
		s.runtime.failed = errors.Join(err, dsv4.ErrClosed, dsv4.ErrDesynchronized)
		return err
	}
	s.closed = true
	s.runtime = nil
	return err
}
