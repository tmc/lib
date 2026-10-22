//go:build linux

package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaStateSnapshot is a device-resident copy of the mutable request state.
// It deliberately belongs to the runtime that created it: device pointers are
// not meaningful to another CUDA context, and copying a snapshot between
// runtimes would silently turn a restore into a partial restore.
type cudaStateSnapshot struct {
	owner                      *cudaRuntime
	arena                      *cudamem.Arena
	state                      cudaSessionState
	logits                     []float32
	pos                        int
	closed                     bool
	closing                    bool
	closeArena                 func(*cudamem.Arena) error
	dspark                     *cudaDSparkFeatureSnapshot
	generation                 uint64
	steering                   steeringIdentity
	supportModel, supportQuant [32]byte
}

var _ dsv4.Snapshot = (*cudaStateSnapshot)(nil)

func cloneCUDAFloat(a *cudamem.Arena, name string, src cudamem.Slice[float32]) (cudamem.Slice[float32], error) {
	if src.Len() == 0 {
		return cudamem.Slice[float32]{}, nil
	}
	dst := cudamem.AllocIn[float32](a, name, src.Len())
	if err := a.Err(); err != nil {
		return dst, err
	}
	if err := dst.CopyDeviceToDevice(src); err != nil {
		return dst, fmt.Errorf("copy %s: %w", name, err)
	}
	return dst, nil
}

func cloneCUDAUint32(a *cudamem.Arena, name string, src cudamem.Slice[uint32]) (cudamem.Slice[uint32], error) {
	if src.Len() == 0 {
		return cudamem.Slice[uint32]{}, nil
	}
	dst := cudamem.AllocIn[uint32](a, name, src.Len())
	if err := a.Err(); err != nil {
		return dst, err
	}
	if err := dst.CopyDeviceToDevice(src); err != nil {
		return dst, fmt.Errorf("copy %s: %w", name, err)
	}
	return dst, nil
}

func cloneCUDAInt32(a *cudamem.Arena, name string, src cudamem.Slice[int32]) (cudamem.Slice[int32], error) {
	if src.Len() == 0 {
		return cudamem.Slice[int32]{}, nil
	}
	dst := cudamem.AllocIn[int32](a, name, src.Len())
	if err := a.Err(); err != nil {
		return dst, err
	}
	if err := dst.CopyDeviceToDevice(src); err != nil {
		return dst, fmt.Errorf("copy %s: %w", name, err)
	}
	return dst, nil
}

func cloneCUDABytes(a *cudamem.Arena, name string, src cudamem.Slice[byte]) (cudamem.Slice[byte], error) {
	if src.Len() == 0 {
		return cudamem.Slice[byte]{}, nil
	}
	dst := cudamem.AllocIn[byte](a, name, src.Len())
	if err := a.Err(); err != nil {
		return dst, err
	}
	if err := dst.CopyDeviceToDevice(src); err != nil {
		return dst, fmt.Errorf("copy %s: %w", name, err)
	}
	return dst, nil
}

func cloneCUDACompressed(a *cudamem.Arena, name string, src cudaCompressedState) (cudaCompressedState, error) {
	dst := src
	var err error
	if dst.cache, err = cloneCUDAFloat(a, name+"-cache", src.cache); err != nil {
		return dst, err
	}
	if dst.stateKV, err = cloneCUDAFloat(a, name+"-kv", src.stateKV); err != nil {
		return dst, err
	}
	if dst.stateScore, err = cloneCUDAFloat(a, name+"-scores", src.stateScore); err != nil {
		return dst, err
	}
	if dst.pooled, err = cloneCUDAFloat(a, name+"-pooled", src.pooled); err != nil {
		return dst, err
	}
	return dst, nil
}

func cloneCUDAState(a *cudamem.Arena, src cudaSessionState) (cudaSessionState, error) {
	dst := src
	var err error
	t := &src.cudaTokenState
	u := &dst.cudaTokenState
	clone := func(name string, p *cudamem.Slice[float32]) error {
		var e error
		*p, e = cloneCUDAFloat(a, name, *p)
		return e
	}
	for i := range t.hc {
		if err = clone(fmt.Sprintf("hc-%d", i), &u.hc[i]); err != nil {
			return dst, err
		}
	}
	for name, p := range map[string]*cudamem.Slice[float32]{
		"attention-pre": &u.attnPre, "query-lora": &u.qLora, "query-heads": &u.qHeads,
		"kv-raw": &u.kvRaw, "kv-current": &u.kvCur, "compressor-kv": &u.compressorKV,
		"compressor-gate": &u.compressorGate, "indexer-kv": &u.indexerKV,
		"indexer-gate": &u.indexerGate, "indexer-query": &u.indexerQ,
		"indexer-weights": &u.indexerWeights, "indexer-query-scratch": &u.indexerScratch,
		"indexer-scores": &u.indexerScores, "raw-scores": &u.rawScores,
		"raw-output": &u.rawOutput, "mixed-scores": &u.mixedScores,
		"mixed-denominators": &u.mixedDenominators, "mixed-output": &u.mixedOutput,
		"attention-low": &u.attnLow, "attention-output": &u.attnOut, "ffn-pre": &u.ffnPre,
		"router": &u.router, "router-probabilities": &u.probs, "expert-weights": &u.expertW,
		"shared-gate": &u.sharedGate, "shared-up": &u.sharedUp, "shared-mid": &u.sharedMid,
		"expert-gate": &u.expertGate, "expert-up": &u.expertUp, "expert-mid": &u.expertMid,
		"expert-down": &u.expertDown, "ffn-blocks": &u.ffnBlocks, "ffn-output": &u.ffnOut,
		"output-mixed": &u.outputMixed, "logits": &u.logits,
	} {
		if err = clone(name, p); err != nil {
			return dst, err
		}
	}
	for name, p := range map[string]*cudamem.Slice[byte]{
		"q80-input": &u.q80Input, "q81-input": &u.q81Input, "q81-experts": &u.q81Experts,
		"q8k-input": &u.q8KInput, "q8k-experts": &u.q8KExperts,
	} {
		*p, err = cloneCUDABytes(a, name, *p)
		if err != nil {
			return dst, err
		}
	}
	if u.allowed, err = cloneCUDAUint32(a, "allowed-rows", t.allowed); err != nil {
		return dst, err
	}
	if u.expertIDs, err = cloneCUDAInt32(a, "expert-ids", t.expertIDs); err != nil {
		return dst, err
	}
	for i := range src.layers {
		dst.layers[i].rawKV, err = cloneCUDAFloat(a, fmt.Sprintf("layer-%d-raw", i), src.layers[i].rawKV)
		if err != nil {
			return dst, err
		}
		dst.layers[i].attn, err = cloneCUDACompressed(a, fmt.Sprintf("layer-%d-attention", i), src.layers[i].attn)
		if err != nil {
			return dst, err
		}
		dst.layers[i].indexer, err = cloneCUDACompressed(a, fmt.Sprintf("layer-%d-indexer", i), src.layers[i].indexer)
		if err != nil {
			return dst, err
		}
	}
	return dst, nil
}

func restoreCUDAFloat(dst, src cudamem.Slice[float32], name string) error {
	if dst.Len() != src.Len() {
		return fmt.Errorf("restore %s: length %d, want %d", name, src.Len(), dst.Len())
	}
	if err := dst.CopyDeviceToDevice(src); err != nil {
		return fmt.Errorf("restore %s: %w", name, err)
	}
	return nil
}

func restoreCUDAState(dst *cudaSessionState, src cudaSessionState) error {
	if dst == nil {
		return dsv4.ErrClosed
	}
	d, s := &dst.cudaTokenState, &src.cudaTokenState
	copyFloat := func(name string, a, b cudamem.Slice[float32]) error { return restoreCUDAFloat(a, b, name) }
	for i := range d.hc {
		if err := copyFloat(fmt.Sprintf("hc-%d", i), d.hc[i], s.hc[i]); err != nil {
			return err
		}
	}
	for _, item := range []struct {
		name     string
		dst, src cudamem.Slice[float32]
	}{
		{"attention-pre", d.attnPre, s.attnPre}, {"query-lora", d.qLora, s.qLora}, {"query-heads", d.qHeads, s.qHeads},
		{"kv-raw", d.kvRaw, s.kvRaw}, {"kv-current", d.kvCur, s.kvCur}, {"compressor-kv", d.compressorKV, s.compressorKV},
		{"compressor-gate", d.compressorGate, s.compressorGate}, {"indexer-kv", d.indexerKV, s.indexerKV}, {"indexer-gate", d.indexerGate, s.indexerGate},
		{"indexer-query", d.indexerQ, s.indexerQ}, {"indexer-weights", d.indexerWeights, s.indexerWeights}, {"indexer-query-scratch", d.indexerScratch, s.indexerScratch},
		{"indexer-scores", d.indexerScores, s.indexerScores}, {"raw-scores", d.rawScores, s.rawScores}, {"raw-output", d.rawOutput, s.rawOutput},
		{"mixed-scores", d.mixedScores, s.mixedScores}, {"mixed-denominators", d.mixedDenominators, s.mixedDenominators}, {"mixed-output", d.mixedOutput, s.mixedOutput},
		{"attention-low", d.attnLow, s.attnLow}, {"attention-output", d.attnOut, s.attnOut}, {"ffn-pre", d.ffnPre, s.ffnPre},
		{"router", d.router, s.router}, {"router-probabilities", d.probs, s.probs}, {"expert-weights", d.expertW, s.expertW},
		{"shared-gate", d.sharedGate, s.sharedGate}, {"shared-up", d.sharedUp, s.sharedUp}, {"shared-mid", d.sharedMid, s.sharedMid},
		{"expert-gate", d.expertGate, s.expertGate}, {"expert-up", d.expertUp, s.expertUp}, {"expert-mid", d.expertMid, s.expertMid},
		{"expert-down", d.expertDown, s.expertDown}, {"ffn-blocks", d.ffnBlocks, s.ffnBlocks}, {"ffn-output", d.ffnOut, s.ffnOut},
		{"output-mixed", d.outputMixed, s.outputMixed}, {"logits", d.logits, s.logits},
	} {
		if err := copyFloat(item.name, item.dst, item.src); err != nil {
			return err
		}
	}
	for _, item := range []struct {
		name     string
		dst, src cudamem.Slice[byte]
	}{{"q80-input", d.q80Input, s.q80Input}, {"q81-input", d.q81Input, s.q81Input}, {"q81-experts", d.q81Experts, s.q81Experts}, {"q8k-input", d.q8KInput, s.q8KInput}, {"q8k-experts", d.q8KExperts, s.q8KExperts}} {
		if item.dst.Len() != item.src.Len() {
			return fmt.Errorf("restore %s: length mismatch", item.name)
		}
		if item.src.Len() == 0 {
			continue
		}
		if err := item.dst.CopyDeviceToDevice(item.src); err != nil {
			return fmt.Errorf("restore %s: %w", item.name, err)
		}
	}
	if d.allowed.Len() != s.allowed.Len() || d.expertIDs.Len() != s.expertIDs.Len() {
		return fmt.Errorf("restore integer state: length mismatch")
	}
	if err := d.allowed.CopyDeviceToDevice(s.allowed); err != nil {
		return fmt.Errorf("restore allowed rows: %w", err)
	}
	if err := d.expertIDs.CopyDeviceToDevice(s.expertIDs); err != nil {
		return fmt.Errorf("restore expert IDs: %w", err)
	}
	for i := range dst.layers {
		if err := copyFloat(fmt.Sprintf("layer-%d-raw", i), dst.layers[i].rawKV, src.layers[i].rawKV); err != nil {
			return err
		}
		dst.layers[i].attn.n = src.layers[i].attn.n
		dst.layers[i].indexer.n = src.layers[i].indexer.n
		for _, item := range []struct {
			name     string
			dst, src cudaCompressedState
		}{{"attention", dst.layers[i].attn, src.layers[i].attn}, {"indexer", dst.layers[i].indexer, src.layers[i].indexer}} {
			for _, item2 := range []struct {
				suffix   string
				dst, src cudamem.Slice[float32]
			}{{"cache", item.dst.cache, item.src.cache}, {"kv", item.dst.stateKV, item.src.stateKV}, {"scores", item.dst.stateScore, item.src.stateScore}, {"pooled", item.dst.pooled, item.src.pooled}} {
				if err := copyFloat(fmt.Sprintf("layer-%d-%s-%s", i, item.name, item2.suffix), item2.dst, item2.src); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (s *cudaStateSnapshot) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closing = true
	if s.arena == nil {
		s.closed = true
		return nil
	}
	release := s.closeArena
	if release == nil {
		release = func(a *cudamem.Arena) error { return a.Free() }
	}
	if err := release(s.arena); err != nil {
		return err
	}
	s.arena = nil
	s.closed = true
	return nil
}

func (s *cudaRequestState) Snapshot() (dsv4.Snapshot, error) {
	return s.snapshotWithOperations(defaultCUDASnapshotOperations())
}

func (s *cudaRequestState) snapshotWithOperations(ops cudaSnapshotOperations) (_ dsv4.Snapshot, err error) {
	if s == nil || s.closed || s.runtime == nil {
		return nil, dsv4.ErrClosed
	}
	if s.runtime.failed != nil {
		return nil, s.runtime.failed
	}
	if err := ops.synchronize(s.runtime); err != nil {
		return nil, fmt.Errorf("cuda state snapshot synchronize: %w", err)
	}
	v := &cudaStateSnapshot{owner: s.runtime, closeArena: ops.free}
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, v)
		}
	}()
	if s.dspark != nil {
		v.supportModel, v.supportQuant, err = s.runtime.cudaFoundation.nativeStateIdentity()
		if err != nil {
			return nil, fmt.Errorf("cuda support snapshot identity: %w", err)
		}
	}
	v.arena, err = ops.arena(s.runtime)
	if err != nil {
		return nil, fmt.Errorf("cuda state snapshot arena: %w", err)
	}
	v.state, err = ops.clone(v.arena, s.runtime.state)
	if err = errors.Join(err, ops.arenaError(v.arena)); err != nil {
		return nil, fmt.Errorf("cuda state snapshot: %w", err)
	}
	feature, err := s.dspark.snapshot(v.arena)
	if err != nil {
		return nil, err
	}
	v.logits = append([]float32(nil), s.runtime.logits...)
	v.pos = s.runtime.pos
	v.dspark = feature
	v.generation = s.generation
	v.steering = s.runtime.steeringStateIdentity()
	return v, nil
}

func (s *cudaRequestState) Restore(snapshot dsv4.Snapshot) (err error) {
	if s == nil || s.closed || s.runtime == nil {
		return dsv4.ErrClosed
	}
	v, ok := snapshot.(*cudaStateSnapshot)
	if !ok || v == nil || v.closed || v.closing || v.owner != s.runtime {
		return fmt.Errorf("cuda state restore: snapshot belongs to another state: %w", dsv4.ErrUnimplemented)
	}
	if v.steering != s.runtime.steeringStateIdentity() {
		return fmt.Errorf("cuda state restore: steering identity mismatch")
	}
	if errors.Is(s.runtime.failed, dsv4.ErrDesynchronized) {
		return s.runtime.failed
	}
	if err := validateCUDAStateFrontier(v.pos, s.runtime.ctx); err != nil {
		return err
	}
	if err := validateCUDARestore(s.runtime.state, v.state, len(s.runtime.logits), len(v.logits)); err != nil {
		return err
	}
	if err := s.dspark.validateSnapshot(v.dspark, v.pos, v.generation); err != nil {
		return err
	}
	backup, err := s.Snapshot()
	if err != nil {
		return err
	}
	defer func() { err = finishBackendCleanup(err, backup) }()
	if err := cudaStateTransaction(func() error { return s.restoreSnapshot(v) }, func() error { return s.restoreSnapshot(backup.(*cudaStateSnapshot)) }, func(err error) {
		s.runtime.failed = err
		s.runtime.headObserver.invalidate()
	}); err != nil {
		return err
	}
	return s.completeObservedCacheReplacement(nil)
}

func validateCUDAStateFrontier(position, context int) error {
	if position < 0 || position > context {
		return fmt.Errorf("cuda state restore: position %d exceeds context %d", position, context)
	}
	return nil
}

func validateCUDARestore(dst, src cudaSessionState, logits, savedLogits int) error {
	if logits != savedLogits {
		return fmt.Errorf("cuda state restore: logits length %d, want %d", savedLogits, logits)
	}
	d, s := cudaStateFields(&dst, nil), cudaStateFields(&src, nil)
	if len(d) != len(s) {
		return fmt.Errorf("cuda state restore: field count mismatch")
	}
	for i := range d {
		if d[i].name != s[i].name || d[i].kind != s[i].kind || cudaStateFieldLength(d[i]) != cudaStateFieldLength(s[i]) {
			return fmt.Errorf("cuda state restore: field %s shape mismatch", d[i].name)
		}
	}
	attn, indexer := make([]int, len(src.layers)), make([]int, len(src.layers))
	for i := range src.layers {
		attn[i], indexer[i] = src.layers[i].attn.n, src.layers[i].indexer.n
	}
	return validateCUDAStateLayerCounts(dst, attn, indexer)
}

func (s *cudaRequestState) restoreSnapshot(v *cudaStateSnapshot) error {
	if err := syncStream(s.runtime.stream()); err != nil {
		return fmt.Errorf("cuda state restore synchronize: %w", err)
	}
	if err := restoreCUDAState(&s.runtime.state, v.state); err != nil {
		return fmt.Errorf("cuda state restore: %w", err)
	}
	if len(s.runtime.logits) != len(v.logits) {
		return fmt.Errorf("cuda state restore: logits length %d, want %d", len(v.logits), len(s.runtime.logits))
	}
	copy(s.runtime.logits, v.logits)
	s.runtime.pos = v.pos
	s.runtime.failed = nil
	if err := s.dspark.restore(v.dspark, v.pos, v.generation); err != nil {
		return err
	}
	s.generation = v.generation
	return nil
}
