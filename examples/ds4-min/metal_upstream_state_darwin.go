//go:build darwin && arm64

package main

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4kvstore"
)

// LoadUpstream stages a version-qualified upstream payload before copying it
// into request-owned Metal shared buffers. A receipt supplies model/token
// provenance; the checkpoint itself cannot authorize those identities.
func (s *metalRequestState) LoadUpstream(record dsv4kvstore.UpstreamRecord, contract dsv4kvstore.UpstreamRestoreContract) error {
	if s == nil || s.closed || s.runtime == nil || s.runtime.arena == nil || (s.owner != nil && s.owner.closed) {
		return dsv4.ErrClosed
	}
	r := s.runtime
	if r.failed != nil {
		return errors.Join(dsv4.ErrDesynchronized, r.failed)
	}
	config, err := ReadConfig(r.model)
	if err != nil {
		return fmt.Errorf("Metal upstream model config: %w", err)
	}
	if len(config.CompressRatio) != len(contract.Ratios) {
		return errors.New("Metal upstream ratio count mismatch")
	}
	for i, ratio := range config.CompressRatio {
		if uint32(ratio) != contract.Ratios[i] {
			return fmt.Errorf("Metal upstream ratio mismatch at layer %d", i)
		}
	}
	layout, err := dsv4kvstore.ValidateUpstreamRecordLayout(record, contract.Ratios)
	if err != nil {
		return fmt.Errorf("Metal upstream layout: %w", err)
	}
	if err := dsv4kvstore.ValidateUpstreamRestoreContract(layout, contract); err != nil {
		return err
	}
	m := layout.Metadata
	if r.ctx != int(m.Context) || m.HeadDim != ds4KVWidth || m.IndexerHeadDim != 128 || len(r.state.layers) != int(m.Layers) || record.Metadata.QuantBits != 2 || contract.QuantBits != 2 || len(record.Trailer) != 0 {
		return errors.New("Metal upstream target geometry or quantization mismatch")
	}
	tokenHash, err := dsv4kvstore.HashUpstreamPayloadTokenIDs(bytes.NewReader(record.Payload), layout)
	if err != nil || tokenHash != contract.TokenIDsSHA256 {
		return errors.New("Metal upstream token provenance mismatch")
	}
	if r.file == nil {
		return errors.New("Metal upstream model file is unavailable")
	}
	info, err := r.file.Stat()
	if err != nil {
		return err
	}
	h := sha256.New()
	if _, err := io.Copy(h, io.NewSectionReader(r.file, 0, info.Size())); err != nil {
		return fmt.Errorf("Metal upstream model hash: %w", err)
	}
	if !bytes.Equal(h.Sum(nil), contract.ModelSHA256[:]) {
		return errors.New("Metal upstream model provenance mismatch")
	}
	layers := make([]upstreamFloatLayer, len(r.state.layers))
	for i := range layers {
		l := &r.state.layers[i]
		if uint32(l.attn.ratio) != contract.Ratios[i] || l.rawKV.Len() != int(contract.TargetRawCapacity)*ds4KVWidth {
			return fmt.Errorf("Metal upstream layer %d geometry mismatch", i)
		}
		layers[i] = upstreamFloatLayer{
			raw: l.rawKV.HostSlice(), cache: l.attn.cache.HostSlice(), kv: l.attn.stateKV.HostSlice(), score: l.attn.stateScore.HostSlice(), pooled: l.attn.pooled.HostSlice(),
			indexCache: l.indexer.cache.HostSlice(), indexKV: l.indexer.stateKV.HostSlice(), indexScore: l.indexer.stateScore.HostSlice(), indexPooled: l.indexer.pooled.HostSlice(),
			ratio: l.attn.ratio, indexRatio: l.indexer.ratio, capacity: l.attn.cap, indexCapacity: l.indexer.cap,
		}
	}
	writes, ns, indexNs, err := planUpstreamFloatRestore(record.Payload, layout, layers, r.state.logits.HostSlice())
	if err != nil {
		return err
	}
	if r.ops == nil || r.ops.hc == nil {
		return errors.New("Metal upstream operations unavailable")
	}
	// Public evaluation waits before returning. This additional queue barrier
	// makes the host-write boundary explicit and drains prior submitted work.
	if err := r.synchronizeState(); err != nil {
		return fmt.Errorf("Metal upstream synchronize: %w", err)
	}
	// Metal has no retained CUDA-style graph executable. HC's cached split is
	// its only address-keyed state; discard it before consuming imported rows.
	r.invalidateStateCaches()
	if err := commitUpstreamFloatRestore(writes, func(dst, values []float32) error { copy(dst, values); return nil }); err != nil {
		if errors.Is(err, dsv4.ErrDesynchronized) {
			r.failed = err
		}
		return err
	}
	for i := range r.state.layers {
		r.state.layers[i].attn.n = ns[i]
		r.state.layers[i].indexer.n = indexNs[i]
	}
	r.pos = int(m.Tokens)
	r.lastBatch = metalBatchStats{}
	return nil
}
