//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4kvstore"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// LoadUpstream restores the version-2 upstream CUDA payload into this State.
// The caller must provide the model and token hashes from an independently
// qualified receipt: the upstream envelope does not identify its model.
// Validation and staging precede all writes to the live State. A failed live
// copy restores the backup; a failed rollback poisons the State.
//
// This codec copies the source's float representation, including compressed
// rows decoded from half storage. It does not certify numerical compatibility
// with a different implementation or imply a continuation parity gate passed.
func (s *cudaRequestState) LoadUpstream(record dsv4kvstore.UpstreamRecord, contract dsv4kvstore.UpstreamRestoreContract) error {
	if s == nil || s.closed || s.runtime == nil {
		return dsv4.ErrClosed
	}
	r := s.runtime
	config, err := ReadConfig(r.gguf)
	if err != nil {
		return fmt.Errorf("cuda upstream load model config: %w", err)
	}
	if len(config.CompressRatio) != len(contract.Ratios) {
		return errors.New("cuda upstream load: model ratio count mismatch")
	}
	for i, ratio := range config.CompressRatio {
		if uint32(ratio) != contract.Ratios[i] {
			return fmt.Errorf("cuda upstream load: model ratio mismatch at layer %d", i)
		}
	}
	layout, err := dsv4kvstore.ValidateUpstreamRecordLayout(record, contract.Ratios)
	if err != nil {
		return fmt.Errorf("cuda upstream load: %w", err)
	}
	if err := dsv4kvstore.ValidateUpstreamRestoreContract(layout, contract); err != nil {
		return fmt.Errorf("cuda upstream load: %w", err)
	}
	if err := validateCUDAUpstreamTarget(layout, contract, &r.state, r.ctx, len(r.logits)); err != nil {
		return err
	}
	if err := validateCUDAUpstreamValues(record.Payload, layout, &r.state); err != nil {
		return err
	}
	if record.Metadata.QuantBits != 2 || record.Metadata.QuantBits != contract.QuantBits || len(record.Trailer) != 0 {
		return errors.New("cuda upstream load: unsupported quantization or trailer")
	}
	reader := bytes.NewReader(record.Payload)
	tokenHash, err := dsv4kvstore.HashUpstreamPayloadTokenIDs(reader, layout)
	if err != nil || tokenHash != contract.TokenIDsSHA256 {
		return errors.New("cuda upstream load: token provenance mismatch")
	}
	if r.file == nil {
		return errors.New("cuda upstream load: model file is unavailable")
	}
	info, err := r.file.Stat()
	if err != nil {
		return fmt.Errorf("cuda upstream load model: %w", err)
	}
	h := sha256.New()
	if _, err := io.Copy(h, io.NewSectionReader(r.file, 0, info.Size())); err != nil {
		return fmt.Errorf("cuda upstream load model hash: %w", err)
	}
	if !bytes.Equal(h.Sum(nil), contract.ModelSHA256[:]) {
		return errors.New("cuda upstream load: model provenance mismatch")
	}
	// A second snapshot is the staging allocation. No live buffer is used as
	// scratch, and the first snapshot remains unchanged until commit succeeds.
	backup, err := s.Snapshot()
	if err != nil {
		return fmt.Errorf("cuda upstream load backup: %w", err)
	}
	defer backup.Close()
	prepared, err := s.Snapshot()
	if err != nil {
		return fmt.Errorf("cuda upstream load stage: %w", err)
	}
	defer prepared.Close()
	v := prepared.(*cudaStateSnapshot)
	if err := stageCUDAUpstream(record.Payload, layout, &v.state, v.logits); err != nil {
		return fmt.Errorf("cuda upstream load stage: %w", err)
	}
	v.pos = int(layout.Metadata.Tokens)
	if err := syncStream(r.stream()); err != nil {
		return fmt.Errorf("cuda upstream load synchronize: %w", err)
	}
	if err := r.invalidateUpstreamGraphs(); err != nil {
		return err
	}
	if err := commitCUDAUpstreamState(func() error { return s.Restore(prepared) }, func() error { return s.Restore(backup) }); err != nil {
		if errors.Is(err, dsv4.ErrDesynchronized) {
			r.failed = err
		}
		return err
	}
	return nil
}

func commitCUDAUpstreamState(commit, rollback func() error) error {
	if err := commit(); err != nil {
		if restoreErr := rollback(); restoreErr != nil {
			return errors.Join(dsv4.ErrDesynchronized, err, fmt.Errorf("rollback upstream load: %w", restoreErr))
		}
		return err
	}
	return nil
}

func validateCUDAUpstreamTarget(layout dsv4kvstore.UpstreamPayloadLayout, c dsv4kvstore.UpstreamRestoreContract, state *cudaSessionState, context, vocab int) error {
	m := layout.Metadata
	if state == nil || context != int(m.Context) || vocab != int(m.Vocab) || len(state.layers) != int(m.Layers) || m.Tokens > uint32(context) || m.HeadDim != cudaKVWidth || m.IndexerHeadDim != 128 {
		return errors.New("cuda upstream load: target geometry mismatch")
	}
	for i := range state.layers {
		l := &state.layers[i]
		if l.rawKV.Len() != int(c.TargetRawCapacity)*cudaKVWidth || uint32(l.attn.ratio) != c.Ratios[i] {
			return fmt.Errorf("cuda upstream load: layer %d geometry mismatch", i)
		}
	}
	return nil
}

func upstreamFloatValues(data []byte) ([]float32, error) {
	if len(data)%4 != 0 {
		return nil, errors.New("unaligned float section")
	}
	v := make([]float32, len(data)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
	}
	return v, nil
}

func validateCUDAUpstreamValues(payload []byte, layout dsv4kvstore.UpstreamPayloadLayout, state *cudaSessionState) error {
	m := layout.Metadata
	end := uint64(52) + uint64(m.Tokens)*4 + uint64(m.Vocab)*4 + uint64(m.Layers)*8
	if end > uint64(len(payload)) || len(state.layers) != int(m.Layers) {
		return errors.New("cuda upstream load: truncated tokens, logits or counts")
	}
	for i := uint32(0); i < m.Tokens; i++ {
		if binary.LittleEndian.Uint32(payload[52+uint64(i)*4:]) >= m.Vocab {
			return fmt.Errorf("cuda upstream load: invalid token at %d", i)
		}
	}
	counts := 52 + int(m.Tokens)*4 + int(m.Vocab)*4
	logitStart := 52 + int(m.Tokens)*4
	for i := uint32(0); i < m.Vocab; i++ {
		v := math.Float32frombits(binary.LittleEndian.Uint32(payload[logitStart+int(i)*4:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return errors.New("cuda upstream load: nonfinite logits")
		}
	}
	for i := range state.layers {
		l := &state.layers[i]
		for _, item := range []struct{ offset, cap, ratio int }{
			{counts + 4*i, l.attn.cap, l.attn.ratio},
			{counts + 4*len(state.layers) + 4*i, l.indexer.cap, l.indexer.ratio},
		} {
			n := uint64(binary.LittleEndian.Uint32(payload[item.offset:]))
			if n > uint64(item.cap) || (item.ratio == 0 && n != 0) || (item.ratio != 0 && n != uint64(m.Tokens)/uint64(item.ratio)) {
				return fmt.Errorf("cuda upstream load: layer %d compressed count mismatch", i)
			}
		}
	}
	return nil
}

func stageCUDAUpstream(payload []byte, layout dsv4kvstore.UpstreamPayloadLayout, state *cudaSessionState, logits []float32) error {
	m := layout.Metadata
	// Neither unused cache rows nor the pooling scratch belongs to the
	// checkpoint. Clear them instead of inheriting the previous request.
	for i := range state.layers {
		l := &state.layers[i]
		for _, dst := range []cudamem.Slice[float32]{l.rawKV, l.attn.cache, l.attn.pooled, l.indexer.cache, l.indexer.pooled} {
			if dst.Len() != 0 {
				if err := dst.CopyFromHost(make([]float32, dst.Len())); err != nil {
					return err
				}
			}
		}
	}
	logitStart := 52 + int(m.Tokens)*4
	logitEnd := logitStart + int(m.Vocab)*4
	if logitEnd > len(payload) || len(logits) != int(m.Vocab) {
		return errors.New("logits geometry mismatch")
	}
	v, _ := upstreamFloatValues(payload[logitStart:logitEnd])
	copy(logits, v)
	if err := state.logits.CopyFromHost(v); err != nil {
		return err
	}
	counts := logitEnd
	for i := range state.layers {
		state.layers[i].attn.n = int(binary.LittleEndian.Uint32(payload[counts+4*i:]))
		state.layers[i].indexer.n = int(binary.LittleEndian.Uint32(payload[counts+4*len(state.layers)+4*i:]))
	}
	mappings, err := dsv4kvstore.MapUpstreamRawRows(m, uint32(state.layers[0].rawKV.Len()/cudaKVWidth))
	if err != nil {
		return err
	}
	for _, section := range layout.Sections {
		if section.Offset > uint64(len(payload)) || section.Bytes > uint64(len(payload))-section.Offset {
			return errors.New("section exceeds payload")
		}
		values, err := upstreamFloatValues(payload[section.Offset : section.Offset+section.Bytes])
		if err != nil {
			return err
		}
		l := &state.layers[section.Layer]
		var dst cudamem.Slice[float32]
		switch section.Name {
		case "raw":
			for i, row := range mappings {
				start := int(row.Slot) * cudaKVWidth
				if err := l.rawKV.Slice(start, start+cudaKVWidth).CopyFromHost(values[i*cudaKVWidth : (i+1)*cudaKVWidth]); err != nil {
					return err
				}
			}
			continue
		case "compressed":
			dst = l.attn.cache
		case "indexer":
			dst = l.indexer.cache
		case "attention_state_kv":
			dst = l.attn.stateKV
		case "attention_state_score":
			dst = l.attn.stateScore
		case "index_state_kv":
			dst = l.indexer.stateKV
		case "index_state_score":
			dst = l.indexer.stateScore
		default:
			return fmt.Errorf("unknown section %q", section.Name)
		}
		if len(values) > dst.Len() || (section.Name != "compressed" && section.Name != "indexer" && len(values) != dst.Len()) {
			return fmt.Errorf("layer %d section %s shape mismatch", section.Layer, section.Name)
		}
		if len(values) != 0 {
			if err := dst.Slice(0, len(values)).CopyFromHost(values); err != nil {
				return err
			}
		}
	}
	return nil
}

func (r *cudaRuntime) invalidateUpstreamGraphs() error {
	var errs []error
	for i := range r.graphs {
		if g := r.graphs[i].qkv; g != nil {
			errs = append(errs, g.Destroy())
			r.graphs[i].qkv = nil
		}
		if g := r.graphs[i].ffn; g != nil {
			errs = append(errs, g.Destroy())
			r.graphs[i].ffn = nil
		}
	}
	if r.batch != nil {
		errs = append(errs, r.batch.destroyGraphs())
	}
	return errors.Join(errs...)
}
