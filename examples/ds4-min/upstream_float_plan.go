package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4kvstore"
)

type upstreamFloatLayer struct {
	raw, cache, kv, score, pooled                []float32
	indexCache, indexKV, indexScore, indexPooled []float32
	ratio, indexRatio, capacity, indexCapacity   int
}

type upstreamFloatWrite struct {
	dst, values []float32
}

// planUpstreamFloatRestore never writes a destination. All payload parsing,
// shape checks and value checks complete before the caller can commit it.
func planUpstreamFloatRestore(payload []byte, layout dsv4kvstore.UpstreamPayloadLayout, layers []upstreamFloatLayer, logits []float32) ([]upstreamFloatWrite, []int, []int, error) {
	m := layout.Metadata
	if len(layers) != int(m.Layers) || len(logits) != int(m.Vocab) || m.HeadDim == 0 {
		return nil, nil, nil, errors.New("upstream float restore: target geometry mismatch")
	}
	logitStart := uint64(52) + uint64(m.Tokens)*4
	counts := logitStart + uint64(m.Vocab)*4
	if counts+uint64(m.Layers)*8 > uint64(len(payload)) {
		return nil, nil, nil, errors.New("upstream float restore: truncated tokens or counts")
	}
	for i := uint32(0); i < m.Tokens; i++ {
		if binary.LittleEndian.Uint32(payload[52+uint64(i)*4:]) >= m.Vocab {
			return nil, nil, nil, fmt.Errorf("upstream float restore: invalid token %d", i)
		}
	}
	ns, indexNs := make([]int, len(layers)), make([]int, len(layers))
	for i, l := range layers {
		ns[i] = int(binary.LittleEndian.Uint32(payload[counts+uint64(4*i):]))
		indexNs[i] = int(binary.LittleEndian.Uint32(payload[counts+uint64(4*len(layers)+4*i):]))
		for _, count := range []struct{ n, cap, ratio int }{{ns[i], l.capacity, l.ratio}, {indexNs[i], l.indexCapacity, l.indexRatio}} {
			if count.n > count.cap || (count.ratio == 0 && count.n != 0) || (count.ratio != 0 && count.n != int(m.Tokens)/count.ratio) {
				return nil, nil, nil, fmt.Errorf("upstream float restore: layer %d count mismatch", i)
			}
		}
	}
	var writes []upstreamFloatWrite
	add := func(dst []float32) int {
		writes = append(writes, upstreamFloatWrite{dst: dst, values: make([]float32, len(dst))})
		return len(writes) - 1
	}
	logitWrite := add(logits)
	decode := func(data []byte, allowNegativeInf bool) ([]float32, error) {
		if len(data)%4 != 0 {
			return nil, errors.New("upstream float restore: unaligned section")
		}
		values := make([]float32, len(data)/4)
		for i := range values {
			v := math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 1) || (!allowNegativeInf && math.IsInf(float64(v), -1)) {
				return nil, errors.New("upstream float restore: nonfinite value")
			}
			values[i] = v
		}
		return values, nil
	}
	v, err := decode(payload[logitStart:counts], false)
	if err != nil {
		return nil, nil, nil, err
	}
	copy(writes[logitWrite].values, v)
	indices := make([]map[string]int, len(layers))
	seen := make([]map[string]bool, len(layers))
	for i, l := range layers {
		seen[i] = make(map[string]bool)
		indices[i] = map[string]int{
			"raw": add(l.raw), "compressed": add(l.cache), "attention_state_kv": add(l.kv), "attention_state_score": add(l.score),
			"indexer": add(l.indexCache), "index_state_kv": add(l.indexKV), "index_state_score": add(l.indexScore),
		}
		add(l.pooled)
		add(l.indexPooled)
	}
	for _, section := range layout.Sections {
		if section.Layer < 0 || section.Layer >= len(layers) || section.Offset > uint64(len(payload)) || section.Bytes > uint64(len(payload))-section.Offset {
			return nil, nil, nil, errors.New("upstream float restore: section out of bounds")
		}
		idx, ok := indices[section.Layer][section.Name]
		if !ok || seen[section.Layer][section.Name] {
			return nil, nil, nil, fmt.Errorf("upstream float restore: unknown section %q", section.Name)
		}
		seen[section.Layer][section.Name] = true
		v, err := decode(payload[section.Offset:section.Offset+section.Bytes], section.Name == "attention_state_score" || section.Name == "index_state_score")
		if err != nil {
			return nil, nil, nil, err
		}
		dst := writes[idx].values
		if section.Name == "raw" {
			if len(dst)%int(m.HeadDim) != 0 || len(dst) == 0 || uint64(len(v)) != uint64(m.RawLive)*uint64(m.HeadDim) {
				return nil, nil, nil, errors.New("upstream float restore: raw shape mismatch")
			}
			rows, err := dsv4kvstore.MapUpstreamRawRows(m, uint32(len(dst)/int(m.HeadDim)))
			if err != nil {
				return nil, nil, nil, err
			}
			for j, row := range rows {
				copy(dst[int(row.Slot)*int(m.HeadDim):], v[j*int(m.HeadDim):(j+1)*int(m.HeadDim)])
			}
			continue
		}
		if section.Name == "compressed" && len(v) != ns[section.Layer]*int(m.HeadDim) || section.Name == "indexer" && len(v) != indexNs[section.Layer]*int(m.IndexerHeadDim) {
			return nil, nil, nil, errors.New("upstream float restore: compressed rows shape mismatch")
		}
		if len(v) > len(dst) || (section.Name != "compressed" && section.Name != "indexer" && len(v) != len(dst)) {
			return nil, nil, nil, fmt.Errorf("upstream float restore: layer %d %s shape mismatch", section.Layer, section.Name)
		}
		copy(dst, v)
	}
	for i, l := range layers {
		required := []string{"raw"}
		if l.ratio != 0 {
			required = append(required, "compressed", "attention_state_kv", "attention_state_score")
		}
		if l.indexRatio != 0 {
			required = append(required, "indexer", "index_state_kv", "index_state_score")
		}
		for _, name := range required {
			if !seen[i][name] {
				return nil, nil, nil, fmt.Errorf("upstream float restore: missing layer %d section %s", i, name)
			}
		}
	}
	return writes, ns, indexNs, nil
}

// commitUpstreamFloatRestore retains old bytes until every write succeeds.
// apply is explicit so a host mutation test can fail after a partial commit.
func commitUpstreamFloatRestore(writes []upstreamFloatWrite, apply func([]float32, []float32) error) error {
	backups := make([][]float32, len(writes))
	for i, w := range writes {
		backups[i] = append([]float32(nil), w.dst...)
	}
	for _, w := range writes {
		if err := apply(w.dst, w.values); err != nil {
			var rollback error
			for i, old := range backups {
				rollback = errors.Join(rollback, apply(writes[i].dst, old))
			}
			if rollback != nil {
				return errors.Join(dsv4.ErrDesynchronized, err, rollback)
			}
			return err
		}
	}
	return nil
}
