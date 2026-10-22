//go:build linux

package main

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4kvstore"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// verifyCUDAUpstreamImport reads the imported values before their first use.
// Raw rows are compared in checkpoint order, not physical ring-slot order.
// It does not infer continuation agreement from successful transport.
func verifyCUDAUpstreamImport(payload []byte, layout dsv4kvstore.UpstreamPayloadLayout, state *cudaSessionState) error {
	m := layout.Metadata
	counts := 52 + int(m.Tokens)*4 + int(m.Vocab)*4
	if counts > len(payload) || len(state.layers) == 0 || len(state.layers) != int(m.Layers) || len(payload)-counts < len(state.layers)*8 {
		return fmt.Errorf("import readback: counts geometry mismatch")
	}
	for i := range state.layers {
		for _, item := range []struct {
			name        string
			offset, got int
		}{
			{"compressed", counts + 4*i, state.layers[i].attn.n},
			{"indexer", counts + 4*len(state.layers) + 4*i, state.layers[i].indexer.n},
		} {
			want := binary.LittleEndian.Uint32(payload[item.offset:])
			if item.got < 0 || uint64(item.got) != uint64(want) {
				return fmt.Errorf("import readback: layer %d %s count got %d want %d", i, item.name, item.got, want)
			}
		}
	}
	ratios := make([]int, len(state.layers))
	capacity := state.layers[0].rawKV.Len() / cudaKVWidth
	for i := range state.layers {
		ratios[i] = state.layers[i].attn.ratio
		if capacity == 0 || state.layers[i].rawKV.Len()%cudaKVWidth != 0 || state.layers[i].rawKV.Len()/cudaKVWidth != capacity {
			return fmt.Errorf("import readback: layer %d raw capacity mismatch", i)
		}
	}
	if err := validateCUDAUpstreamSectionCensus(layout.Sections, ratios); err != nil {
		return err
	}
	mappings, err := dsv4kvstore.MapUpstreamRawRows(m, uint32(state.layers[0].rawKV.Len()/cudaKVWidth))
	if err != nil {
		return err
	}
	for i, row := range mappings {
		position := m.Tokens - m.RawLive + uint32(i)
		// The consumer selects the logical position modulo the target ring
		// capacity. Check independently of the importer's shared map helper.
		if row.Position != position || row.Slot != position%uint32(capacity) {
			return fmt.Errorf("import readback: raw row %d logical slot mismatch", i)
		}
	}
	for _, section := range layout.Sections {
		if section.Layer < 0 || section.Layer >= len(state.layers) {
			return fmt.Errorf("import readback: invalid layer %d", section.Layer)
		}
		l := &state.layers[section.Layer]
		var src cudamem.Slice[float32]
		switch section.Name {
		case "raw":
			src = l.rawKV
		case "compressed":
			src = l.attn.cache
		case "indexer":
			src = l.indexer.cache
		case "attention_state_kv":
			src = l.attn.stateKV
		case "attention_state_score":
			src = l.attn.stateScore
		case "index_state_kv":
			src = l.indexer.stateKV
		case "index_state_score":
			src = l.indexer.stateScore
		default:
			return fmt.Errorf("import readback: unknown section %q", section.Name)
		}
		if section.Offset > uint64(len(payload)) || section.Bytes > uint64(len(payload))-section.Offset || section.Bytes%4 != 0 {
			return fmt.Errorf("import readback: section exceeds payload")
		}
		got := make([]float32, int(section.Bytes/4))
		if section.Name == "raw" {
			if len(got) != len(mappings)*cudaKVWidth {
				return fmt.Errorf("import readback: raw geometry mismatch")
			}
			for i, row := range mappings {
				start := int(row.Slot) * cudaKVWidth
				if err := src.Slice(start, start+cudaKVWidth).CopyToHost(got[i*cudaKVWidth : (i+1)*cudaKVWidth]); err != nil {
					return fmt.Errorf("import readback: layer %d raw position %d: %w", section.Layer, row.Position, err)
				}
			}
		} else if len(got) > src.Len() {
			return fmt.Errorf("import readback: layer %d %s geometry mismatch", section.Layer, section.Name)
		} else if len(got) != 0 {
			if err := src.Slice(0, len(got)).CopyToHost(got); err != nil {
				return fmt.Errorf("import readback: layer %d %s: %w", section.Layer, section.Name, err)
			}
		}
		if err := compareCUDAUpstreamSection(section.Layer, section.Name, payload[section.Offset:section.Offset+section.Bytes], got); err != nil {
			return err
		}
	}
	return nil
}

func validateCUDAUpstreamSectionCensus(sections []dsv4kvstore.UpstreamPayloadSection, ratios []int) error {
	seen := make(map[string]bool)
	for _, section := range sections {
		if section.Layer < 0 || section.Layer >= len(ratios) {
			return fmt.Errorf("import readback: invalid layer %d", section.Layer)
		}
		key := fmt.Sprintf("%d/%s", section.Layer, section.Name)
		if seen[key] {
			return fmt.Errorf("import readback: duplicate section %s", key)
		}
		seen[key] = true
	}
	for layer, ratio := range ratios {
		names := []string{"raw"}
		if ratio != 0 {
			names = append(names, "compressed", "attention_state_kv", "attention_state_score")
		}
		if ratio == 4 {
			names = append(names, "indexer", "index_state_kv", "index_state_score")
		}
		for _, name := range names {
			key := fmt.Sprintf("%d/%s", layer, name)
			if !seen[key] {
				return fmt.Errorf("import readback: missing section %s", key)
			}
			delete(seen, key)
		}
	}
	if len(seen) != 0 {
		return fmt.Errorf("import readback: unexpected section")
	}
	return nil
}

func compareCUDAUpstreamSection(layer int, name string, want []byte, got []float32) error {
	if len(want)%4 != 0 || len(want)/4 != len(got) {
		return fmt.Errorf("import readback: layer %d %s length mismatch", layer, name)
	}
	for i, value := range got {
		w, g := binary.LittleEndian.Uint32(want[4*i:]), math.Float32bits(value)
		if w != g {
			return fmt.Errorf("import readback: layer %d %s element %d got %08x want %08x", layer, name, i, g, w)
		}
	}
	return nil
}
