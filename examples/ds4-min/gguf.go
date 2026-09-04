package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/tmc/lib/gguf"
)

// The pinned checkpoint. ds4-min runs against one exact model file, so it
// checks the file size, the manifest digests of the parsed container, and the
// digest of the file itself before touching any weights.
const (
	pinnedModelBytes    = 86720111488
	pinnedModelSHA256   = "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
	pinnedMetadataHash  = "857d48d201da0faaaa6f2af0cdf0aa5fef523323217e29a902579130c24a32b5"
	pinnedTensorHash    = "0bcce58e7234ba89f247bfab49387121f375e8322d843fc166fb95ed12db99ca"
	pinnedQuantHash     = "b07c8ece42d1ff0273a8b7db85514e66ae6aca48dc55ea5b33d50f8f02380788"
	pinnedTokenizerHash = "d58f8cbcd40bfa9319e0954e01ad619c179567cf001258423713d3d604998eec"
)

type profileDigests struct {
	metadata  string
	tensors   string
	quant     string
	tokenizer string
}

type metadataEntry struct {
	Key    string `json:"key"`
	TypeID uint32 `json:"type_id"`
	Value  any    `json:"value"`
}

type tensorEntry struct {
	Name       string   `json:"name"`
	Dimensions []uint64 `json:"dimensions"`
	TypeID     uint32   `json:"type_id"`
	Offset     uint64   `json:"offset"`
	Bytes      uint64   `json:"bytes"`
}

type quantEntry struct {
	Bytes   uint64 `json:"bytes"`
	Tensors uint64 `json:"tensors"`
	TypeID  uint32 `json:"type_id"`
}

func validatePinnedProfile(g *gguf.File) error {
	if g.Size != pinnedModelBytes {
		return fmt.Errorf("model size %d, want %d", g.Size, uint64(pinnedModelBytes))
	}
	d, err := profileHashes(g)
	if err != nil {
		return err
	}
	checks := []struct {
		name, got, want string
	}{
		{"metadata", d.metadata, pinnedMetadataHash},
		{"tensor", d.tensors, pinnedTensorHash},
		{"quantization", d.quant, pinnedQuantHash},
		{"tokenizer", d.tokenizer, pinnedTokenizerHash},
	}
	for _, check := range checks {
		if check.got != check.want {
			return fmt.Errorf("%s manifest sha256 %s, want %s", check.name, check.got, check.want)
		}
	}
	return nil
}

func validatePinnedModel(r io.ReaderAt, size int64) (*gguf.File, error) {
	g, err := gguf.Read(r, size)
	if err != nil {
		return nil, err
	}
	if err := validatePinnedProfile(g); err != nil {
		return nil, err
	}
	if err := gguf.CheckSHA256(r, size, pinnedModelSHA256); err != nil {
		return nil, err
	}
	return g, nil
}

func profileHashes(g *gguf.File) (profileDigests, error) {
	const schema = "ds4-min-gguf-manifest-v1"
	keys := make([]string, 0, len(g.Metadata))
	for key := range g.Metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	metadata := make([]metadataEntry, 0, len(keys))
	tokenizer := make([]metadataEntry, 0)
	for _, key := range keys {
		v := g.Metadata[key]
		e := metadataEntry{key, v.TypeID, v.Value}
		metadata = append(metadata, e)
		if strings.HasPrefix(key, "tokenizer.") {
			tokenizer = append(tokenizer, e)
		}
	}
	tensors := make([]tensorEntry, 0, len(g.Tensors))
	quantByType := make(map[uint32]quantEntry)
	for _, t := range g.Tensors {
		n, err := t.Bytes()
		if err != nil {
			return profileDigests{}, err
		}
		tensors = append(tensors, tensorEntry{t.Name, t.Dimensions, t.TypeID, t.Offset, n})
		q := quantByType[t.TypeID]
		q.TypeID = t.TypeID
		q.Tensors++
		if q.Bytes > ^uint64(0)-n {
			return profileDigests{}, fmt.Errorf("quantization type %d byte count overflow", t.TypeID)
		}
		q.Bytes += n
		quantByType[t.TypeID] = q
	}
	types := make([]int, 0, len(quantByType))
	for typeID := range quantByType {
		types = append(types, int(typeID))
	}
	sort.Ints(types)
	quant := make([]quantEntry, 0, len(types))
	for _, typeID := range types {
		quant = append(quant, quantByType[uint32(typeID)])
	}
	values := []any{
		struct {
			Schema     string          `json:"schema"`
			FileBytes  uint64          `json:"file_bytes"`
			DataOffset uint64          `json:"data_offset"`
			Metadata   []metadataEntry `json:"metadata"`
		}{schema, g.Size, g.DataOffset, metadata},
		struct {
			Schema     string        `json:"schema"`
			FileBytes  uint64        `json:"file_bytes"`
			DataOffset uint64        `json:"data_offset"`
			Tensors    []tensorEntry `json:"tensors"`
		}{schema, g.Size, g.DataOffset, tensors},
		struct {
			Schema string       `json:"schema"`
			Types  []quantEntry `json:"types"`
		}{schema, quant},
		struct {
			Metadata []metadataEntry `json:"metadata"`
			Schema   string          `json:"schema"`
		}{tokenizer, schema},
	}
	var d profileDigests
	out := []*string{&d.metadata, &d.tensors, &d.quant, &d.tokenizer}
	for i, value := range values {
		b, err := json.Marshal(value)
		if err != nil {
			return profileDigests{}, fmt.Errorf("encode profile manifest: %w", err)
		}
		sum := sha256.Sum256(b)
		*out[i] = hex.EncodeToString(sum[:])
	}
	return d, nil
}
