package main

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"

	"github.com/tmc/lib/gguf"
)

// modelSummary is the backend-independent part of -inspect. It describes the
// container and tensor layout without claiming that the model is executable.
type modelSummary struct {
	Schema         string         `json:"schema"`
	Model          string         `json:"model"`
	FileBytes      uint64         `json:"file_bytes"`
	DataOffset     uint64         `json:"data_offset"`
	MetadataCount  int            `json:"metadata_count"`
	TensorCount    int            `json:"tensor_count"`
	Architecture   string         `json:"architecture,omitempty"`
	Variant        string         `json:"checkpoint_variant,omitempty"`
	ContextLength  uint64         `json:"context_length,omitempty"`
	VocabularySize uint64         `json:"vocabulary_size,omitempty"`
	BlockCount     uint64         `json:"block_count,omitempty"`
	Quantization   []quantSummary `json:"quantization"`
}

type quantSummary struct {
	TypeID  uint32 `json:"type_id"`
	Tensors uint64 `json:"tensors"`
	Bytes   uint64 `json:"bytes"`
}

// inspectModel writes one JSON object followed by a newline. It only parses
// GGUF and never opens a backend.
func inspectModel(w io.Writer, path string, g *gguf.File) error {
	if g == nil {
		return fmt.Errorf("inspect: missing GGUF")
	}
	s := modelSummary{
		Schema: "ds4-min-model-summary-v1", Model: path,
		FileBytes: g.Size, DataOffset: g.DataOffset,
		MetadataCount: len(g.Metadata), TensorCount: len(g.Tensors),
	}
	for key, value := range g.Metadata {
		switch key {
		case "general.architecture":
			if v, ok := value.Value.(string); ok {
				s.Architecture = v
			}
		case "deepseek4.checkpoint_variant":
			if v, ok := value.Value.(string); ok {
				s.Variant = v
			}
		case "general.context_length", "deepseek4.context_length":
			if v, ok := value.Uint(); ok {
				s.ContextLength = v
			}
		case "general.vocabulary_size", "deepseek4.vocab_size":
			if v, ok := value.Uint(); ok {
				s.VocabularySize = v
			}
		case "deepseek4.block_count":
			if v, ok := value.Uint(); ok {
				s.BlockCount = v
			}
		}
	}
	byType := make(map[uint32]quantSummary)
	for _, tensor := range g.Tensors {
		n, err := tensor.Bytes()
		if err != nil {
			return fmt.Errorf("inspect tensor %q: %w", tensor.Name, err)
		}
		q := byType[tensor.TypeID]
		q.TypeID, q.Tensors = tensor.TypeID, q.Tensors+1
		if ^uint64(0)-q.Bytes < n {
			return fmt.Errorf("inspect quantization type %d: byte count overflow", tensor.TypeID)
		}
		q.Bytes += n
		byType[tensor.TypeID] = q
	}
	types := make([]int, 0, len(byType))
	for typeID := range byType {
		types = append(types, int(typeID))
	}
	sort.Ints(types)
	for _, typeID := range types {
		s.Quantization = append(s.Quantization, byType[uint32(typeID)])
	}
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(s); err != nil {
		return fmt.Errorf("inspect summary: %w", err)
	}
	return nil
}
