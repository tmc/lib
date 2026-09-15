package hfsync

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
)

const (
	// ALPN is the Iroh Application-Layer Protocol Negotiation identifier for hfsync.
	ALPN = "tmc/hfsync/v1"

	// DefaultChunkSize is the standard chunk payload size (2 MB) used for streaming large weight tensors.
	DefaultChunkSize = 2 * 1024 * 1024

	// Message type identifiers
	msgTypeManifestReq    byte = 0x01
	msgTypeManifestResp   byte = 0x02
	msgTypeChunkReq       byte = 0x03
	msgTypeChunkResp      byte = 0x04
	msgTypeListModelsReq  byte = 0x05
	msgTypeListModelsResp byte = 0x06
)

// ModelSummary provides a high-level summary of a model snapshot in the cache.
type ModelSummary struct {
	ModelID    string `json:"model_id"`
	Revision   string `json:"revision"`
	TotalBytes int64  `json:"total_bytes"`
	FileCount  int    `json:"file_count"`
}

// ListModelsRequest queries available models on a node.
type ListModelsRequest struct {
	Filter string `json:"filter,omitempty"`
}

// ListModelsResponse returns the list of available model summaries.
type ListModelsResponse struct {
	Models []ModelSummary `json:"models"`
}

// FileInfo describes a single file within a model snapshot.
type FileInfo struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

// Manifest represents the complete file layout and metadata of a model snapshot.
type Manifest struct {
	ModelID  string     `json:"model_id"`
	Revision string     `json:"revision"`
	Files    []FileInfo `json:"files"`
}

// ManifestRequest is sent by a client to query available files for a model.
type ManifestRequest struct {
	ModelID string `json:"model_id"`
}

// ChunkRequest requests a byte range of a specific file.
type ChunkRequest struct {
	ModelID string `json:"model_id"`
	Path    string `json:"path"`
	Offset  int64  `json:"offset"`
	Length  int64  `json:"length"`
}

// ChunkHeader prefixes binary chunk data in responses.
type ChunkHeader struct {
	Status string `json:"status"` // "ok" or error string
	Offset int64  `json:"offset"`
	Length int64  `json:"length"`
}

func writeFrame(w io.Writer, msgType byte, payload []byte) error {
	var header [5]byte
	header[0] = msgType
	binary.BigEndian.PutUint32(header[1:], uint32(len(payload)))
	if _, err := w.Write(header[:]); err != nil {
		return fmt.Errorf("write frame header: %w", err)
	}
	if len(payload) > 0 {
		if _, err := w.Write(payload); err != nil {
			return fmt.Errorf("write frame payload: %w", err)
		}
	}
	return nil
}

func readFrame(r io.Reader) (byte, []byte, error) {
	var header [5]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return 0, nil, fmt.Errorf("read frame header: %w", err)
	}
	msgType := header[0]
	length := binary.BigEndian.Uint32(header[1:])

	payload := make([]byte, length)
	if length > 0 {
		if _, err := io.ReadFull(r, payload); err != nil {
			return 0, nil, fmt.Errorf("read frame payload: %w", err)
		}
	}
	return msgType, payload, nil
}

func writeJSONFrame(w io.Writer, msgType byte, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal json frame: %w", err)
	}
	return writeFrame(w, msgType, data)
}

func readJSONFrame[T any](r io.Reader, expectedType byte) (*T, error) {
	msgType, payload, err := readFrame(r)
	if err != nil {
		return nil, err
	}
	if msgType != expectedType {
		return nil, fmt.Errorf("unexpected frame type: got 0x%02x, want 0x%02x", msgType, expectedType)
	}
	var out T
	if err := json.Unmarshal(payload, &out); err != nil {
		return nil, fmt.Errorf("unmarshal json frame: %w", err)
	}
	return &out, nil
}
