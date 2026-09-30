package tracecontract

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// Source identifies retained evidence in the exact capture bytes parsed.
// Sequence is a raw journal sequence. Index is available only when an older
// record has no sequence; its Kind identifies the indexed decoder collection.
// Exactly one of Sequence and Index must be supplied.
// The zero value lacks the identity required for encoding.
type Source struct {
	CaptureSHA256 string  `json:"capture_sha256"`
	Journal       string  `json:"journal"`
	Kind          string  `json:"kind"`
	Sequence      uint64  `json:"sequence,omitempty"`
	Index         *uint64 `json:"index,omitempty"`
}

// String returns compact, deterministic JSON for a stable source reference.
// It rejects missing identity rather than referring to an imported slice ID.
func (s Source) String() (string, error) {
	if !validHash(s.CaptureSHA256) || s.Journal == "" || s.Kind == "" || (s.Sequence != 0) == (s.Index != nil) {
		return "", fmt.Errorf("trace source: missing or ambiguous identity")
	}
	switch s.Kind {
	case "activity", "api", "marker", "range", "graph_lifecycle", "dependency", "memory", "telemetry", "gemm", "host":
	default:
		return "", fmt.Errorf("trace source: unsupported kind %q", s.Kind)
	}
	data, err := json.Marshal(s)
	return string(data), err
}

func validHash(hash string) bool {
	if len(hash) != 64 {
		return false
	}
	decoded, err := hex.DecodeString(hash)
	return err == nil && len(decoded) == 32 && hash == fmt.Sprintf("%x", decoded)
}
