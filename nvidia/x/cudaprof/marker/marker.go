package marker

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// Prefix identifies version 1 application metadata in an NVTX message.
const Prefix = "cudaprof:v1 "

// ErrNotMarker reports an NVTX message without the application marker prefix.
var ErrNotMarker = errors.New("not an application marker")

// Metadata describes application state observed through explicit instrumentation.
// Phase is required and is either prefill or decode. Nil optional fields mean
// that the application did not supply a value; they do not imply zero or false.
// Operation is an application-defined name, not an inferred kernel identity.
type Metadata struct {
	Phase           string  `json:"phase"`
	Token           *int    `json:"token,omitempty"`
	Layer           *int    `json:"layer,omitempty"`
	Operation       string  `json:"operation,omitempty"`
	DType           *string `json:"dtype,omitempty"`
	CacheLength     *int    `json:"cache_length,omitempty"`
	Lookahead       *bool   `json:"lookahead,omitempty"`
	Materialization *bool   `json:"materialization,omitempty"`
}

// Encode validates m and returns an NVTX range message.
func Encode(m Metadata) (string, error) {
	if err := validate(m); err != nil {
		return "", err
	}
	b, err := json.Marshal(m)
	if err != nil {
		return "", fmt.Errorf("encode marker: %w", err)
	}
	return Prefix + string(b), nil
}

// Parse reads an application marker from the original NVTX message. It rejects
// unknown fields, duplicate fields, unsupported versions, and trailing data.
// Callers must retain the original message separately for capture identity.
func Parse(message string) (Metadata, error) {
	var m Metadata
	if !strings.HasPrefix(message, Prefix) {
		if strings.HasPrefix(message, "cudaprof:") {
			return m, errors.New("parse marker: unsupported version")
		}
		return m, ErrNotMarker
	}
	body := strings.TrimPrefix(message, Prefix)
	// Reject duplicate keys instead of silently accepting the last observation.
	dec := json.NewDecoder(strings.NewReader(body))
	t, err := dec.Token()
	if err != nil || t != json.Delim('{') {
		return m, errors.New("parse marker: expected object")
	}
	seen := make(map[string]bool)
	for dec.More() {
		t, err := dec.Token()
		if err != nil {
			return m, fmt.Errorf("parse marker: %w", err)
		}
		key, ok := t.(string)
		if !ok || seen[key] {
			return m, errors.New("parse marker: duplicate or invalid field")
		}
		switch key {
		case "phase", "token", "layer", "operation", "dtype", "cache_length", "lookahead", "materialization":
		default:
			return m, fmt.Errorf("parse marker: unknown field %q", key)
		}
		seen[key] = true
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return m, fmt.Errorf("parse marker: %w", err)
		}
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return m, fmt.Errorf("parse marker: null field %q", key)
		}
	}
	if _, err := dec.Token(); err != nil {
		return m, fmt.Errorf("parse marker: %w", err)
	}
	dec = json.NewDecoder(strings.NewReader(body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return Metadata{}, fmt.Errorf("parse marker: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		return Metadata{}, errors.New("parse marker: trailing data")
	}
	if err := validate(m); err != nil {
		return Metadata{}, err
	}
	return m, nil
}

func validate(m Metadata) error {
	if m.Phase != "prefill" && m.Phase != "decode" {
		return errors.New("marker phase must be prefill or decode")
	}
	if m.Token != nil && *m.Token < 0 {
		return errors.New("marker token must be nonnegative")
	}
	if m.Layer != nil && *m.Layer < 0 {
		return errors.New("marker layer must be nonnegative")
	}
	if m.DType != nil && strings.TrimSpace(*m.DType) == "" {
		return errors.New("marker dtype must be nonempty")
	}
	if m.CacheLength != nil && *m.CacheLength < 0 {
		return errors.New("marker cache length must be nonnegative")
	}
	return nil
}

// ParseRange parses a timeline range name, which may prepend an NVTX domain
// followed by a slash. A domain-qualified name retains the same semantics as
// its original message. Ordinary range names return ErrNotMarker.
func ParseRange(name string) (Metadata, error) {
	if strings.HasPrefix(name, "cudaprof:") {
		return Parse(name)
	}
	if i := strings.Index(name, "/cudaprof:"); i >= 0 {
		return Parse(name[i+1:])
	}
	return Metadata{}, ErrNotMarker
}
