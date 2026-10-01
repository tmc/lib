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

const prefixV2 = "cudaprof:v2 "

// ErrNotMarker reports an NVTX message without the application marker prefix.
var ErrNotMarker = errors.New("not an application marker")

// Metadata describes application state observed through explicit instrumentation.
// Phase is required: prefill or decode, with cleanup also supported by v2.
// Version 2 requires RequestID and Role. Nil optional fields mean
// that the application did not supply a value; they do not imply zero or false.
// Operation is an application-defined name, not an inferred kernel identity.
type Metadata struct {
	// Version is 0 or 1 for legacy metadata, or 2 for request-scoped metadata.
	Version   int    `json:"marker_version,omitempty"`
	RequestID string `json:"request_id,omitempty"`
	Role      string `json:"role,omitempty"`
	Boundary  string `json:"boundary,omitempty"`
	// ExecutionUnit declares the unit of the optional cumulative counter.
	ExecutionUnit string `json:"execution_unit,omitempty"`
	// Executions counts cumulative successful application executions in this
	// request, including prefill, with ExecutionUnit logical_forward. Each
	// successful logical model forward counts once. Tracing callbacks and failed
	// calls are excluded; these are not GPU completion or kernel counts.
	Executions      *int    `json:"executions,omitempty"`
	Phase           string  `json:"phase"`
	Event           string  `json:"event,omitempty"`
	Token           *int    `json:"token,omitempty"`
	Layer           *int    `json:"layer,omitempty"`
	Operation       string  `json:"operation,omitempty"`
	DType           *string `json:"dtype,omitempty"`
	CacheLength     *int    `json:"cache_length,omitempty"`
	Lookahead       *bool   `json:"lookahead,omitempty"`
	Materialization *bool   `json:"materialization,omitempty"`
}

// Encode validates m and returns an NVTX message.
func Encode(m Metadata) (string, error) {
	if err := validate(m); err != nil {
		return "", err
	}
	prefix := Prefix
	if m.Version == 2 {
		prefix = prefixV2
	}
	// The wire prefix carries the version; JSON receipts retain it separately.
	m.Version = 0
	b, err := json.Marshal(m)
	if err != nil {
		return "", fmt.Errorf("encode marker: %w", err)
	}
	return prefix + string(b), nil
}

// Parse reads an application marker from the original NVTX message. It rejects
// unknown fields, duplicate fields, unsupported versions, and trailing data.
// Callers must retain the original message separately for capture identity.
func Parse(message string) (Metadata, error) {
	var m Metadata
	prefix := Prefix
	version := 1
	if strings.HasPrefix(message, prefixV2) {
		prefix, version = prefixV2, 2
	}
	if !strings.HasPrefix(message, prefix) {
		if strings.HasPrefix(message, "cudaprof:") {
			return m, errors.New("parse marker: unsupported version")
		}
		return m, ErrNotMarker
	}
	body := strings.TrimPrefix(message, prefix)
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
		case "request_id", "role", "boundary", "executions", "execution_unit":
			if version != 2 {
				return m, fmt.Errorf("parse marker: unknown field %q", key)
			}
		case "phase", "token", "layer", "operation", "dtype", "cache_length", "lookahead", "materialization", "event":
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
	m.Version = version
	if err := validate(m); err != nil {
		return Metadata{}, err
	}
	return m, nil
}

func validate(m Metadata) error {
	if m.Version != 0 && m.Version != 1 && m.Version != 2 {
		return errors.New("marker version must be 1 or 2")
	}
	if m.Version != 2 && (m.RequestID != "" || m.Role != "" || m.Boundary != "" || m.Executions != nil || m.ExecutionUnit != "") {
		return errors.New("request metadata requires marker version 2")
	}
	if m.Version == 2 {
		if strings.TrimSpace(m.RequestID) == "" {
			return errors.New("marker request id must be nonempty")
		}
		if m.Role != "warmup" && m.Role != "measured" && m.Role != "control" {
			return errors.New("marker role must be warmup, measured or control")
		}
		if m.Boundary != "" && m.Boundary != "submission" && m.Boundary != "completion" && m.Boundary != "consumer_delivery" {
			return errors.New("marker boundary must be submission, completion or consumer_delivery")
		}
		if m.ExecutionUnit != "" && m.ExecutionUnit != "logical_forward" {
			return errors.New("marker execution unit must be logical_forward")
		}
		if m.Executions != nil && m.ExecutionUnit != "logical_forward" {
			return errors.New("marker executions require logical_forward unit")
		}
		if m.Executions != nil && *m.Executions < 0 {
			return errors.New("marker executions must be nonnegative")
		}
	}
	if m.Event != "" && m.Event != "delivered" && m.Event != "eos" && m.Event != "token_limit" && !(m.Version == 2 && (m.Event == "request_start" || m.Event == "request_end")) {
		return errors.New("marker event is unsupported for this version")
	}
	if m.Event != "" && m.Event != "request_start" && m.Event != "request_end" && m.Token == nil {
		return errors.New("marker event requires token")
	}
	if m.Phase != "prefill" && m.Phase != "decode" && !(m.Version == 2 && m.Phase == "cleanup") {
		return errors.New("marker phase is unsupported for this version")
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
