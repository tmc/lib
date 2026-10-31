package cudashadow

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

const historyVersion = 1
const descriptionHistoryVersion = 2
const maxHistoryBytes = 16 << 20
const maxHistorySnapshots = 4096

type historyWire struct {
	Version   int        `json:"version"`
	Snapshots []Snapshot `json:"snapshots"`
}

// WriteHistory writes a version 1 or 2 offline replay corpus. It validates every
// snapshot and refuses more than 4096 snapshots or 16 MiB of encoded data
// before encoding the history, without writing to w.
// A stateful policy requires a complete history from its initial state;
// a wrapped evaluator's retained suffix is not necessarily replayable.
func WriteHistory(w io.Writer, snapshots []Snapshot) error {
	if w == nil {
		return errors.New("cudashadow: nil history writer")
	}
	version, size, err := checkHistory(snapshots, "cudashadow history snapshot")
	if err != nil {
		return err
	}
	buf := bytes.NewBuffer(make([]byte, 0, size))
	fmt.Fprintf(buf, `{"version":%d,"snapshots":`, version)
	if snapshots == nil {
		buf.WriteString("null")
	} else {
		enc := json.NewEncoder(buf)
		buf.WriteByte('[')
		for i, snapshot := range snapshots {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := enc.Encode(snapshot); err != nil {
				return err
			}
			buf.Truncate(buf.Len() - 1) // newline added by Encode
		}
		buf.WriteByte(']')
	}
	buf.WriteByte('}')
	data := buf.Bytes()
	n, err := w.Write(data)
	if err != nil {
		return err
	}
	if n != len(data) {
		return io.ErrShortWrite
	}
	return nil
}

// checkHistory validates snapshots against the history bounds and returns
// the minimal history version and the exact size of the compact encoding.
// It refuses more than 4096 snapshots or 16 MiB, encoding one validated,
// individually bounded snapshot at a time rather than the whole history.
// Validation errors are prefixed by step and the snapshot index.
func checkHistory(snapshots []Snapshot, step string) (version, size int, err error) {
	if len(snapshots) > maxHistorySnapshots {
		return 0, 0, errors.New("cudashadow: history capacity exceeded")
	}
	version = historyVersion
	for i, snapshot := range snapshots {
		if successorSnapshot(snapshot) {
			version = descriptionHistoryVersion
		}
		if err := snapshot.Validate(); err != nil {
			return 0, 0, fmt.Errorf("%s %d: %w", step, i, err)
		}
	}
	if snapshots == nil {
		return version, len(`{"version":1,"snapshots":null}`), nil
	}
	// Brackets, braces and commas; each Encode also adds a newline.
	size = len(`{"version":1,"snapshots":[]}`) + max(len(snapshots)-1, 0)
	var n byteCounter
	enc := json.NewEncoder(&n)
	for i, snapshot := range snapshots {
		if err := enc.Encode(snapshot); err != nil {
			return 0, 0, err
		}
		if size+int(n)-(i+1) > maxHistoryBytes {
			return 0, 0, errors.New("cudashadow: history exceeds 16 MiB")
		}
	}
	return version, size + int(n) - len(snapshots), nil
}

// A byteCounter is an io.Writer that counts the bytes written to it.
type byteCounter int64

func (c *byteCounter) Write(p []byte) (int, error) {
	*c += byteCounter(len(p))
	return len(p), nil
}

// ReadHistory reads a bounded version 1 or 2 replay corpus, rejecting unknown
// fields, invalid snapshots, unsupported versions, and trailing data.
func ReadHistory(r io.Reader) ([]Snapshot, error) {
	if r == nil {
		return nil, errors.New("cudashadow: nil history reader")
	}
	data, err := io.ReadAll(io.LimitReader(r, maxHistoryBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxHistoryBytes {
		return nil, errors.New("cudashadow: history exceeds 16 MiB")
	}
	if err := uniqueJSON(data); err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var wire historyWire
	if err := decoder.Decode(&wire); err != nil {
		return nil, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return nil, errors.New("cudashadow: trailing history data")
	}
	if wire.Version != historyVersion && wire.Version != descriptionHistoryVersion {
		return nil, errors.New("cudashadow: unsupported history version")
	}
	if len(wire.Snapshots) > maxHistorySnapshots {
		return nil, errors.New("cudashadow: history capacity exceeded")
	}
	if wire.Version == historyVersion {
		// Reject even zero-valued successor fields disguised as the old schema.
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, err
		}

		for name, value := range raw {
			if !strings.EqualFold(name, "snapshots") {
				continue
			}
			var list []map[string]json.RawMessage
			if err := json.Unmarshal(value, &list); err != nil {
				return nil, err
			}
			for _, snapshot := range list {
				if successorJSON(snapshot, "snapshot") {
					return nil, errors.New("cudashadow: successor fields in version 1 history")
				}
			}
		}
	}
	for i, snapshot := range wire.Snapshots {
		if err := snapshot.Validate(); err != nil {
			return nil, fmt.Errorf("cudashadow history snapshot %d: %w", i, err)
		}
	}
	return wire.Snapshots, nil
}

func successorSnapshot(s Snapshot) bool {
	return s.Provenance.Kind != "" || s.Provenance.Source != "" || s.Input.Provenance.Kind != "" || s.Input.Provenance.Source != "" || s.Input.PTX != nil && (s.Input.PTX.FunctionOffsetUnknown || s.Input.PTX.ArtifactDigest != "" || s.Input.PTX.Geometry != nil || s.Input.PTX.Schema != nil) || s.ObservedOutcome != nil && s.ObservedOutcome.LatencyUnavailable
}
func successorJSON(fields map[string]json.RawMessage, parent string) bool {
	for name, raw := range fields {
		name = historyName(name)
		child := ""
		switch parent {
		case "snapshot":
			switch name {
			case "provenance":
				child = "provenance"
			case "input":
				child = "input"
			case "observed_outcome":
				child = "outcome"
			}
		case "input":
			switch name {
			case "provenance":
				child = "provenance"
			case "ptx":
				child = "ptx"
			}
		case "provenance":
			if name == "kind" || name == "source" {
				return true
			}
		case "ptx":
			switch name {
			case "function_offset_unknown", "artifact_digest", "geometry", "schema":
				return true
			}
		case "outcome":
			if name == "latency_unavailable" {
				return true
			}
		}
		if child != "" {
			var nested map[string]json.RawMessage
			if json.Unmarshal(raw, &nested) == nil && successorJSON(nested, child) {
				return true
			}
		}
	}
	return false
}

func historyName(name string) string {
	for _, key := range []string{"provenance", "input", "observed_outcome", "ptx", "kind", "source", "function_offset_unknown", "artifact_digest", "geometry", "schema", "latency_unavailable"} {
		if strings.EqualFold(name, key) {
			return key
		}
	}
	return ""
}
