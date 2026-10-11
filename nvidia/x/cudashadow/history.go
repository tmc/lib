package cudashadow

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

const historyVersion = 1
const maxHistoryBytes = 16 << 20
const maxHistorySnapshots = 4096

type historyWire struct {
	Version   int        `json:"version"`
	Snapshots []Snapshot `json:"snapshots"`
}

// WriteHistory writes a version 1 offline replay corpus. It validates every
// snapshot and refuses more than 4096 snapshots or 16 MiB of encoded data.
// A stateful policy requires a complete history from its initial state;
// a wrapped evaluator's retained suffix is not necessarily replayable.
func WriteHistory(w io.Writer, snapshots []Snapshot) error {
	if w == nil {
		return errors.New("cudashadow: nil history writer")
	}
	if len(snapshots) > maxHistorySnapshots {
		return errors.New("cudashadow: history capacity exceeded")
	}
	for i, snapshot := range snapshots {
		if err := snapshot.Validate(); err != nil {
			return fmt.Errorf("cudashadow history snapshot %d: %w", i, err)
		}
	}
	data, err := json.Marshal(historyWire{Version: historyVersion, Snapshots: snapshots})
	if err != nil {
		return err
	}
	if len(data) > maxHistoryBytes {
		return errors.New("cudashadow: history exceeds 16 MiB")
	}
	n, err := w.Write(data)
	if err != nil {
		return err
	}
	if n != len(data) {
		return io.ErrShortWrite
	}
	return nil
}

// ReadHistory reads a bounded version 1 replay corpus, rejecting unknown
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
	if wire.Version != historyVersion {
		return nil, errors.New("cudashadow: unsupported history version")
	}
	if len(wire.Snapshots) > maxHistorySnapshots {
		return nil, errors.New("cudashadow: history capacity exceeded")
	}
	for i, snapshot := range wire.Snapshots {
		if err := snapshot.Validate(); err != nil {
			return nil, fmt.Errorf("cudashadow history snapshot %d: %w", i, err)
		}
	}
	return wire.Snapshots, nil
}
