package cudaprof

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const maxExternalStackSidecarBytes = 16 << 20
const externalStackLossCheckpointInterval = 64

type externalStackSidecarLine struct {
	Version      int      `json:"version"`
	Kind         string   `json:"kind"`
	Losses       string   `json:"losses,omitempty"`
	LowerBound   bool     `json:"losses_lower_bound"`
	Process      uint64   `json:"process,omitempty"`
	ExternalKind uint32   `json:"external_kind,omitempty"`
	ExternalID   uint64   `json:"external_id,omitempty"`
	Frames       []string `json:"frames,omitempty"`
}

// appendExternalStackRecord writes one complete JSONL record to this process's
// sidecar. The controller gives each capture a shared base path; the PID
// suffix keeps child processes from mixing IDs or racing on the header.
func appendExternalStackRecord(record ExternalStackRecord, losses uint64) error {
	path, err := externalStackSidecarPath()
	if err != nil || path == "" {
		return err
	}
	f, info, err := openExternalStackSidecar(path)
	if err != nil {
		return err
	}
	defer f.Close()
	process := uint64(os.Getpid())
	var data []byte
	if info.Size() == 0 {
		header := externalStackSidecarLine{Version: 1, Kind: "header", Process: process, Losses: fmt.Sprintf("%020d", losses), LowerBound: true}
		line, err := marshalExternalStackSidecarLine(header)
		if err != nil {
			return err
		}
		data = append(data, line...)
	}
	line := externalStackSidecarLine{Version: 1, Kind: "stack", Process: process, ExternalKind: record.ExternalKind, ExternalID: record.ExternalID, Frames: record.Frames}
	encoded, err := marshalExternalStackSidecarLine(line)
	if err != nil {
		return err
	}
	data = append(data, encoded...)
	if info.Size()+int64(len(data)) > maxExternalStackSidecarBytes {
		return fmt.Errorf("cudaprof: external stack sidecar exceeds %d bytes", maxExternalStackSidecarBytes)
	}
	if _, err := f.Seek(0, io.SeekEnd); err != nil {
		return fmt.Errorf("cudaprof: seek external stack sidecar: %w", err)
	}
	if _, err := f.Write(data); err != nil {
		return fmt.Errorf("cudaprof: append external stack sidecar: %w", err)
	}
	return nil
}

// appendExternalStackLoss checkpoints losses once when the budget first fills
// and then every fixed interval. The header is therefore a lower bound between
// checkpoints, and repeated rejected Do calls do no file I/O.
func appendExternalStackLoss(losses uint64) error {
	if losses == 0 || (losses != 1 && (losses-1)%externalStackLossCheckpointInterval != 0) {
		return nil
	}
	path, err := externalStackSidecarPath()
	if err != nil || path == "" {
		return err
	}
	f, info, err := openExternalStackSidecar(path)
	if err != nil {
		return err
	}
	defer f.Close()
	header := externalStackSidecarLine{Version: 1, Kind: "header", Process: uint64(os.Getpid()), Losses: fmt.Sprintf("%020d", losses), LowerBound: true}
	data, err := marshalExternalStackSidecarLine(header)
	if err != nil {
		return err
	}
	if info.Size() == 0 {
		if _, err := f.Write(data); err != nil {
			return fmt.Errorf("cudaprof: write external stack sidecar header: %w", err)
		}
	} else if _, err := f.WriteAt(data, 0); err != nil {
		return fmt.Errorf("cudaprof: update external stack sidecar loss checkpoint: %w", err)
	}
	return nil
}

func externalStackSidecarPath() (string, error) {
	base := os.Getenv(ExternalStacksEnv)
	if base == "" {
		return "", nil
	}
	return fmt.Sprintf("%s.%d", base, os.Getpid()), nil
}

func openExternalStackSidecar(path string) (*os.File, os.FileInfo, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return nil, nil, fmt.Errorf("cudaprof: create external stack sidecar directory: %w", err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, nil, fmt.Errorf("cudaprof: open external stack sidecar: %w", err)
	}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("cudaprof: stat external stack sidecar: %w", err)
	}
	return f, info, nil
}

func marshalExternalStackSidecarLine(line externalStackSidecarLine) ([]byte, error) {
	data, err := json.Marshal(line)
	if err != nil {
		return nil, fmt.Errorf("cudaprof: encode external stack sidecar: %w", err)
	}
	return append(data, '\n'), nil
}
