package cudaprof

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
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

type externalStackSidecarFile interface {
	Write([]byte) (int, error)
	Stat() (os.FileInfo, error)
	Close() error
	Sync() error
}

var openExternalStackSidecarFile = func(path string) (externalStackSidecarFile, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return nil, fmt.Errorf("cudaprof: create external stack sidecar directory: %w", err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0600)
	if err != nil {
		return nil, fmt.Errorf("cudaprof: open external stack sidecar: %w", err)
	}
	return f, nil
}

var externalStackSidecarState struct {
	sync.Mutex
	path string
	file externalStackSidecarFile
	size int64
}

func closeExternalStackSidecar() {
	externalStackSidecarState.Lock()
	defer externalStackSidecarState.Unlock()
	if externalStackSidecarState.file != nil {
		_ = externalStackSidecarState.file.Close()
	}
	externalStackSidecarState.path, externalStackSidecarState.file, externalStackSidecarState.size = "", nil, 0
}

func sidecarFile(path string) (externalStackSidecarFile, int64, error) {
	if externalStackSidecarState.path == path && externalStackSidecarState.file != nil {
		return externalStackSidecarState.file, externalStackSidecarState.size, nil
	}
	if externalStackSidecarState.file != nil {
		_ = externalStackSidecarState.file.Close()
	}
	f, err := openExternalStackSidecarFile(path)
	if err != nil {
		externalStackSidecarState.file = nil
		return nil, 0, err
	}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		externalStackSidecarState.file = nil
		return nil, 0, fmt.Errorf("cudaprof: stat external stack sidecar: %w", err)
	}
	externalStackSidecarState.path, externalStackSidecarState.file, externalStackSidecarState.size = path, f, info.Size()
	return f, info.Size(), nil
}

func writeExternalStackSidecar(data []byte) error {
	externalStackSidecarState.Lock()
	defer externalStackSidecarState.Unlock()
	path, err := externalStackSidecarPath()
	if err != nil || path == "" {
		return err
	}
	f, size, err := sidecarFile(path)
	if err != nil {
		return err
	}
	if size+int64(len(data)) > maxExternalStackSidecarBytes {
		return fmt.Errorf("cudaprof: external stack sidecar exceeds %d bytes", maxExternalStackSidecarBytes)
	}
	n, err := f.Write(data)
	if err != nil {
		return fmt.Errorf("cudaprof: append external stack sidecar: %w", err)
	}
	if n != len(data) {
		return io.ErrShortWrite
	}
	externalStackSidecarState.size += int64(n)
	return nil
}

func appendExternalStackRecord(record ExternalStackRecord, losses uint64) error {
	path, err := externalStackSidecarPath()
	if err != nil || path == "" {
		return err
	}
	var data []byte
	externalStackSidecarState.Lock()
	defer externalStackSidecarState.Unlock()
	f, size, err := sidecarFile(path)
	if err != nil {
		return err
	}
	if size == 0 {
		header, err := marshalExternalStackSidecarLine(externalStackSidecarLine{Version: 1, Kind: "header", Process: uint64(os.Getpid()), Losses: fmt.Sprintf("%020d", losses), LowerBound: true})
		if err != nil {
			return err
		}
		data = append(data, header...)
	}
	line, err := marshalExternalStackSidecarLine(externalStackSidecarLine{Version: 1, Kind: "stack", Process: uint64(os.Getpid()), ExternalKind: record.ExternalKind, ExternalID: record.ExternalID, Frames: record.Frames})
	if err != nil {
		return err
	}
	data = append(data, line...)
	if size+int64(len(data)) > maxExternalStackSidecarBytes {
		return fmt.Errorf("cudaprof: external stack sidecar exceeds %d bytes", maxExternalStackSidecarBytes)
	}
	n, err := f.Write(data)
	if err != nil {
		return fmt.Errorf("cudaprof: append external stack sidecar: %w", err)
	}
	if n != len(data) {
		return io.ErrShortWrite
	}
	externalStackSidecarState.size += int64(n)
	return nil
}

func appendExternalStackLoss(losses uint64) error {
	if losses == 0 || (losses != 1 && (losses-1)%externalStackLossCheckpointInterval != 0) {
		return nil
	}
	data, err := marshalExternalStackSidecarLine(externalStackSidecarLine{Version: 1, Kind: "losses", Process: uint64(os.Getpid()), Losses: fmt.Sprintf("%020d", losses), LowerBound: true})
	if err != nil {
		return err
	}
	return writeExternalStackSidecar(data)
}

func externalStackSidecarPath() (string, error) {
	base := os.Getenv(ExternalStacksEnv)
	if base == "" {
		return "", nil
	}
	return fmt.Sprintf("%s.%d", base, os.Getpid()), nil
}

func marshalExternalStackSidecarLine(line externalStackSidecarLine) ([]byte, error) {
	data, err := json.Marshal(line)
	if err != nil {
		return nil, fmt.Errorf("cudaprof: encode external stack sidecar: %w", err)
	}
	return append(data, '\n'), nil
}
