package cudaprof

import (
	"encoding/json"
	"fmt"
	"os"
)

const maxExternalStackSidecarBytes = 16 << 20

type externalStackSidecarLine struct {
	Version      int      `json:"version"`
	Kind         string   `json:"kind"`
	Losses       string   `json:"losses,omitempty"`
	ExternalKind uint32   `json:"external_kind,omitempty"`
	ExternalID   uint64   `json:"external_id,omitempty"`
	Frames       []string `json:"frames,omitempty"`
}

// appendExternalStackRecord appends one retained record. The fixed-width loss
// count in the first line is updated in place; stack records remain append-only.
func appendExternalStackRecord(record ExternalStackRecord, losses uint64) error {
	return updateExternalStackSidecar(&record, losses)
}

func appendExternalStackLoss(losses uint64) error {
	return updateExternalStackSidecar(nil, losses)
}

func updateExternalStackSidecar(record *ExternalStackRecord, losses uint64) error {
	path := os.Getenv(ExternalStacksEnv)
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return fmt.Errorf("cudaprof: open external stack sidecar: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return fmt.Errorf("cudaprof: stat external stack sidecar: %w", err)
	}
	if info.Size() == 0 {
		if _, err := f.WriteAt(externalStackSidecarHeader(losses), 0); err != nil {
			return fmt.Errorf("cudaprof: write external stack sidecar header: %w", err)
		}
		info, err = f.Stat()
		if err != nil {
			return fmt.Errorf("cudaprof: stat external stack sidecar: %w", err)
		}
	} else if err := writeExternalStackSidecarLosses(f, losses); err != nil {
		return err
	}
	if record == nil {
		return f.Sync()
	}
	line := externalStackSidecarLine{
		Version: 1, Kind: "stack", ExternalKind: record.ExternalKind,
		ExternalID: record.ExternalID, Frames: record.Frames,
	}
	data, err := json.Marshal(line)
	if err != nil {
		return fmt.Errorf("cudaprof: encode external stack sidecar: %w", err)
	}
	data = append(data, '\n')
	if info.Size()+int64(len(data)) > maxExternalStackSidecarBytes {
		return fmt.Errorf("cudaprof: external stack sidecar exceeds %d bytes", maxExternalStackSidecarBytes)
	}
	if _, err := f.WriteAt(data, info.Size()); err != nil {
		return fmt.Errorf("cudaprof: append external stack sidecar: %w", err)
	}
	if err := writeExternalStackSidecarLosses(f, losses); err != nil {
		return err
	}
	return f.Sync()
}

func externalStackSidecarHeader(losses uint64) []byte {
	return []byte(fmt.Sprintf("{\"version\":1,\"kind\":\"header\",\"losses\":\"%020d\"}\n", losses))
}

func writeExternalStackSidecarLosses(f *os.File, losses uint64) error {
	header := externalStackSidecarHeader(losses)
	if _, err := f.WriteAt(header, 0); err != nil {
		return fmt.Errorf("cudaprof: update external stack sidecar losses: %w", err)
	}
	return nil
}
