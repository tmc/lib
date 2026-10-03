package main

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// admitCapturePlanCode checks a producer's retained module against the actual
// artifact receipt using the existing capture verifier. It is not a driver
// attestation. Virtual artifacts need a separate JIT image association, which
// this path does not infer. Returned bytes are the exact original code packet.
func admitCapturePlanCode(dir string, limit int, pub cudaprof.CapturePublication) ([]byte, error) {
	if dir == "" {
		return nil, fmt.Errorf("capture plan code: transport unavailable: %w", os.ErrNotExist)
	}
	if limit <= 0 || limit > 64<<20 {
		return nil, fmt.Errorf("capture plan code: invalid byte budget")
	}
	switch pub.Result.Provider {
	case cudaprof.CaptureActivity, cudaprof.CaptureRange, cudaprof.CapturePM, cudaprof.CapturePC, cudaprof.CaptureSASS:
	default:
		return nil, fmt.Errorf("capture plan code: unknown provider")
	}
	raw, err := readCapturePlanFile(dir, "capture-provider-"+string(pub.Result.Provider)+".pprof", limit)
	if err != nil {
		return nil, fmt.Errorf("capture plan code: %w", err)
	}
	// Bound the outer profile's decompression before its protobuf parser runs.
	// Embedded capture files receive their own independent limits below.
	data := raw
	if len(raw) >= 2 && raw[0] == 0x1f && raw[1] == 0x8b {
		reader, err := gzip.NewReader(bytes.NewReader(raw))
		if err != nil {
			return raw, fmt.Errorf("capture plan code: %w", err)
		}
		decoded, readErr := io.ReadAll(io.LimitReader(reader, int64(limit)+1))
		closeErr := reader.Close()
		if readErr != nil {
			return raw, fmt.Errorf("capture plan code: %w", readErr)
		}
		if closeErr != nil {
			return raw, fmt.Errorf("capture plan code: %w", closeErr)
		}
		if len(decoded) > limit {
			return raw, fmt.Errorf("capture plan code: decompressed byte budget exceeded")
		}
		data = decoded
	}
	return admitCapturePlanCodeBytes(raw, data, limit, pub)
}

// readCapturePlanFile checks the same opened file that will be read, so replacing
// a checked regular file with a symlink cannot redirect publication admission.
func readCapturePlanFile(dir, name string, limit int) ([]byte, error) {
	if limit <= 0 || limit > 64<<20 || !filepath.IsLocal(name) {
		return nil, fmt.Errorf("capture plan file: invalid name or byte budget")
	}
	path := filepath.Join(dir, name)
	info, err := os.Lstat(path)
	if err != nil {
		return nil, fmt.Errorf("capture plan code: %w", err)
	}
	if !info.Mode().IsRegular() || info.Size() > int64(limit) {
		return nil, fmt.Errorf("capture plan code: not a bounded regular file")
	}
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, fmt.Errorf("capture plan code: %w", err)
	}
	defer root.Close()
	f, err := root.Open(name)
	if err != nil {
		return nil, fmt.Errorf("capture plan code: %w", err)
	}
	opened, statErr := f.Stat()
	if statErr != nil || !os.SameFile(info, opened) {
		f.Close()
		return nil, fmt.Errorf("capture plan code: file changed before read")
	}
	raw, readErr := io.ReadAll(io.LimitReader(f, int64(limit)+1))
	closeErr := f.Close()
	if readErr != nil {
		return nil, fmt.Errorf("capture plan code: %w", readErr)
	}
	if closeErr != nil {
		return nil, fmt.Errorf("capture plan code: %w", closeErr)
	}
	if len(raw) > limit {
		return nil, fmt.Errorf("capture plan code: input byte budget exceeded")
	}
	return raw, nil
}

func admitCapturePlanCodeBytes(raw, data []byte, limit int, pub cudaprof.CapturePublication) ([]byte, error) {
	capture, err := cudacapture.ReadLimits(bytes.NewReader(data), cudacapture.Limits{Files: 64, FileBytes: int64(limit), TotalBytes: int64(limit), InputBytes: int64(limit)})
	if err != nil {
		return raw, fmt.Errorf("capture plan code: %w", err)
	}
	if pub.Artifact == nil || pub.Artifact.Kind != cudakernel.KindCUBIN {
		return raw, fmt.Errorf("capture plan code: native image association unavailable: %w", os.ErrNotExist)
	}
	if err := pub.Artifact.Validate(); err != nil {
		return raw, fmt.Errorf("capture plan code: %w", err)
	}
	id, err := cudaprof.BindEvidence(pub.Artifact, pub.Workload, pub.CompilerKey)
	if err != nil || id != pub.Result.Identity {
		return raw, fmt.Errorf("capture plan code: artifact identity mismatch")
	}
	infoCUBIN, err := cudakernel.ParseCUBIN(pub.Artifact.Raw)
	if err != nil {
		return raw, fmt.Errorf("capture plan code: %w", err)
	}
	if infoCUBIN.Arch == "" {
		return raw, fmt.Errorf("capture plan code: native target unavailable: %w", os.ErrNotExist)
	}
	if infoCUBIN.Arch != strings.TrimRight(pub.Artifact.Target, "af") {
		return raw, fmt.Errorf("capture plan code: native target mismatch")
	}
	if infoCUBIN.Arch != pub.Artifact.Target {
		return raw, fmt.Errorf("capture plan code: target feature suffix unverified: %w", os.ErrNotExist)
	}
	var module *cudacapture.Module
	var imageRetained bool
	for i := range capture.Modules {
		candidate := &capture.Modules[i]
		image, ok := capture.Files[candidate.Code.CUBIN]
		if candidate.Code.CUBIN == "" || !ok {
			continue
		}
		imageRetained = true
		if !bytes.Equal(image, pub.Artifact.Raw) {
			continue
		}
		if module != nil {
			return raw, fmt.Errorf("capture plan code: ambiguous image generation")
		}
		module = candidate
	}
	if module == nil {
		if !imageRetained {
			return raw, fmt.Errorf("capture plan code: native image unavailable: %w", os.ErrNotExist)
		}
		return raw, fmt.Errorf("capture plan code: artifact image not retained")
	}
	for _, role := range []string{module.Code.Source, module.Code.PTX, module.Code.CUBIN, module.Code.SASS} {
		if role == "" {
			return raw, fmt.Errorf("capture plan code: code role unavailable: %w", os.ErrNotExist)
		}
		if _, ok := capture.Files[role]; !ok {
			return raw, fmt.Errorf("capture plan code: code artifact unavailable: %w", os.ErrNotExist)
		}
	}
	if _, err := cudacapture.VerifyModule(capture, module.ID, pub.Artifact.Digest, []string{pub.Result.Identity.Entry}); err != nil {
		return raw, fmt.Errorf("capture plan code: %w", err)
	}
	return raw, nil
}
