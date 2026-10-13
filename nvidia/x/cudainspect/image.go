package cudainspect

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudaobjdump"
)

// ImageReport binds a selected ELF report to its input container and the
// commands used to extract it. Commands retain stdout and stderr, including
// failures. A nonnil result accompanying an error is incomplete evidence.
type ImageReport struct {
	ContainerDigest string               `json:"container_digest"`
	ContainerBytes  int                  `json:"container_bytes"`
	ImageName       string               `json:"image_name"`
	Commands        []cudaobjdump.Output `json:"commands"`
	Report          *ReportSchemaV1      `json:"report,omitempty"`
}

// InspectImage extracts ELF images with tool and inspects exactly imageName.
// It snapshots container under WorkDir so every command reads the same input
// bytes. The selected executable must be trusted. No image is loaded or run.
// The caller must not mutate container until this method returns. Temporary
// paths in command receipts are removed afterward; digests identify the bytes.
func (i *Inspector) InspectImage(ctx context.Context, tool cudaobjdump.Tool, container []byte, imageName, symbol string) (result *ImageReport, err error) {
	if i == nil || i.WorkDir == "" || len(container) == 0 || imageName == "" || symbol == "" {
		return nil, fmt.Errorf("cudarecipe: work directory, container, image name and symbol are required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	digest := sha256.Sum256(container)
	result = &ImageReport{ContainerDigest: hex.EncodeToString(digest[:]), ContainerBytes: len(container), ImageName: imageName}
	dir, err := os.MkdirTemp(i.WorkDir, "cudarecipe-")
	if err != nil {
		return result, err
	}
	defer func() { err = errors.Join(err, os.RemoveAll(dir)) }()
	path := filepath.Join(dir, "input.fatbin")
	if err := os.WriteFile(path, container, 0400); err != nil {
		return result, err
	}
	// Keep the extraction scratch beside the immutable input, not in a default
	// system temp directory whose lifetime the inspector does not control.
	tool.TempDir = dir
	version, err := tool.Version(ctx)
	result.Commands = append(result.Commands, version)
	if err != nil {
		return result, fmt.Errorf("cudarecipe: tool version: %w", err)
	}
	image, receipts, err := tool.ExtractImage(ctx, path, cudaobjdump.ELF, imageName)
	result.Commands = append(result.Commands, receipts...)
	if err != nil {
		return result, fmt.Errorf("cudarecipe: extract images: %w", err)
	}
	report, err := i.Inspect(image.Data, symbol)
	if err != nil {
		return result, err
	}
	result.Report = report
	if !report.ELFParsed || report.ELFMachine != "EM_CUDA" {
		return result, fmt.Errorf("cudarecipe: selected image is not a parsed CUDA ELF")
	}
	return result, nil
}
