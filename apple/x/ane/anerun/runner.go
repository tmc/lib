package anerun

import (
	"fmt"
	"os"
	"path/filepath"
)

// Runner manages compiled Apple Neural Engine models.
type Runner struct {
	ModelPath string
}

// NewRunner creates an ANE runner targeting a compiled .mlmodelc package.
func NewRunner(modelPath string) *Runner {
	return &Runner{
		ModelPath: modelPath,
	}
}

// PackageMLModel packages a raw MIL program string into a standard CoreML .mlpackage directory.
func PackageMLModel(milText string, outDir string) (string, error) {

	if outDir == "" {
		tmpDir, err := os.MkdirTemp("", "ane-mil-*")
		if err != nil {
			return "", fmt.Errorf("anerun: create temp dir: %w", err)
		}
		outDir = tmpDir
	}

	// 1. Create .mlpackage directory structure
	pkgDir := filepath.Join(outDir, "model.mlpackage")
	dataDir := filepath.Join(pkgDir, "Data", "com.apple.CoreML", "weights")
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return "", fmt.Errorf("anerun: mkdir mlpackage: %w", err)
	}

	milPath := filepath.Join(pkgDir, "Data", "com.apple.CoreML", "model.mil")
	if err := os.WriteFile(milPath, []byte(milText), 0644); err != nil {
		return "", fmt.Errorf("anerun: write model.mil: %w", err)
	}

	// Write package metadata
	manifestJSON := `{
    "fileFormatVersion": "1.0.0",
    "itemInfoEntries": {
        "com.apple.CoreML/model.mil": {
            "author": "tmc/lib",
            "description": "CoreML Model Specification",
            "name": "model.mil",
            "path": "com.apple.CoreML/model.mil"
        }
    },
    "rootModelIdentifier": "com.apple.CoreML/model.mil"
}`
	if err := os.WriteFile(filepath.Join(pkgDir, "Manifest.json"), []byte(manifestJSON), 0644); err != nil {
		return "", fmt.Errorf("anerun: write manifest: %w", err)
	}

	return pkgDir, nil
}
