package main

import (
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
)

//go:embed cudaprofstacks.py
var pythonStackAdapter []byte

func prepareStackTransport(dir string) error {
	if err := os.Mkdir(filepath.Join(dir, "stacks"), 0700); err != nil {
		return fmt.Errorf("create stack transport: %w", err)
	}
	python := filepath.Join(dir, "python")
	if err := os.Mkdir(python, 0700); err != nil {
		return fmt.Errorf("create Python adapter directory: %w", err)
	}
	if err := os.WriteFile(filepath.Join(python, "cudaprofstacks.py"), pythonStackAdapter, 0600); err != nil {
		return fmt.Errorf("write Python stack adapter: %w", err)
	}
	return nil
}
