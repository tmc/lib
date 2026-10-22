package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Publish only a complete, closed output, without replacing an existing file.
func writeCollectorOutput(path string, write func(io.Writer) error) (err error) {
	f, err := os.CreateTemp(filepath.Dir(path), ".ds4-collector-*")
	if err != nil {
		return err
	}
	name := f.Name()
	defer func() { err = errors.Join(err, os.Remove(name)) }()
	writeErr := write(f)
	if writeErr == nil {
		writeErr = f.Sync()
	}
	if err := errors.Join(writeErr, f.Close()); err != nil {
		return fmt.Errorf("write collector output: %w", err)
	}
	if err := os.Link(name, path); err != nil {
		return fmt.Errorf("publish collector output: %w", err)
	}
	return nil
}
