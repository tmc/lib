//go:build linux

package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func (f *cudaFoundation) nativeStateIdentity() ([32]byte, [32]byte, error) {
	if f == nil || f.stateIdentity == nil || f.file == nil || f.gguf == nil {
		return [32]byte{}, [32]byte{}, errors.New("cuda state: model identity unavailable")
	}
	info, err := f.file.Stat()
	if err != nil {
		return [32]byte{}, [32]byte{}, err
	}
	return f.stateIdentity.get(f.file, info.Size(), f.gguf)
}

func validateCUDAStateIdentity(data []byte, model, quant [32]byte) error {
	if len(data) < cudaStateHeaderBytes || model == ([32]byte{}) || quant == ([32]byte{}) {
		return errors.New("cuda state load: missing model identity")
	}
	if string(data[40:72]) != string(model[:]) || string(data[72:104]) != string(quant[:]) {
		return errors.New("cuda state load: model or quantization identity mismatch")
	}
	return nil
}

// A failed rollback is terminal: a caller must never reuse partially restored
// device state. Preserve both errors for diagnostics.
func cudaStateTransaction(apply, rollback func() error, poison func(error)) error {
	if err := apply(); err != nil {
		if rerr := rollback(); rerr != nil {
			joined := errors.Join(err, fmt.Errorf("rollback: %w", rerr), dsv4.ErrDesynchronized)
			poison(joined)
			return joined
		}
		return err
	}
	return nil
}
