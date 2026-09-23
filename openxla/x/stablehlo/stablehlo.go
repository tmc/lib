package stablehlo

import (
	"fmt"
	"io"
)

// Emitter defines the interface for emitting MLIR text.
type Emitter interface {
	Emit(w io.Writer) error
}

// Module represents a builtin.module in MLIR.
type Module struct {
	Funcs []Emitter
}

// Emit writes the module to the given io.Writer.
func (m *Module) Emit(w io.Writer) error {
	if _, err := io.WriteString(w, "module {\n"); err != nil {
		return fmt.Errorf("emit module start: %w", err)
	}
	for _, f := range m.Funcs {
		if err := f.Emit(w); err != nil {
			return err
		}
	}
	if _, err := io.WriteString(w, "}\n"); err != nil {
		return fmt.Errorf("emit module end: %w", err)
	}
	return nil
}
