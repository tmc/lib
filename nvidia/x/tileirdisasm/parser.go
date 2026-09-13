package tileirdisasm

import (
	"io"

	"github.com/tmc/lib/nvidia/x/tileir"
)

// Parse parses textual Tile IR from an io.Reader into an AST Module.
func Parse(r io.Reader) (*Module, error) {
	return tileir.Parse(r)
}

// ParseString parses textual Tile IR from a string.
func ParseString(s string) (*Module, error) {
	return tileir.ParseString(s)
}

// ParseBytes parses textual Tile IR from a byte slice.
func ParseBytes(b []byte) (*Module, error) {
	return tileir.ParseBytes(b)
}
