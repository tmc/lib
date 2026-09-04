//go:build linux

package main

import (
	"errors"
	"fmt"
	"io"
)

func selectBackend(name string) (string, error) {
	switch name {
	case "cuda":
		return "cuda", nil
	case "auto":
		return "", errors.New("automatic CUDA detection not implemented; select cuda or cpu explicitly")
	case "cpu":
		return "cpu", nil
	case "metal":
		return "", errors.New("metal backend unsupported on linux")
	default:
		return "", fmt.Errorf("unknown backend %q", name)
	}
}

func openSessionBackend(cfg config) (sessionBackend, io.Closer, error) {
	return nil, nil, fmt.Errorf("%s backend execution not implemented", cfg.backend)
}
