//go:build linux

package main

import (
	"errors"
	"fmt"
	"io"
)

func selectBackend(name string) (string, error) {
	switch name {
	case "auto", "cuda":
		return "cuda", nil
	case "cpu":
		return "cpu", nil
	case "metal":
		return "", errors.New("metal backend unsupported on linux")
	default:
		return "", fmt.Errorf("unknown backend %q", name)
	}
}

func openSessionBackend(cfg config) (sessionBackend, io.Closer, error) {
	if cfg.backend != "cuda" {
		return nil, nil, fmt.Errorf("%s backend execution not implemented", cfg.backend)
	}
	r, err := newCUDAFoundation(cfg.model)
	if err != nil {
		return nil, nil, err
	}
	defer r.Close()
	return nil, nil, fmt.Errorf("cuda device %q sm_%d%d qualified; token evaluation not implemented", r.name, r.major, r.minor)
}
