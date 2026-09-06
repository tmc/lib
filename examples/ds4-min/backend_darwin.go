//go:build darwin && arm64

package main

import (
	"errors"
	"fmt"
	"io"
)

func selectBackend(name string) (string, error) {
	switch name {
	case "auto", "metal":
		return "metal", nil
	case "cpu":
		return "cpu", nil
	case "cuda":
		return "", errors.New("cuda backend unsupported on darwin/arm64")
	default:
		return "", fmt.Errorf("unknown backend %q", name)
	}
}

func openSessionBackend(cfg config) (sessionBackend, io.Closer, error) {
	if cfg.backend != "metal" {
		return nil, nil, errors.New("cpu backend is limited to test fixtures")
	}
	r, err := newMetalRuntimeLogged(cfg.model, cfg.context, cfg.log)
	if err != nil {
		return nil, nil, err
	}
	return r, r, nil
}
