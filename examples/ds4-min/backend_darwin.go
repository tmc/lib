//go:build darwin && arm64

package main

import (
	"errors"
	"fmt"
	"io"
	"os"
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
	if err := validatePower(defaultPower(cfg.power), cfg.family); err != nil {
		return nil, nil, err
	}
	if cfg.family == "v41" {
		return openV41SessionBackend(cfg)
	}
	if cfg.backend != "metal" {
		return nil, nil, errors.New("cpu backend is limited to test fixtures")
	}
	r, err := newMetalRuntimeStreamingLogged(cfg.model, cfg.context, cfg.log, cfg.streaming)
	if err != nil {
		if r != nil {
			err = finishBackendCleanup(err, r)
		}
		return nil, nil, err
	}
	r.power = newPowerControl(defaultPower(cfg.power), len(r.layers))
	s, closer, err := newStateSessionBackend(r.stateBackend(), cfg.context)
	if err != nil {
		return nil, nil, finishBackendCleanup(fmt.Errorf("create Metal request state: %w", err), r)
	}
	return s, closer, nil
}

func exitProcess(code int) {
	os.Exit(code)
}
