//go:build linux

package main

import (
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/ebitengine/purego"
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
	if err := validatePower(defaultPower(cfg.power), cfg.family); err != nil {
		return nil, nil, err
	}
	if err := cfg.streaming.validate(cfg.backend); err != nil {
		return nil, nil, err
	}
	if cfg.family == "v41" {
		if cfg.streaming.enabled {
			return nil, nil, errors.New("ssd-streaming V41 execution unsupported")
		}
		return openV41SessionBackend(cfg)
	}
	if cfg.backend != "cuda" {
		return nil, nil, fmt.Errorf("%s backend execution not implemented", cfg.backend)
	}
	r, err := newCUDARuntimeWithOptions(cfg.model, cfg.context, cudaRuntimeOptions{
		streaming:      cfg.streaming,
		ballastGib:     cfg.ballastGib,
		ballastSet:     cfg.ballastGibSet,
		warmWeights:    cfg.warmWeights,
		warmWeightsSet: cfg.warmWeightsSet,
	})
	if err != nil {
		return nil, nil, err
	}
	r.power = newPowerControl(defaultPower(cfg.power), len(r.model.layers))
	s, closer, err := newStateSessionBackend(r.stateBackend(), cfg.context)
	if err != nil {
		_ = r.Close()
		return nil, nil, fmt.Errorf("create CUDA request state: %w", err)
	}
	return s, closer, nil
}

var libcExit func(int)

func init() {
	libc, err := purego.Dlopen("libc.so.6", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err == nil {
		purego.RegisterLibFunc(&libcExit, libc, "exit")
	}
}

func exitProcess(code int) {
	if libcExit != nil {
		libcExit(code)
	}
	os.Exit(code)
}
