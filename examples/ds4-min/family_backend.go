package main

import (
	"encoding/hex"
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/gguf"
)

func validateFamilySelection(cfg config) error {
	switch cfg.family {
	case "", "auto", "flash":
	case "v41":
		if err := cfg.streaming.validate("cpu"); err != nil {
			return fmt.Errorf("V4.1 streaming: %w", err)
		}
		if cfg.backend != "cpu" {
			return fmt.Errorf("V4.1 requires explicit -backend cpu; %s execution not implemented", cfg.backend)
		}
		if cfg.context <= 0 || cfg.context > 1048576 {
			return errors.New("V4.1 context must be in [1,1048576]")
		}
	default:
		return fmt.Errorf("unknown model family %q", cfg.family)
	}
	if cfg.modelSHA256 != "" {
		data, err := hex.DecodeString(cfg.modelSHA256)
		if err != nil || len(data) != 32 {
			return errors.New("model-sha256 must contain 64 hexadecimal digits")
		}
		if cfg.family == "flash" {
			return errors.New("model-sha256 is only supported for V4.1")
		}
	}
	return nil
}

func validateFamilyProfile(cfg *config, g *gguf.File) error {
	family := "flash"
	if g != nil && g.Metadata["general.architecture"].Value == "deepseek41" {
		family = "v41"
	}
	if cfg.family != "" && cfg.family != "auto" && cfg.family != family {
		return fmt.Errorf("model family %s does not match -family %s", family, cfg.family)
	}
	cfg.family = family
	if err := validateFamilySelection(*cfg); err != nil {
		return err
	}
	if family == "v41" {
		_, err := dsv41.ValidateMetadata(g)
		return err
	}
	return validatePinnedProfile(g)
}

func openV41SessionBackend(cfg config) (sessionBackend, io.Closer, error) {
	if err := validateFamilySelection(cfg); err != nil {
		return nil, nil, err
	}
	var options v41CPUOptions
	if cfg.modelSHA256 != "" {
		data, err := hex.DecodeString(cfg.modelSHA256)
		if err != nil || len(data) != 32 {
			return nil, nil, errors.New("invalid V4.1 asset SHA-256")
		}
		copy(options.AssetSHA256[:], data)
	}
	b, err := newV41CPUBackend(cfg.model, options)
	if err != nil {
		if b != nil {
			return nil, nil, &backendCleanupOwner{cause: err, backend: b}
		}
		return nil, nil, err
	}
	s, closer, err := newStateSessionBackend(b, cfg.context)
	if err != nil {
		closeErr := b.Close()
		if closeErr != nil {
			return nil, nil, &backendCleanupOwner{cause: errors.Join(err, closeErr), backend: b}
		}
		return nil, nil, err
	}
	return s, closer, nil
}
