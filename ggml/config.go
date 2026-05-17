package ggml

import (
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	// Backend selects the registered ggml backend adapter.
	Backend string
	// Rank is reserved for distributed model-runtime wrappers.
	Rank int
	// Size is reserved for distributed model-runtime wrappers.
	Size int
	// Threads configures CPU graph execution when supported.
	Threads int
}

func ConfigFromEnv() (Config, error) {
	var cfg Config
	cfg.Backend = os.Getenv("GGML_BACKEND")
	if v := os.Getenv("GGML_RANK"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("ggml: parse rank: %w", err)
		}
		cfg.Rank = n
	}
	if v := os.Getenv("GGML_SIZE"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("ggml: parse size: %w", err)
		}
		cfg.Size = n
	}
	if v := os.Getenv("GGML_THREADS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("ggml: parse threads: %w", err)
		}
		cfg.Threads = n
	}
	return cfg, nil
}
