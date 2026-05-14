package gorccl

import (
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	// Backend selects fake, auto, or a real provider.
	Backend string
	// Rank is the local rank.
	Rank int
	// Size is the number of ranks.
	Size int
	// NodeID identifies the local node.
	NodeID string
	// Device selects local accelerator resources.
	Device string
	// Rendezvous names the context-aware store endpoint.
	Rendezvous string
	// Store provides optional in-process rendezvous.
	Store Store
	// RetryLimit bounds retry loops.
	RetryLimit int
}

func ConfigFromEnv() (Config, error) {
	var cfg Config
	cfg.Backend = os.Getenv("GORCCL_BACKEND")
	if v := os.Getenv("GORCCL_RANK"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("gorccl: parse rank: %w", err)
		}
		cfg.Rank = n
	}
	if v := os.Getenv("GORCCL_SIZE"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("gorccl: parse size: %w", err)
		}
		cfg.Size = n
	}
	cfg.NodeID = os.Getenv("GORCCL_NODE_ID")
	cfg.Device = os.Getenv("GORCCL_DEVICE")
	cfg.Rendezvous = os.Getenv("GORCCL_RENDEZVOUS")
	if v := os.Getenv("GORCCL_RETRY_LIMIT"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("gorccl: parse retrylimit: %w", err)
		}
		cfg.RetryLimit = n
	}
	return cfg, nil
}
