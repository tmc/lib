package gopccl

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
	// Coordinator is the PCCL master host:port.
	Coordinator string
	// PeerGroup selects the PCCL peer group.
	PeerGroup uint32
	// P2PPoolSize controls PCCL peer connection pooling.
	P2PPoolSize uint32
	// StartMaster starts a local PCCL master in this process.
	StartMaster bool
}

func ConfigFromEnv() (Config, error) {
	var cfg Config
	cfg.Backend = os.Getenv("GOPCCL_BACKEND")
	if v := os.Getenv("GOPCCL_RANK"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("gopccl: parse rank: %w", err)
		}
		cfg.Rank = n
	}
	if v := os.Getenv("GOPCCL_SIZE"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("gopccl: parse size: %w", err)
		}
		cfg.Size = n
	}
	cfg.NodeID = os.Getenv("GOPCCL_NODE_ID")
	cfg.Device = os.Getenv("GOPCCL_DEVICE")
	cfg.Rendezvous = os.Getenv("GOPCCL_RENDEZVOUS")
	if v := os.Getenv("GOPCCL_RETRY_LIMIT"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("gopccl: parse retrylimit: %w", err)
		}
		cfg.RetryLimit = n
	}
	cfg.Coordinator = os.Getenv("GOPCCL_COORDINATOR")
	if v := os.Getenv("GOPCCL_PEER_GROUP"); v != "" {
		n, err := strconv.ParseUint(v, 10, 32)
		if err != nil {
			return Config{}, fmt.Errorf("gopccl: parse peergroup: %w", err)
		}
		cfg.PeerGroup = uint32(n)
	}
	if v := os.Getenv("GOPCCL_P2P_POOL_SIZE"); v != "" {
		n, err := strconv.ParseUint(v, 10, 32)
		if err != nil {
			return Config{}, fmt.Errorf("gopccl: parse p2ppoolsize: %w", err)
		}
		cfg.P2PPoolSize = uint32(n)
	}
	if v := os.Getenv("GOPCCL_START_MASTER"); v != "" {
		n, err := strconv.ParseBool(v)
		if err != nil {
			return Config{}, fmt.Errorf("gopccl: parse startmaster: %w", err)
		}
		cfg.StartMaster = n
	}
	return cfg, nil
}
