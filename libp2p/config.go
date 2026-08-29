package libp2p

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	golibp2p "github.com/libp2p/go-libp2p"
	ma "github.com/multiformats/go-multiaddr"
)

type Config struct {
	ListenAddrs     []string
	UserAgent       string
	ProtocolVersion string
	DisablePing     bool
}

func DefaultConfig() Config {
	return Config{
		ListenAddrs: []string{"/ip4/127.0.0.1/tcp/0"},
		DisablePing: true,
	}
}

func ConfigFromEnv() (Config, error) {
	cfg := DefaultConfig()
	if v := strings.TrimSpace(os.Getenv("LIBP2P_LISTEN_ADDRS")); v != "" {
		cfg.ListenAddrs = strings.Split(v, ",")
	}
	cfg.UserAgent = os.Getenv("LIBP2P_USER_AGENT")
	cfg.ProtocolVersion = os.Getenv("LIBP2P_PROTOCOL_VERSION")
	if v := os.Getenv("LIBP2P_DISABLE_PING"); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return Config{}, fmt.Errorf("libp2p: parse disable ping: %w", err)
		}
		cfg.DisablePing = b
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c Config) Validate() error {
	for _, addr := range c.ListenAddrs {
		addr = strings.TrimSpace(addr)
		if addr == "" {
			return fmt.Errorf("libp2p: empty listen address: %w", ErrInvalidArgument)
		}
		if _, err := ma.NewMultiaddr(addr); err != nil {
			return fmt.Errorf("libp2p: parse listen address %q: %w", addr, err)
		}
	}
	return nil
}

func (c Config) options() ([]golibp2p.Option, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	var opts []golibp2p.Option
	if len(c.ListenAddrs) > 0 {
		opts = append(opts, golibp2p.ListenAddrStrings(c.ListenAddrs...))
	}
	if c.UserAgent != "" {
		opts = append(opts, golibp2p.UserAgent(c.UserAgent))
	}
	if c.ProtocolVersion != "" {
		opts = append(opts, golibp2p.ProtocolVersion(c.ProtocolVersion))
	}
	if c.DisablePing {
		opts = append(opts, golibp2p.Ping(false))
	}
	return opts, nil
}
