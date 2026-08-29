package libp2p

import "testing"

func TestConfigValidate(t *testing.T) {
	cfg := DefaultConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
	cfg.ListenAddrs = []string{""}
	if err := cfg.Validate(); err == nil {
		t.Fatal("empty listen address was accepted")
	}
	cfg.ListenAddrs = []string{"not-a-multiaddr"}
	if err := cfg.Validate(); err == nil {
		t.Fatal("invalid listen address was accepted")
	}
}

func TestConfigFromEnv(t *testing.T) {
	t.Setenv("LIBP2P_LISTEN_ADDRS", "/ip4/127.0.0.1/tcp/0")
	t.Setenv("LIBP2P_USER_AGENT", "tmc-lib-test")
	t.Setenv("LIBP2P_DISABLE_PING", "false")
	cfg, err := ConfigFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.UserAgent != "tmc-lib-test" || cfg.DisablePing {
		t.Fatalf("ConfigFromEnv = %#v", cfg)
	}
}
