package hfsync_test

import (
	"context"
	"fmt"
	"net/netip"
	"os"
	"path/filepath"
	"time"

	"github.com/tmc/go-iroh/iroh"
	"github.com/tmc/go-iroh/key"
	"github.com/tmc/go-iroh/netaddr"
	"github.com/tmc/go-iroh/relay"
	"github.com/tmc/lib/hfsync"
)

func Example() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// 1. Setup local temporary Hugging Face store with a model snapshot
	tmpDir, _ := os.MkdirTemp("", "hf-example-*")
	defer os.RemoveAll(tmpDir)

	snapshotDir := filepath.Join(tmpDir, "models--example--tiny-llama", "snapshots", "rev123")
	_ = os.MkdirAll(snapshotDir, 0755)
	_ = os.WriteFile(filepath.Join(snapshotDir, "config.json"), []byte(`{"model": "tiny"}`), 0644)

	store := hfsync.NewStore(tmpDir)

	// 2. Start an Iroh Server
	serverSK, _ := key.GenerateSecretKey()
	serverEP, _ := iroh.Bind(ctx,
		iroh.WithSecretKey(serverSK),
		iroh.WithALPNs(hfsync.ALPN),
		iroh.WithBindAddr(netip.MustParseAddrPort("127.0.0.1:0")),
		iroh.WithRelayMode(relay.ModeDisabled()),
	)
	defer serverEP.Shutdown(ctx)

	server := hfsync.NewServer(serverEP, store)
	go func() { _ = server.Serve(ctx) }()

	// 3. Start a Client and fetch model manifest
	clientSK, _ := key.GenerateSecretKey()
	clientEP, _ := iroh.Bind(ctx,
		iroh.WithSecretKey(clientSK),
		iroh.WithALPNs(hfsync.ALPN),
		iroh.WithBindAddr(netip.MustParseAddrPort("127.0.0.1:0")),
		iroh.WithRelayMode(relay.ModeDisabled()),
	)
	defer clientEP.Shutdown(ctx)

	client := hfsync.NewClient(clientEP, store)
	serverAddr := netaddr.NewEndpointAddr(serverEP.ID()).WithIP(serverEP.LocalAddr())
	manifest, err := client.FetchManifest(ctx, serverAddr, "example/tiny-llama")
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	fmt.Printf("Model: %s, Files: %d\n", manifest.ModelID, len(manifest.Files))
	// Output:
	// Model: example/tiny-llama, Files: 1
}
