package hfsync

import (
	"bytes"
	"context"
	"crypto/rand"
	"net/netip"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/go-iroh/iroh"
	"github.com/tmc/go-iroh/key"
	"github.com/tmc/go-iroh/netaddr"
	"github.com/tmc/go-iroh/relay"
)

func TestModelDirName(t *testing.T) {
	tests := []struct {
		modelID string
		want    string
	}{
		{"gpt2", "models--gpt2"},
		{"meta-llama/Llama-3.2-1B", "models--meta-llama--Llama-3.2-1B"},
		{"unsloth/Llama-3.2-1B-Instruct", "models--unsloth--Llama-3.2-1B-Instruct"},
	}

	for _, tt := range tests {
		got := ModelDirName(tt.modelID)
		if got != tt.want {
			t.Errorf("ModelDirName(%q) = %q, want %q", tt.modelID, got, tt.want)
		}
	}
}

func setupMockHFRepo(t *testing.T, rootDir, modelID, snapshotID string) {
	t.Helper()

	snapshotDir := filepath.Join(rootDir, ModelDirName(modelID), "snapshots", snapshotID)
	if err := os.MkdirAll(snapshotDir, 0755); err != nil {
		t.Fatalf("mkdir snapshot: %v", err)
	}

	files := map[string][]byte{
		"config.json":    []byte(`{"architectures": ["LlamaForCausalLM"], "hidden_size": 2048}`),
		"tokenizer.json": []byte(`{"version": "1.0", "model": {"type": "BPE"}}`),
	}

	// 5 MB random weights payload to test multi-chunk streaming
	weights := make([]byte, 5*1024*1024)
	if _, err := rand.Read(weights); err != nil {
		t.Fatalf("generate weights: %v", err)
	}
	files["model.safetensors"] = weights

	for relPath, data := range files {
		fullPath := filepath.Join(snapshotDir, relPath)
		if err := os.WriteFile(fullPath, data, 0644); err != nil {
			t.Fatalf("write mock file %s: %v", relPath, err)
		}
	}
}

func TestP2PHuggingFaceSync(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// 1. Prepare server storage
	serverDir := t.TempDir()
	modelID := "unsloth/Llama-3.2-1B-Instruct"
	snapshotID := "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
	setupMockHFRepo(t, serverDir, modelID, snapshotID)

	serverStore := NewStore(serverDir)

	// 2. Start server Iroh endpoint
	serverSK, err := key.GenerateSecretKey()
	if err != nil {
		t.Fatalf("server sk: %v", err)
	}
	serverEP, err := iroh.Bind(ctx,
		iroh.WithSecretKey(serverSK),
		iroh.WithALPNs(ALPN),
		iroh.WithBindAddr(netip.MustParseAddrPort("127.0.0.1:0")),
		iroh.WithRelayMode(relay.ModeDisabled()),
	)
	if err != nil {
		t.Fatalf("server bind: %v", err)
	}
	defer serverEP.Shutdown(ctx)

	server := NewServer(serverEP, serverStore)
	go func() {
		_ = server.Serve(ctx)
	}()

	// 3. Prepare client storage and endpoint
	clientDir := t.TempDir()
	clientStore := NewStore(clientDir)

	clientSK, err := key.GenerateSecretKey()
	if err != nil {
		t.Fatalf("client sk: %v", err)
	}
	clientEP, err := iroh.Bind(ctx,
		iroh.WithSecretKey(clientSK),
		iroh.WithALPNs(ALPN),
		iroh.WithBindAddr(netip.MustParseAddrPort("127.0.0.1:0")),
		iroh.WithRelayMode(relay.ModeDisabled()),
	)
	if err != nil {
		t.Fatalf("client bind: %v", err)
	}
	defer clientEP.Shutdown(ctx)

	client := NewClient(clientEP, clientStore)

	// 4. Perform Initial Synchronization
	serverAddr := netaddr.NewEndpointAddr(serverEP.ID()).WithIP(serverEP.LocalAddr())
	stats, err := client.Sync(ctx, serverAddr, modelID)
	if err != nil {
		t.Fatalf("sync failed: %v", err)
	}

	if stats.FilesSynced != 3 {
		t.Errorf("expected 3 files synced, got %d", stats.FilesSynced)
	}
	if stats.BytesDownloaded <= 5*1024*1024 {
		t.Errorf("expected >5MB downloaded, got %d bytes", stats.BytesDownloaded)
	}

	t.Logf("Sync completed successfully: %d files, %.2f MB in %v (%.2f MB/s)",
		stats.FilesSynced, float64(stats.BytesDownloaded)/(1024*1024), stats.Duration, stats.ThroughputMBs)

	// 5. Verify local files exist and match exactly
	serverWeights, _ := os.ReadFile(filepath.Join(serverDir, ModelDirName(modelID), "snapshots", snapshotID, "model.safetensors"))
	clientWeights, err := os.ReadFile(filepath.Join(clientDir, ModelDirName(modelID), "snapshots", snapshotID, "model.safetensors"))
	if err != nil {
		t.Fatalf("read client weights: %v", err)
	}
	if !bytes.Equal(serverWeights, clientWeights) {
		t.Fatal("synced weights do not match server weights byte-for-byte")
	}

	// 6. Test Idempotent Sync (All files should be skipped with 0 bytes downloaded)
	cachedStats, err := client.Sync(ctx, serverAddr, modelID)
	if err != nil {
		t.Fatalf("cached sync failed: %v", err)
	}

	if cachedStats.FilesSkipped != 3 {
		t.Errorf("expected 3 files skipped on second sync, got %d", cachedStats.FilesSkipped)
	}
	if cachedStats.BytesDownloaded != 0 {
		t.Errorf("expected 0 bytes downloaded on cached sync, got %d", cachedStats.BytesDownloaded)
	}
}
