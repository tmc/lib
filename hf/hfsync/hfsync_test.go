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

func TestStoreListModelsUsesMainRefAndTargetFileSize(t *testing.T) {
	root := t.TempDir()
	modelID := "example/tiny-llama"
	modelDir := filepath.Join(root, ModelDirName(modelID))
	blobPath := filepath.Join(modelDir, "blobs", "weights")
	if err := os.MkdirAll(filepath.Dir(blobPath), 0755); err != nil {
		t.Fatal(err)
	}
	weights := bytes.Repeat([]byte("x"), 4096)
	if err := os.WriteFile(blobPath, weights, 0644); err != nil {
		t.Fatal(err)
	}
	for _, revision := range []string{"older", "main-revision"} {
		if err := os.MkdirAll(filepath.Join(modelDir, "snapshots", revision), 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(filepath.Join("..", "..", "blobs", "weights"), filepath.Join(modelDir, "snapshots", "main-revision", "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(modelDir, "refs"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "refs", "main"), []byte("main-revision\n"), 0644); err != nil {
		t.Fatal(err)
	}

	models, err := NewStore(root).ListModels()
	if err != nil {
		t.Fatalf("list models: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("listed %d models, want 1", len(models))
	}
	if got, want := models[0].Revision, "main-revision"; got != want {
		t.Errorf("revision = %q, want %q", got, want)
	}
	if got, want := models[0].TotalBytes, int64(len(weights)); got != want {
		t.Errorf("total bytes = %d, want %d", got, want)
	}
}

func TestStoreWriteChunkPreallocates(t *testing.T) {
	store := NewStore(t.TempDir())
	if err := store.WriteChunk("example/model", "revision", "weights.bin", 10, 5, []byte("world")); err != nil {
		t.Fatalf("write chunk: %v", err)
	}
	path := filepath.Join(store.ModelPath("example/model"), "snapshots", "revision", "weights.bin")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat preallocated file: %v", err)
	}
	if got, want := info.Size(), int64(10); got != want {
		t.Errorf("preallocated size = %d, want %d", got, want)
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
	var progressCalls int
	stats, err := client.SyncWithProgress(ctx, serverAddr, modelID, func(string, int64, int64, float64) {
		progressCalls++
	})
	if err != nil {
		t.Fatalf("sync failed: %v", err)
	}

	if stats.FilesSynced != 3 {
		t.Errorf("expected 3 files synced, got %d", stats.FilesSynced)
	}
	if stats.BytesDownloaded <= 5*1024*1024 {
		t.Errorf("expected >5MB downloaded, got %d bytes", stats.BytesDownloaded)
	}
	if progressCalls < 5 { // config, tokenizer, and three 2 MB weight chunks
		t.Errorf("progress calls = %d, want at least 5", progressCalls)
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

	// 7. Corrupt a cached file. The next sync must reject the stale digest and
	// download it again instead of treating its presence as a cache hit.
	if err := os.WriteFile(filepath.Join(clientDir, ModelDirName(modelID), "snapshots", snapshotID, "model.safetensors"), []byte("corrupt"), 0644); err != nil {
		t.Fatalf("corrupt cached weights: %v", err)
	}
	repairedStats, err := client.Sync(ctx, serverAddr, modelID)
	if err != nil {
		t.Fatalf("repair sync failed: %v", err)
	}
	if repairedStats.BytesDownloaded < 5*1024*1024 {
		t.Errorf("repair downloaded %d bytes, want weight file", repairedStats.BytesDownloaded)
	}

	// 8. Test ListModels API and its prefix filter.
	models, err := client.ListModels(ctx, serverAddr)
	if err != nil {
		t.Fatalf("list models: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("expected 1 model listed, got %d", len(models))
	}
	if models[0].ModelID != modelID {
		t.Errorf("model ID mismatch: got %q, want %q", models[0].ModelID, modelID)
	}
	if models[0].FileCount != 3 {
		t.Errorf("file count mismatch: got %d, want 3", models[0].FileCount)
	}
	models, err = client.ListModelsWithFilter(ctx, serverAddr, "other/")
	if err != nil {
		t.Fatalf("list filtered models: %v", err)
	}
	if len(models) != 0 {
		t.Errorf("filtered model count = %d, want 0", len(models))
	}
}
