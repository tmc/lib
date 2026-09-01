package hfsync

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/tmc/go-iroh/iroh"
	"github.com/tmc/go-iroh/netaddr"
)

// SyncStats summarizes the network and disk metrics of a model synchronization.
type SyncStats struct {
	ModelID         string        `json:"model_id"`
	Revision        string        `json:"revision"`
	FilesSynced     int           `json:"files_synced"`
	FilesSkipped    int           `json:"files_skipped"`
	BytesDownloaded int64         `json:"bytes_downloaded"`
	Duration        time.Duration `json:"duration"`
	ThroughputMBs   float64       `json:"throughput_mbs"`
}

// Client manages P2P model synchronization against remote Iroh endpoints.
type Client struct {
	ep        *iroh.Endpoint
	store     *Store
	chunkSize int64
}

// NewClient initializes a new hfsync Client.
func NewClient(ep *iroh.Endpoint, store *Store) *Client {
	return &Client{
		ep:        ep,
		store:     store,
		chunkSize: DefaultChunkSize,
	}
}

// ProgressFunc is invoked periodically to report download progress.
type ProgressFunc func(file string, downloadedBytes, totalBytes int64, currentMBs float64)

// ListModels queries the list of available model snapshots hosted on a remote peer.
func (c *Client) ListModels(ctx context.Context, peerAddr netaddr.EndpointAddr) ([]ModelSummary, error) {
	return c.ListModelsWithFilter(ctx, peerAddr, "")
}

// ListModelsWithFilter queries snapshots whose model ID has filter as a prefix.
func (c *Client) ListModelsWithFilter(ctx context.Context, peerAddr netaddr.EndpointAddr, filter string) ([]ModelSummary, error) {
	conn, err := c.ep.Connect(ctx, peerAddr, ALPN)
	if err != nil {
		return nil, fmt.Errorf("connect to peer %s: %w", peerAddr.ID, err)
	}
	defer conn.Close()

	stream, err := conn.OpenStreamSync(ctx)
	if err != nil {
		return nil, fmt.Errorf("open list models stream: %w", err)
	}
	defer stream.Close()

	if err := writeJSONFrame(stream, msgTypeListModelsReq, ListModelsRequest{Filter: filter}); err != nil {
		return nil, fmt.Errorf("send list models request: %w", err)
	}

	resp, err := readJSONFrame[ListModelsResponse](stream, msgTypeListModelsResp)
	if err != nil {
		return nil, fmt.Errorf("receive list models response: %w", err)
	}
	return resp.Models, nil
}

// FetchManifest retrieves the model's file layout and hashes from a peer.
func (c *Client) FetchManifest(ctx context.Context, peerAddr netaddr.EndpointAddr, modelID string) (*Manifest, error) {
	conn, err := c.ep.Connect(ctx, peerAddr, ALPN)
	if err != nil {
		return nil, fmt.Errorf("connect to peer %s: %w", peerAddr.ID, err)
	}
	defer conn.Close()

	stream, err := conn.OpenStreamSync(ctx)
	if err != nil {
		return nil, fmt.Errorf("open manifest stream: %w", err)
	}
	defer stream.Close()

	if err := writeJSONFrame(stream, msgTypeManifestReq, ManifestRequest{ModelID: modelID}); err != nil {
		return nil, fmt.Errorf("send manifest request: %w", err)
	}

	manifest, err := readJSONFrame[Manifest](stream, msgTypeManifestResp)
	if err != nil {
		return nil, fmt.Errorf("receive manifest response: %w", err)
	}
	return manifest, nil
}

// Sync synchronizes a model repository snapshot from a peer into the local Hugging Face store.
func (c *Client) Sync(ctx context.Context, peerAddr netaddr.EndpointAddr, modelID string) (*SyncStats, error) {
	return c.SyncWithProgress(ctx, peerAddr, modelID, nil)
}

// SyncWithProgress synchronizes a model snapshot while invoking onProgress callback.
func (c *Client) SyncWithProgress(ctx context.Context, peerAddr netaddr.EndpointAddr, modelID string, onProgress ProgressFunc) (*SyncStats, error) {
	start := time.Now()

	conn, err := c.ep.Connect(ctx, peerAddr, ALPN)
	if err != nil {
		return nil, fmt.Errorf("connect to peer: %w", err)
	}
	defer conn.Close()

	// 1. Fetch Remote Manifest
	stream, err := conn.OpenStreamSync(ctx)
	if err != nil {
		return nil, fmt.Errorf("open manifest stream: %w", err)
	}
	if err := writeJSONFrame(stream, msgTypeManifestReq, ManifestRequest{ModelID: modelID}); err != nil {
		stream.Close()
		return nil, fmt.Errorf("send manifest request: %w", err)
	}
	manifest, err := readJSONFrame[Manifest](stream, msgTypeManifestResp)
	stream.Close()
	if err != nil {
		return nil, fmt.Errorf("receive manifest: %w", err)
	}

	if len(manifest.Files) == 0 {
		return nil, fmt.Errorf("model %q has no files or not found on peer", modelID)
	}

	stats := &SyncStats{
		ModelID:  manifest.ModelID,
		Revision: manifest.Revision,
	}

	// 2. Download missing/outdated files
	for _, fi := range manifest.Files {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		valid, err := c.store.VerifyFile(manifest.ModelID, manifest.Revision, fi.Path, fi.SHA256)
		if err == nil && valid {
			stats.FilesSkipped++
			continue
		}

		if fi.Size < 0 {
			return nil, fmt.Errorf("invalid size for %s: %d", fi.Path, fi.Size)
		}
		if fi.Size == 0 {
			if err := c.store.WriteChunk(manifest.ModelID, manifest.Revision, fi.Path, 0, 0, nil); err != nil {
				return nil, fmt.Errorf("create empty file %s: %w", fi.Path, err)
			}
		}

		// Pull chunks for this file.
		var offset int64 = 0
		for offset < fi.Size {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}

			chunkLen := c.chunkSize
			if offset+chunkLen > fi.Size {
				chunkLen = fi.Size - offset
			}

			chunkData, err := c.fetchChunk(ctx, conn, manifest.ModelID, fi.Path, offset, chunkLen)
			if err != nil {
				return nil, fmt.Errorf("fetch chunk for %s (offset %d): %w", fi.Path, offset, err)
			}

			if err := c.store.WriteChunk(manifest.ModelID, manifest.Revision, fi.Path, fi.Size, offset, chunkData); err != nil {
				return nil, fmt.Errorf("write chunk for %s (offset %d): %w", fi.Path, offset, err)
			}

			stats.BytesDownloaded += int64(len(chunkData))
			offset += int64(len(chunkData))

			if onProgress != nil {
				elapsed := time.Since(start).Seconds()
				curSpeed := 0.0
				if elapsed > 0 {
					curSpeed = (float64(stats.BytesDownloaded) / (1024 * 1024)) / elapsed
				}
				onProgress(fi.Path, offset, fi.Size, curSpeed)
			}
		}

		// Verify downloaded file integrity
		valid, err = c.store.VerifyFile(manifest.ModelID, manifest.Revision, fi.Path, fi.SHA256)
		if err != nil || !valid {
			return nil, fmt.Errorf("file verification failed for %s (expected %s)", fi.Path, fi.SHA256)
		}

		stats.FilesSynced++
	}

	stats.Duration = time.Since(start)
	if stats.Duration > 0 {
		stats.ThroughputMBs = (float64(stats.BytesDownloaded) / (1024 * 1024)) / stats.Duration.Seconds()
	}
	return stats, nil
}

func (c *Client) fetchChunk(ctx context.Context, conn *iroh.Conn, modelID, relPath string, offset, length int64) ([]byte, error) {
	stream, err := conn.OpenStreamSync(ctx)
	if err != nil {
		return nil, fmt.Errorf("open chunk stream: %w", err)
	}
	defer stream.Close()

	req := ChunkRequest{
		ModelID: modelID,
		Path:    relPath,
		Offset:  offset,
		Length:  length,
	}

	if err := writeJSONFrame(stream, msgTypeChunkReq, req); err != nil {
		return nil, fmt.Errorf("send chunk request: %w", err)
	}

	msgType, headerPayload, err := readFrame(stream)
	if err != nil {
		return nil, fmt.Errorf("read chunk header frame: %w", err)
	}
	if msgType != msgTypeChunkResp {
		return nil, fmt.Errorf("unexpected frame type: got 0x%02x, want 0x%02x", msgType, msgTypeChunkResp)
	}

	var header ChunkHeader
	if err := json.Unmarshal(headerPayload, &header); err != nil {
		return nil, fmt.Errorf("unmarshal chunk header: %w", err)
	}
	if header.Status != "ok" {
		return nil, fmt.Errorf("chunk error from server: %s", header.Status)
	}
	if header.Offset != offset {
		return nil, fmt.Errorf("unexpected chunk offset: got %d, want %d", header.Offset, offset)
	}
	if header.Length <= 0 || header.Length > length {
		return nil, fmt.Errorf("invalid chunk length %d for request length %d", header.Length, length)
	}

	data := make([]byte, header.Length)
	if _, err := io.ReadFull(stream, data); err != nil {
		return nil, fmt.Errorf("read binary chunk payload: %w", err)
	}
	return data, nil
}
