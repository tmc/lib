package cudacache

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
)

// HTTPRemoteSource implements RemoteSource using HTTP GET/PUT requests against a remote registry.
type HTTPRemoteSource struct {
	client           *http.Client
	baseURL          string
	maxDownloadBytes int64
}

// NewHTTPRemoteSource creates an HTTPRemoteSource pointing at baseURL.
func NewHTTPRemoteSource(baseURL string, client *http.Client) *HTTPRemoteSource {
	if client == nil {
		client = http.DefaultClient
	}
	return &HTTPRemoteSource{
		client:           client,
		baseURL:          baseURL,
		maxDownloadBytes: DefaultMaxRemoteDownloadBytes,
	}
}

// FetchArtifact downloads the artifact for key from the remote server.
func (h *HTTPRemoteSource) FetchArtifact(ctx context.Context, key string) (io.ReadCloser, int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/artifacts/%s", h.baseURL, key), nil)
	if err != nil {
		return nil, 0, err
	}

	resp, err := h.client.Do(req)
	if err != nil {
		return nil, 0, err
	}

	if resp.StatusCode == http.StatusNotFound {
		resp.Body.Close()
		return nil, 0, ErrRemoteNotFound
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, 0, fmt.Errorf("remote returned HTTP %d", resp.StatusCode)
	}

	return resp.Body, resp.ContentLength, nil
}

// FetchReceipt retrieves the search receipt for key from the remote server.
func (h *HTTPRemoteSource) FetchReceipt(ctx context.Context, key string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/receipts/%s", h.baseURL, key), nil)
	if err != nil {
		return nil, err
	}

	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, ErrRemoteNotFound
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("remote returned HTTP %d", resp.StatusCode)
	}

	if h.maxDownloadBytes > 0 && resp.ContentLength > h.maxDownloadBytes {
		return nil, ErrDownloadLimitExceeded
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, h.maxDownloadBytes+1))
	if err != nil {
		return nil, err
	}
	if h.maxDownloadBytes > 0 && int64(len(data)) > h.maxDownloadBytes {
		return nil, ErrDownloadLimitExceeded
	}
	return data, nil
}

// PublishArtifact uploads an artifact payload to the remote server.
func (h *HTTPRemoteSource) PublishArtifact(ctx context.Context, key string, data []byte) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, fmt.Sprintf("%s/artifacts/%s", h.baseURL, key), bytes.NewReader(data))
	if err != nil {
		return err
	}

	resp, err := h.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("remote put returned HTTP %d", resp.StatusCode)
	}
	return nil
}

// PublishReceipt uploads a search receipt to the remote server.
func (h *HTTPRemoteSource) PublishReceipt(ctx context.Context, key string, data []byte) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, fmt.Sprintf("%s/receipts/%s", h.baseURL, key), bytes.NewReader(data))
	if err != nil {
		return err
	}

	resp, err := h.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("remote put returned HTTP %d", resp.StatusCode)
	}
	return nil
}

// MemoryRegistry is an in-memory RemoteSource implementation useful for testing and mocking.
type MemoryRegistry struct {
	mu        sync.RWMutex
	artifacts map[string][]byte
	receipts  map[string][]byte
}

// NewMemoryRegistry creates an empty MemoryRegistry.
func NewMemoryRegistry() *MemoryRegistry {
	return &MemoryRegistry{
		artifacts: make(map[string][]byte),
		receipts:  make(map[string][]byte),
	}
}

// FetchArtifact fetches artifact bytes from memory.
func (m *MemoryRegistry) FetchArtifact(ctx context.Context, key string) (io.ReadCloser, int64, error) {
	m.mu.RLock()
	data, ok := m.artifacts[key]
	m.mu.RUnlock()
	if !ok {
		return nil, 0, ErrRemoteNotFound
	}
	return io.NopCloser(bytes.NewReader(data)), int64(len(data)), nil
}

// FetchReceipt fetches receipt bytes from memory.
func (m *MemoryRegistry) FetchReceipt(ctx context.Context, key string) ([]byte, error) {
	m.mu.RLock()
	data, ok := m.receipts[key]
	m.mu.RUnlock()
	if !ok {
		return nil, ErrRemoteNotFound
	}
	return append([]byte(nil), data...), nil
}

// PublishArtifact stores artifact bytes in memory.
func (m *MemoryRegistry) PublishArtifact(ctx context.Context, key string, data []byte) error {
	m.mu.Lock()
	m.artifacts[key] = append([]byte(nil), data...)
	m.mu.Unlock()
	return nil
}

// PublishReceipt stores receipt bytes in memory.
func (m *MemoryRegistry) PublishReceipt(ctx context.Context, key string, data []byte) error {
	m.mu.Lock()
	m.receipts[key] = append([]byte(nil), data...)
	m.mu.Unlock()
	return nil
}

// NewTestServer creates an httptest.Server serving artifacts and receipts from reg.
func NewTestServer(reg *MemoryRegistry) *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/artifacts/", func(w http.ResponseWriter, r *http.Request) {
		key := r.URL.Path[len("/artifacts/"):]
		switch r.Method {
		case http.MethodGet:
			reg.mu.RLock()
			data, ok := reg.artifacts[key]
			reg.mu.RUnlock()
			if !ok {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.Write(data)
		case http.MethodPut:
			data, err := io.ReadAll(r.Body)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			reg.mu.Lock()
			reg.artifacts[key] = data
			reg.mu.Unlock()
			w.WriteHeader(http.StatusOK)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/receipts/", func(w http.ResponseWriter, r *http.Request) {
		key := r.URL.Path[len("/receipts/"):]
		switch r.Method {
		case http.MethodGet:
			reg.mu.RLock()
			data, ok := reg.receipts[key]
			reg.mu.RUnlock()
			if !ok {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.Write(data)
		case http.MethodPut:
			data, err := io.ReadAll(r.Body)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			reg.mu.Lock()
			reg.receipts[key] = data
			reg.mu.Unlock()
			w.WriteHeader(http.StatusOK)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	return httptest.NewServer(mux)
}

// FetchProof retrieves canonical proof bytes from /proofs/{kind}/{key}.
// The cache bounds and verifies the reader; no content-type or format guessing
// is performed. HTTPS peer authentication is separate from producer identity.
func (h *HTTPRemoteSource) FetchProof(ctx context.Context, kind, key string) (io.ReadCloser, error) {
	if (kind != RemoteArtifact && kind != RemoteReceipt) || validateKey(key) != nil {
		return nil, ErrUntrustedSource
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/proofs/%s/%s", h.baseURL, kind, url.PathEscape(key)), nil)
	if err != nil {
		return nil, err
	}
	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK || resp.ContentLength > MaxRemoteProofBytes {
		resp.Body.Close()
		return nil, ErrUntrustedSource
	}
	return resp.Body, nil
}
