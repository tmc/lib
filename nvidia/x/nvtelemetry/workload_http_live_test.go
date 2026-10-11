package nvtelemetry

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestWorkloadHTTPLive_AllSevenProviders(t *testing.T) {
	// Pinned providers, versions, fixture filenames, and expected SHA-256 hashes
	// from testdata/workload/provenance.json.
	tests := []struct {
		provider   string
		version    string
		fixture    string
		wantSHA256 string
		runID      string
	}{
		{
			provider:   "triton",
			version:    "25.06",
			fixture:    "triton-before.prom",
			wantSHA256: "cd65b6479f727628c8975b74fcf2153f8d686cc904277959f132b2eee92a976c",
			runID:      "test-run-triton",
		},
		{
			provider:   "vllm",
			version:    "0.8.5",
			fixture:    "vllm.prom",
			wantSHA256: "71a857f6ff8c4c82aff06bae7f56fc822c8f8c0904c04dfa3fb5805f7361434f",
			runID:      "test-run-vllm",
		},
		{
			provider:   "ray",
			version:    "2.44.0",
			fixture:    "ray.prom",
			wantSHA256: "9df8d969ff4b202f81e53defce1e5ee23fc43f2925db2d996aa1ec04a6c07c8a",
			runID:      "test-run-ray",
		},
		{
			provider:   "mlflow",
			version:    "2.22.0",
			fixture:    "mlflow.json",
			wantSHA256: "04ecf8816028139b7479a8a08b8ee4ebfa60be1b5acfe19bb15f85fa78453b2d",
			runID:      "test-run-mlflow",
		},
		{
			provider:   "wandb",
			version:    "0.19.11",
			fixture:    "wandb.jsonl",
			wantSHA256: "3c7db6657a28ecd2ae56212fe3dd9788990da98af4a712a35abe9b3c39c8d740",
			runID:      "test-run-wandb",
		},
		{
			provider:   "clearml",
			version:    "1.18.0/api-2.9",
			fixture:    "clearml.json",
			wantSHA256: "e60af7446f1cc1544acc51b6e1647417b59fb2357fe475be5158b68d80390f04",
			runID:      "train1", // Task ID inside clearml.json fixture.
		},
		{
			provider:   "pytorch",
			version:    "2.7.0",
			fixture:    "pytorch.json",
			wantSHA256: "47249bb039f7641b655d8aeb843ca1607c61e893979ffafdd94d7dd1cda3c054",
			runID:      "test-run-pytorch",
		},
	}

	const expectedAuthToken = "Bearer live-test-secret-token-xyz"

	for _, tc := range tests {
		tc := tc
		t.Run(tc.provider, func(t *testing.T) {
			fixtureBytes, err := os.ReadFile("testdata/workload/" + tc.fixture)
			if err != nil {
				t.Fatalf("read fixture %s: %v", tc.fixture, err)
			}

			// Verify fixture integrity on disk before serving.
			diskHash := sha256.Sum256(fixtureBytes)
			diskHashHex := hex.EncodeToString(diskHash[:])
			if diskHashHex != tc.wantSHA256 {
				t.Fatalf("fixture %s disk hash mismatch: got %s, want %s", tc.fixture, diskHashHex, tc.wantSHA256)
			}

			var serverCalls int64
			// Spin up a live loopback HTTP server requiring authenticating token.
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt64(&serverCalls, 1)
				if r.Header.Get("Authorization") != expectedAuthToken {
					http.Error(w, "unauthorized", http.StatusUnauthorized)
					return
				}
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(fixtureBytes)
			}))
			defer ts.Close()

			req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL+"/"+tc.fixture, nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			req.Header.Set("Authorization", expectedAuthToken)

			source := WorkloadSource{
				Provider: tc.provider,
				Version:  tc.version,
				RunID:    tc.runID,
			}

			result, err := FetchWorkload(context.Background(), ts.Client(), req, source, WorkloadLimits{})
			if err != nil {
				t.Fatalf("FetchWorkload over live network failed: %v", err)
			}

			if atomic.LoadInt64(&serverCalls) != 1 {
				t.Fatalf("expected exactly 1 server call over live network, got %d", atomic.LoadInt64(&serverCalls))
			}

			// Verify exact raw SHA-256 matches provenance hash.
			if result.Export.RawSHA256 != tc.wantSHA256 {
				t.Fatalf("export RawSHA256 mismatch: got %s, want %s", result.Export.RawSHA256, tc.wantSHA256)
			}

			// Verify exact raw bytes are preserved without modification.
			if !bytes.Equal(result.Export.Raw, fixtureBytes) {
				t.Fatalf("export Raw bytes do not match original fixture bytes")
			}

			// Verify provenance metadata.
			if result.Export.Source.Provider != tc.provider {
				t.Fatalf("export Provider mismatch: got %s, want %s", result.Export.Source.Provider, tc.provider)
			}
			if result.Export.Source.Version != tc.version {
				t.Fatalf("export Version mismatch: got %s, want %s", result.Export.Source.Version, tc.version)
			}
			if result.Export.Source.RunID != tc.runID {
				t.Fatalf("export RunID mismatch: got %s, want %s", result.Export.Source.RunID, tc.runID)
			}

			// Verify parsed output is non-empty.
			if tc.provider == "pytorch" {
				if len(result.Annotations) == 0 {
					t.Fatalf("expected non-empty Annotations for PyTorch profile")
				}
			} else {
				if len(result.Export.Metrics) == 0 {
					t.Fatalf("expected non-empty parsed Metrics for %s", tc.provider)
				}
			}
		})
	}
}

func TestWorkloadHTTPLive_AuthenticationEnforcement(t *testing.T) {
	// Verifies that 401 Unauthorized over a live TCP connection is properly refused
	// and does not leak credentials or URLs in error strings.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer valid-token" {
			http.Error(w, "unauthorized secret error message", http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"metrics":[]}`))
	}))
	defer ts.Close()

	source := WorkloadSource{
		Provider: "mlflow",
		Version:  "2.22.0",
		RunID:    "run-1",
	}

	// Request without authorization header.
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL+"?token=supersecret123", nil)
	_, err := FetchWorkload(context.Background(), ts.Client(), req, source, WorkloadLimits{})
	if err == nil {
		t.Fatal("expected error on 401 Unauthorized over live connection")
	}
	if strings.Contains(err.Error(), "supersecret123") || strings.Contains(err.Error(), "secret error message") {
		t.Fatalf("error leaked secret information: %v", err)
	}
}

func TestWorkloadHTTPLive_ContextCancellation(t *testing.T) {
	// Verifies that context cancellation halts live network transfer.
	started := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		// Delay to allow client cancellation.
		time.Sleep(200 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"metrics":[]}`))
	}))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL, nil)
	source := WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "run-1"}

	go func() {
		<-started
		cancel()
	}()

	_, err := FetchWorkload(ctx, ts.Client(), req, source, WorkloadLimits{})
	if err == nil {
		t.Fatal("expected error on cancelled context over live network")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestWorkloadHTTPLive_LimitsEnforcement(t *testing.T) {
	// Verifies that response size limits are enforced over a real live network connection.
	payload := strings.Repeat("x", 1024)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, payload)
	}))
	defer ts.Close()

	req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL, nil)
	source := WorkloadSource{Provider: "ray", Version: "2.44.0", RunID: "run-1"}

	// MaxBytes limit smaller than payload.
	_, err := FetchWorkload(context.Background(), ts.Client(), req, source, WorkloadLimits{MaxBytes: 100})
	if err == nil {
		t.Fatal("expected error on exceeded MaxBytes over live network")
	}
}

func TestWorkloadHTTPLive_ClearMLPagination(t *testing.T) {
	// Live authenticated loopback test for FetchClearMLScalarPages.
	calls := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("Authorization") != "Bearer clearml-secret" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if calls == 1 {
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [
						{"type":"training_stats_scalar","task":"live-task","metric":"loss","variant":"train","value":0.5,"iter":0,"timestamp":1000}
					],
					"scroll_id": "scroll-live-2",
					"returned": 1,
					"total": 2
				},
				"meta": {"result_code": 200}
			}`)
		} else if calls == 2 {
			_, _ = io.WriteString(w, `{
				"data": {
					"events": [
						{"type":"training_stats_scalar","task":"live-task","metric":"loss","variant":"train","value":0.25,"iter":1,"timestamp":1010}
					],
					"scroll_id": "",
					"returned": 1,
					"total": 2
				},
				"meta": {"result_code": 200}
			}`)
		}
	}))
	defer ts.Close()

	body := []byte(`{"task":"live-task","event_type":"training_stats_scalar","batch_size":1}`)
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, ts.URL+"/events.get_task_events", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer clearml-secret")
	source := WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "live-task"}

	pages, err := FetchClearMLScalarPages(context.Background(), ts.Client(), req, source, WorkloadLimits{}, 4)
	if err != nil {
		t.Fatalf("FetchClearMLScalarPages over live network: %v", err)
	}
	if len(pages) != 2 || calls != 2 {
		t.Fatalf("expected 2 pages and 2 network calls, got pages=%d calls=%d", len(pages), calls)
	}
	if pages[0].Export.RawSHA256 == pages[1].Export.RawSHA256 {
		t.Fatal("expected distinct raw hashes for distinct pages")
	}
}

func TestWorkloadHTTPLive_MLflowPagination(t *testing.T) {
	// Live authenticated loopback test for FetchMLflowHistoryPages.
	calls := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("Authorization") != "Bearer mlflow-secret" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		token := r.URL.Query().Get("page_token")
		if token == "" {
			_, _ = io.WriteString(w, `{
				"metrics": [
					{"key":"loss","value":0.8,"timestamp":1000,"step":0}
				],
				"next_page_token": "token-2"
			}`)
		} else if token == "token-2" {
			_, _ = io.WriteString(w, `{
				"metrics": [
					{"key":"loss","value":0.4,"timestamp":1010,"step":1}
				]
			}`)
		}
	}))
	defer ts.Close()

	req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL+"/api/2.0/mlflow/metrics/get-history?run_id=live-run&metric_key=loss", nil)
	req.Header.Set("Authorization", "Bearer mlflow-secret")
	source := WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "live-run"}

	pages, err := FetchMLflowHistoryPages(context.Background(), ts.Client(), req, source, WorkloadLimits{}, 4)
	if err != nil {
		t.Fatalf("FetchMLflowHistoryPages over live network: %v", err)
	}
	if len(pages) != 2 || calls != 2 {
		t.Fatalf("expected 2 pages and 2 network calls, got pages=%d calls=%d", len(pages), calls)
	}
	if pages[0].Export.RawSHA256 == pages[1].Export.RawSHA256 {
		t.Fatal("expected distinct raw hashes for distinct pages")
	}
}
