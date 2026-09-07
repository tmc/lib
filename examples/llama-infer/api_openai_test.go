//go:build !tpu
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"
)

func TestOpenAIEndpoints(t *testing.T) {
	cfg := Config{
		Name:      "Llama-3.2-1B-Test",
		Dim:       64,
		NumLayers: 2,
	}

	eng := &ModelEngine{}
	srv := NewTelemetryServer(eng, cfg)

	// 1. Test GET /v1/models
	reqModels := httptest.NewRequest("GET", "/v1/models", nil)
	wModels := httptest.NewRecorder()
	srv.ServeHTTP(wModels, reqModels)

	if wModels.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for /v1/models, got %d", wModels.Code)
	}

	var mResp ModelsResponse
	if err := json.Unmarshal(wModels.Body.Bytes(), &mResp); err != nil {
		t.Fatalf("failed decoding /v1/models response: %v", err)
	}
	if len(mResp.Data) == 0 || mResp.Data[0].ID != "Llama-3.2-1B-Test" {
		t.Errorf("unexpected models response: %+v", mResp)
	}

	// 2. Test POST /v1/chat/completions (Non-streaming)
	chatReq := ChatCompletionRequest{
		Model: "Llama-3.2-1B-Test",
		User:  "tenant-a",
		Messages: []ChatMessage{
			{Role: "user", Content: "Hello AI"},
		},
		MaxTokens: 10,
		Stream:    false,
	}
	reqBody, _ := json.Marshal(chatReq)
	reqChat := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(reqBody))
	reqChat.Header.Set("Content-Type", "application/json")
	wChat := httptest.NewRecorder()
	srv.ServeHTTP(wChat, reqChat)

	if wChat.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for chat completions, got %d: %s", wChat.Code, wChat.Body.String())
	}

	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(wChat.Body.Bytes(), &chatResp); err != nil {
		t.Fatalf("failed decoding chat response: %v", err)
	}
	if len(chatResp.Choices) == 0 {
		t.Fatal("expected at least 1 choice in response")
	}

	// 3. Test POST /v1/chat/completions (SSE Streaming)
	chatReqStream := ChatCompletionRequest{
		Model:     "Llama-3.2-1B-Test",
		Prompt:    "Streaming test",
		MaxTokens: 5,
		Stream:    true,
	}
	reqStreamBody, _ := json.Marshal(chatReqStream)
	reqStream := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(reqStreamBody))
	reqStream.Header.Set("Content-Type", "application/json")
	wStream := httptest.NewRecorder()
	srv.ServeHTTP(wStream, reqStream)

	if wStream.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for streaming completions, got %d", wStream.Code)
	}
	if !stringsContains(wStream.Body.String(), "chat.completion.chunk") {
		t.Errorf("expected SSE chunk in response, got: %s", wStream.Body.String())
	}
	if !stringsContains(wStream.Body.String(), "[DONE]") {
		t.Errorf("expected [DONE] in SSE stream, got: %s", wStream.Body.String())
	}

	// 4. The completed request appears in actual scheduler history. The user
	// field is request-provided metadata, not an authenticated tenant identity.
	reqScheduling := httptest.NewRequest(http.MethodGet, "/api/runtime/scheduling", nil)
	wScheduling := httptest.NewRecorder()
	srv.ServeHTTP(wScheduling, reqScheduling)
	if wScheduling.Code != http.StatusOK {
		t.Fatalf("runtime scheduling status = %d, want 200", wScheduling.Code)
	}
	var scheduling RuntimeSchedulingTelemetry
	if err := json.Unmarshal(wScheduling.Body.Bytes(), &scheduling); err != nil {
		t.Fatalf("decode runtime scheduling: %v", err)
	}
	if scheduling.RequestQueue.State != "observed" || scheduling.RequestQueue.Capacity != 1 || len(scheduling.RequestQueue.Recent) == 0 {
		t.Errorf("request queue = %+v, want recorded service request", scheduling.RequestQueue)
	}
	foundUser := false
	for _, slot := range scheduling.RequestQueue.Recent {
		foundUser = foundUser || slot.User == "tenant-a"
	}
	if !foundUser {
		t.Errorf("recent slots = %+v, want request-provided user", scheduling.RequestQueue.Recent)
	}
	if scheduling.Speculation.State != "unavailable" {
		t.Errorf("speculation = %+v, want unavailable without paired execution", scheduling.Speculation)
	}
}

func TestRequestSlotScheduler(t *testing.T) {
	srv := NewTelemetryServer(&ModelEngine{}, Config{})
	firstRelease, err := srv.acquireRequestSlot(context.Background(), "first", "user-a", "chat_completion", false)
	if err != nil {
		t.Fatalf("acquire first slot: %v", err)
	}

	type result struct {
		release func()
		err     error
	}
	second := make(chan result, 1)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	go func() {
		release, err := srv.acquireRequestSlot(ctx, "second", "user-b", "chat_completion", true)
		second <- result{release: release, err: err}
	}()

	deadline := time.Now().Add(time.Second)
	for {
		snapshot := srv.runtimeSchedulingTelemetry().RequestQueue
		if snapshot.Running == 1 && snapshot.Queued == 1 {
			if len(snapshot.Slots) != 2 || snapshot.Slots[1].State != "queued" || snapshot.Slots[1].QueuePosition != 1 {
				t.Errorf("queued slots = %+v, want one FIFO waiter", snapshot.Slots)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("scheduler snapshot = %+v, want one running and one queued", snapshot)
		}
		time.Sleep(time.Millisecond)
	}

	firstRelease()
	select {
	case got := <-second:
		if got.err != nil {
			t.Fatalf("acquire second slot: %v", got.err)
		}
		got.release()
	case <-time.After(time.Second):
		t.Fatal("second request did not receive the released slot")
	}

	snapshot := srv.runtimeSchedulingTelemetry().RequestQueue
	if snapshot.Running != 0 || snapshot.Queued != 0 || len(snapshot.Recent) != 2 {
		t.Errorf("final scheduler snapshot = %+v, want two completed requests", snapshot)
	}
}

func TestExecutionNodeSelection(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	dir := t.TempDir()
	local, err := NewIrohClusterNode(ctx, &ModelEngine{metrics: NewRuntimeMetrics(4)}, Config{
		NodeKeyPath:   filepath.Join(dir, "local.key"),
		PeerStatePath: filepath.Join(dir, "local-peers.json"),
	}, 0)
	if err != nil {
		t.Fatalf("create local node: %v", err)
	}
	defer local.Close()
	remote, err := NewIrohClusterNode(ctx, &ModelEngine{metrics: NewRuntimeMetrics(4)}, Config{
		NodeKeyPath:   filepath.Join(dir, "remote.key"),
		PeerStatePath: filepath.Join(dir, "remote-peers.json"),
	}, 0)
	if err != nil {
		t.Fatalf("create remote node: %v", err)
	}
	defer remote.Close()
	var connectErr error
	for attempt := 0; attempt < 3; attempt++ {
		connectErr = local.ConnectIroh(ctx, testEndpointAddr(remote))
		if connectErr == nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if connectErr != nil {
		t.Fatalf("connect nodes: %v", connectErr)
	}

	srv := NewTelemetryServer(local.engine, Config{Name: "local model"})
	srv.SetClusterNode(local)

	request := func(body ChatCompletionRequest) *httptest.ResponseRecorder {
		data, err := json.Marshal(body)
		if err != nil {
			t.Fatalf("marshal request: %v", err)
		}
		w := httptest.NewRecorder()
		srv.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(data)))
		return w
	}

	wNodes := httptest.NewRecorder()
	srv.ServeHTTP(wNodes, httptest.NewRequest(http.MethodGet, "/api/runtime/execution-nodes", nil))
	if wNodes.Code != http.StatusOK {
		t.Fatalf("execution nodes status = %d, want 200: %s", wNodes.Code, wNodes.Body.String())
	}
	var nodes ExecutionNodesResponse
	if err := json.Unmarshal(wNodes.Body.Bytes(), &nodes); err != nil {
		t.Fatalf("decode execution nodes: %v", err)
	}
	if nodes.Self.NodeID != local.NodeID() || !nodes.Self.Local || len(nodes.Peers) != 1 || nodes.Peers[0].NodeID != remote.NodeID() || !nodes.Peers[0].Available {
		t.Fatalf("execution nodes = %+v, want local self and selectable remote peer", nodes)
	}

	wRemote := request(ChatCompletionRequest{Prompt: "selected peer", MaxTokens: 2, ExecutionNodeID: remote.NodeID()})
	if wRemote.Code != http.StatusOK {
		t.Fatalf("remote completion status = %d, want 200: %s", wRemote.Code, wRemote.Body.String())
	}
	var remoteResponse ChatCompletionResponse
	if err := json.Unmarshal(wRemote.Body.Bytes(), &remoteResponse); err != nil {
		t.Fatalf("decode remote response: %v", err)
	}
	if remoteResponse.Execution == nil || remoteResponse.Execution.NodeID != remote.NodeID() || remoteResponse.Execution.Local {
		t.Errorf("remote execution = %+v, want selected remote node", remoteResponse.Execution)
	}
	if remoteResponse.NodeRouting != nil {
		t.Errorf("node routing = %+v, want no advisory split-execution claim", remoteResponse.NodeRouting)
	}
	if len(remoteResponse.Choices) != 1 || remoteResponse.Choices[0].Message.Content == "" || len(remoteResponse.TokenIDs) != 2 {
		t.Errorf("remote response = %+v, want text generated by selected peer", remoteResponse)
	}

	wLocal := request(ChatCompletionRequest{Prompt: "this node", MaxTokens: 1, ExecutionNodeID: "local"})
	if wLocal.Code != http.StatusOK {
		t.Fatalf("local completion status = %d, want 200: %s", wLocal.Code, wLocal.Body.String())
	}
	var localResponse ChatCompletionResponse
	if err := json.Unmarshal(wLocal.Body.Bytes(), &localResponse); err != nil {
		t.Fatalf("decode local response: %v", err)
	}
	if localResponse.Execution == nil || localResponse.Execution.NodeID != local.NodeID() || !localResponse.Execution.Local {
		t.Errorf("local execution = %+v, want local node", localResponse.Execution)
	}

	for _, test := range []struct {
		name string
		body ChatCompletionRequest
		want string
	}{
		{
			name: "unknown peer",
			body: ChatCompletionRequest{Prompt: "no peer", ExecutionNodeID: "unknown"},
			want: "unknown execution node",
		},
		{
			name: "self ID",
			body: ChatCompletionRequest{Prompt: "self", ExecutionNodeID: local.NodeID()},
			want: "use \"local\"",
		},
		{
			name: "remote stream",
			body: ChatCompletionRequest{Prompt: "stream", ExecutionNodeID: remote.NodeID(), Stream: true},
			want: "streaming is unavailable",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			w := request(test.body)
			if w.Code != http.StatusBadRequest || !stringsContains(w.Body.String(), test.want) {
				t.Errorf("status/body = %d %q, want 400 containing %q", w.Code, w.Body.String(), test.want)
			}
		})
	}
}
