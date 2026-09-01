package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
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
}
