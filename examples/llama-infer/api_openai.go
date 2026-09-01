package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"runtime"
	"strings"
	"time"
)

// ChatMessage represents a single message in the chat history.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest is the incoming payload for /v1/chat/completions.
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Prompt      string        `json:"prompt,omitempty"`
	User        string        `json:"user,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float32       `json:"temperature,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	// ExecutionNodeID selects where this request runs. An omitted value or
	// "local" runs on this server; a registered peer ID runs remotely.
	ExecutionNodeID string `json:"execution_node_id,omitempty"`
}

// ChatCompletionChoice is a single generation candidate.
type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// NodeRouting is a legacy advisory prefill/decode role plan. It does not
// identify where a completion ran; new responses use ExecutionTarget instead.
type NodeRouting struct {
	PrefillNodeID   string  `json:"prefill_node_id"`
	PrefillPlatform string  `json:"prefill_platform"`
	PrefillTFLOPs   float64 `json:"prefill_tflops"`
	DecodeNodeID    string  `json:"decode_node_id"`
	DecodePlatform  string  `json:"decode_platform"`
	DecodeBandwidth float64 `json:"decode_bandwidth_gbs"`
	TotalNodes      int     `json:"total_nodes"`
	Reason          string  `json:"reason"`
}

// ExecutionTarget identifies the node that ran a completion.
// It is distinct from an advisory prefill/decode role plan.
type ExecutionTarget struct {
	NodeID   string `json:"node_id"`
	Platform string `json:"platform"`
	Local    bool   `json:"local"`
}

// ExecutionNode describes one currently registered execution target.
type ExecutionNode struct {
	ExecutionTarget
	Available bool   `json:"available"`
	Reason    string `json:"reason,omitempty"`
}

// ExecutionNodesResponse lists the local server and registered mesh peers.
type ExecutionNodesResponse struct {
	Self  ExecutionNode   `json:"self"`
	Peers []ExecutionNode `json:"peers"`
}

// ChatCompletionResponse is the non-streaming response format.
type ChatCompletionResponse struct {
	ID          string                 `json:"id"`
	Object      string                 `json:"object"`
	Created     int64                  `json:"created"`
	Model       string                 `json:"model"`
	Choices     []ChatCompletionChoice `json:"choices"`
	Usage       map[string]int         `json:"usage"`
	Execution   *ExecutionTarget       `json:"execution_target,omitempty"`
	TokenIDs    []int32                `json:"token_ids,omitempty"`
	NodeRouting *NodeRouting           `json:"node_routing,omitempty"`
}

// ChatCompletionChunkChoice represents a delta chunk in SSE streaming.
type ChatCompletionChunkChoice struct {
	Index        int         `json:"index"`
	Delta        ChatMessage `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

// ChatCompletionChunk is the SSE stream chunk format.
type ChatCompletionChunk struct {
	ID          string                      `json:"id"`
	Object      string                      `json:"object"`
	Created     int64                       `json:"created"`
	Model       string                      `json:"model"`
	Choices     []ChatCompletionChunkChoice `json:"choices"`
	Execution   *ExecutionTarget            `json:"execution_target,omitempty"`
	NodeRouting *NodeRouting                `json:"node_routing,omitempty"`
}

// ModelObject represents a model descriptor in /v1/models.
type ModelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse is the response payload for /v1/models.
type ModelsResponse struct {
	Object string        `json:"object"`
	Data   []ModelObject `json:"data"`
}

func executionTargetFromTelemetry(node NodeTelemetry, local bool) ExecutionTarget {
	platform := node.OS + "/" + node.Arch
	if node.OS == "" || node.Arch == "" {
		platform = runtime.GOOS + "/" + runtime.GOARCH
	}
	return ExecutionTarget{
		NodeID:   node.NodeID,
		Platform: platform,
		Local:    local,
	}
}

func (s *TelemetryServer) executionNodes() ExecutionNodesResponse {
	local := ExecutionNode{
		ExecutionTarget: ExecutionTarget{
			NodeID:   "local",
			Platform: runtime.GOOS + "/" + runtime.GOARCH,
			Local:    true,
		},
		Available: true,
	}
	if s.clusterNode == nil {
		return ExecutionNodesResponse{Self: local}
	}

	mesh := s.clusterNode.MeshState()
	local.ExecutionTarget = executionTargetFromTelemetry(mesh.SelfNode, true)
	peers := make([]ExecutionNode, 0, len(mesh.Peers))
	for _, peer := range mesh.Peers {
		node := ExecutionNode{ExecutionTarget: executionTargetFromTelemetry(peer, false)}
		if _, err := s.clusterNode.PeerAddr(peer.NodeID); err != nil {
			node.Reason = err.Error()
		} else {
			node.Available = true
		}
		peers = append(peers, node)
	}
	return ExecutionNodesResponse{Self: local, Peers: peers}
}

func (s *TelemetryServer) resolveExecutionTarget(nodeID string) (ExecutionTarget, error) {
	selection := strings.TrimSpace(nodeID)
	nodes := s.executionNodes()
	if selection == "" || selection == "local" {
		return nodes.Self.ExecutionTarget, nil
	}
	if selection == nodes.Self.NodeID {
		return ExecutionTarget{}, fmt.Errorf("execution node %q is this server; use \"local\" or omit execution_node_id", selection)
	}
	for _, peer := range nodes.Peers {
		if peer.NodeID != selection {
			continue
		}
		if !peer.Available {
			return ExecutionTarget{}, fmt.Errorf("execution node %q is unavailable: %s", selection, peer.Reason)
		}
		return peer.ExecutionTarget, nil
	}
	return ExecutionTarget{}, fmt.Errorf("unknown execution node %q", selection)
}

func (s *TelemetryServer) handleExecutionNodes(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	_ = json.NewEncoder(w).Encode(s.executionNodes())
}

func (s *TelemetryServer) handleV1Models(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	var data []ModelObject
	if s.clusterNode != nil {
		for _, m := range s.clusterNode.SelfTelemetry().AvailableModels {
			data = append(data, ModelObject{
				ID:      m.ModelID,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "huggingface",
			})
		}
	}
	if len(data) == 0 {
		data = []ModelObject{
			{
				ID:      s.cfg.Name,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "tmc/lib",
			},
		}
	}

	resp := ModelsResponse{
		Object: "list",
		Data:   data,
	}
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *TelemetryServer) handleV1ChatCompletions(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON payload: %v", err), http.StatusBadRequest)
		return
	}

	// Format prompt from messages using standard chat header template
	var promptBuilder strings.Builder
	for _, m := range req.Messages {
		promptBuilder.WriteString(fmt.Sprintf("<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", m.Role, m.Content))
	}
	if len(req.Messages) > 0 {
		promptBuilder.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	}
	prompt := promptBuilder.String()
	if prompt == "" {
		prompt = req.Prompt
	}
	if prompt == "" {
		prompt = "The key to scalable AI architectures is"
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 64
	}

	promptTokensCount := len(prompt) / 4
	if s.engine != nil && s.engine.tokenizer != nil {
		if toks, err := s.engine.tokenizer.Encode(prompt); err == nil && len(toks) > 0 {
			promptTokensCount = len(toks)
		}
	}

	target, err := s.resolveExecutionTarget(req.ExecutionNodeID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.Stream && !target.Local {
		http.Error(w, "streaming is unavailable for a remote execution node", http.StatusBadRequest)
		return
	}

	reqID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	releaseSlot, err := s.acquireRequestSlot(r.Context(), reqID, strings.TrimSpace(req.User), "chat_completion", req.Stream)
	if err != nil {
		http.Error(w, "request canceled while waiting for inference slot", http.StatusRequestTimeout)
		return
	}
	defer releaseSlot()

	if req.Stream {
		// Server-Sent Events (SSE) Streaming
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}

		var fullText strings.Builder
		tokenCount := 0
		if s.engine != nil {
			_ = s.engine.Generate(r.Context(), s.cfg, prompt, maxTokens, func(tokenID int32, tokenText string) bool {
				fullText.WriteString(tokenText)
				tokenCount++
				chunk := ChatCompletionChunk{
					ID:        reqID,
					Object:    "chat.completion.chunk",
					Created:   created,
					Model:     s.cfg.Name,
					Execution: &target,
					Choices: []ChatCompletionChunkChoice{
						{
							Index: 0,
							Delta: ChatMessage{
								Role:    "assistant",
								Content: tokenText,
							},
						},
					},
				}
				bytes, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", bytes)
				flusher.Flush()
				return true
			})
		}

		// Send stream terminator
		finish := "stop"
		if tokenCount >= maxTokens {
			finish = "length"
		}
		finalChunk := ChatCompletionChunk{
			ID:        reqID,
			Object:    "chat.completion.chunk",
			Created:   created,
			Model:     s.cfg.Name,
			Execution: &target,
			Choices: []ChatCompletionChunkChoice{
				{
					Index:        0,
					Delta:        ChatMessage{},
					FinishReason: &finish,
				},
			},
		}
		finalBytes, _ := json.Marshal(finalChunk)
		fmt.Fprintf(w, "data: %s\n\n", finalBytes)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}

	// Non-streaming response
	var fullText strings.Builder
	var tokenIDs []int32
	tokenCount := 0
	if target.Local {
		if s.engine != nil {
			_ = s.engine.Generate(r.Context(), s.cfg, prompt, maxTokens, func(tokenID int32, tokenText string) bool {
				fullText.WriteString(tokenText)
				tokenIDs = append(tokenIDs, tokenID)
				tokenCount++
				return true
			})
		}
	} else {
		result, err := s.clusterNode.GenerateRemote(r.Context(), target.NodeID, prompt, maxTokens)
		if err != nil {
			http.Error(w, fmt.Sprintf("remote generation: %v", err), http.StatusBadGateway)
			return
		}
		target.NodeID = result.NodeID
		fullText.WriteString(result.Text)
		tokenIDs = result.Tokens
		tokenCount = len(result.Tokens)
	}

	finishReason := "stop"
	if tokenCount >= maxTokens {
		finishReason = "length"
	}

	resp := ChatCompletionResponse{
		ID:        reqID,
		Object:    "chat.completion",
		Created:   created,
		Model:     s.cfg.Name,
		Execution: &target,
		TokenIDs:  tokenIDs,
		Choices: []ChatCompletionChoice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: fullText.String(),
				},
				FinishReason: finishReason,
			},
		},
		Usage: map[string]int{
			"prompt_tokens":     promptTokensCount,
			"completion_tokens": tokenCount,
			"total_tokens":      promptTokensCount + tokenCount,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
