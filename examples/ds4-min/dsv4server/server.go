// Package dsv4server provides a deliberately small HTTP front end for ds4-min.
//
// The handler implements only deterministic /v1/chat/completions requests.
// Streaming is supported as the same bounded SSE response shape, while
// unsupported request fields are rejected rather than silently changing the
// model run. A Model creates an independent dsv4.Session for each request; the
// handler never shares mutable inference state between clients.
package dsv4server

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4tools"
	"github.com/tmc/lib/examples/ds4-min/sampler"
)

var (
	ErrInvalidRequest = errors.New("invalid request")
	ErrUnsupported    = errors.New("unsupported request")
	ErrClosed         = errors.New("server is closed")
)

type contextLimitError struct {
	n, limit int
}

func (e *contextLimitError) Error() string {
	return fmt.Sprintf("prompt exceeds context: length %d, context %d", e.n, e.limit)
}

// Model supplies the model-specific prompt and token conversion around the
// backend-independent dsv4 session. Render must return the complete prompt
// token sequence, including any chat template markers.
type Model interface {
	Render([]Message) ([]int, error)
	Decode([]int) (string, error)
	NewSession() (*dsv4.Session, error)
}

// ThinkMode selects the request-scoped reasoning prompt mode.
type ThinkMode uint8

const (
	ThinkNone ThinkMode = iota
	ThinkHigh
	ThinkMax
)

// ThinkingModel opts into request-scoped reasoning prompt rendering. Legacy
// Model implementations reject thinking requests rather than silently
// changing their prompt.
type ThinkingModel interface {
	Model
	RenderThinking([]Message, ThinkMode) ([]int, error)
}

// DefaultThinkingModel reports the mode used when a request omits an
// explicit thinking control. It lets the response adapter classify an
// unmarked default reasoning stream without changing plain Model behavior.
type DefaultThinkingModel interface {
	Model
	DefaultThinkMode() ThinkMode
}

// ToolAwareModel opts into rendering validated tool schemas. A plain Model
// continues to reject tool requests: accepting a schema without passing it to
// Render would silently run a different prompt.
type ToolAwareModel interface {
	Model
	RenderTools([]Message, []dsv4tools.Schema) ([]int, error)
}

// IndependentSessionModel opts into more than one server slot. A model must
// prove that NewSession returns independent mutable state before the handler
// will allocate multiple slots; a single-state backend cannot be made safe by
// host-side queueing alone.
type IndependentSessionModel interface {
	Model
	IndependentSessions() bool
}

// Message is the bounded message shape accepted by Handler. Tool fields are
// retained so a tool transcript can be rendered by a model that supports it.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall is the OpenAI-compatible function-call history shape.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction is the name and JSON argument text for a tool call.
type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Config controls Handler limits. Zero values select conservative defaults.
type Config struct {
	Model string
	// ModelAliases are additional names accepted by the model endpoint and
	// included in /v1/models. They identify the same loaded engine.
	ModelAliases []string
	// The remaining model fields describe the public model resource. Zero
	// values retain the small handler defaults used by tests and callers that
	// do not need an OpenAI-compatible catalog entry.
	ModelCreated             int64
	ModelOwner               string
	ModelName                string
	ModelMaxCompletion       int
	ModelSupportedParameters []string
	MaxTokens                int
	MaxBody                  int64
	// ContextLength is the maximum number of prompt tokens accepted by the
	// model. Zero selects the upstream default of 32768.
	ContextLength int
	CORS          bool
	// SessionSlots bounds independent host-side sessions. Slots are reused
	// between requests, in FIFO order, but no device-side batching is implied.
	SessionSlots int
}

// Handler serves the supported chat-completions subset.
type Handler struct {
	model Model
	cfg   Config
	pool  *sessionPool
	next  atomic.Uint64
}

// New returns a handler backed by model. It rejects a nil model and invalid
// limits before returning, so a constructed Handler is ready to serve.
func New(model Model, cfg Config) (*Handler, error) {
	if model == nil {
		return nil, errors.New("missing model")
	}
	if cfg.Model == "" {
		cfg.Model = "ds4-min"
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 16
	}
	if cfg.MaxBody <= 0 {
		cfg.MaxBody = 64 << 20
	}
	if cfg.ContextLength <= 0 {
		cfg.ContextLength = 32768
	}
	if cfg.SessionSlots <= 0 {
		cfg.SessionSlots = 1
	}
	if cfg.SessionSlots > 64 {
		return nil, errors.New("too many session slots")
	}
	if cfg.SessionSlots > 1 {
		independent, ok := model.(IndependentSessionModel)
		if !ok || !independent.IndependentSessions() {
			return nil, fmt.Errorf("%w: model does not provide independent sessions", ErrUnsupported)
		}
	}
	pool, err := newSessionPool(model, cfg.SessionSlots)
	if err != nil {
		return nil, fmt.Errorf("create session pool: %w", err)
	}
	return &Handler{model: model, cfg: cfg, pool: pool}, nil
}

// Close stops accepting requests and releases idle session state. A request
// already holding a session may finish; its session is released when that
// request returns. Close is idempotent.
func (h *Handler) Close() error {
	if h == nil || h.pool == nil {
		return nil
	}
	return h.pool.close()
}

// ServeHTTP handles GET /v1/models, OPTIONS, and deterministic POST
// /v1/chat/completions. Other endpoints and unsupported request fields fail
// closed with a JSON error.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h.pool.isClosed() {
		writeError(w, http.StatusServiceUnavailable, "server shutting down")
		return
	}
	if h.cfg.CORS {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "*")
	}
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/v1/models":
		h.models(w)
	case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/models/"):
		h.modelByID(w, strings.TrimPrefix(r.URL.Path, "/v1/models/"))
	case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
		h.chat(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/completions":
		h.completions(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/responses":
		h.responses(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/messages":
		h.messages(w, r)
	default:
		writeError(w, http.StatusNotFound, "unknown endpoint")
	}
}

type chatRequest struct {
	Model         string    `json:"model"`
	Messages      []Message `json:"messages"`
	MaxTokens     *int      `json:"max_tokens"`
	MaxCompletion *int      `json:"max_completion_tokens"`
	samplingRequest
	Stream        bool            `json:"stream"`
	StreamOptions *streamOptions  `json:"stream_options"`
	Tools         json.RawMessage `json:"tools"`
	ToolChoice    any             `json:"tool_choice"`
	IgnoreEOS     any             `json:"ignore_eos"`
	Stop          any             `json:"stop"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

func chatUsage(prompt, completion int) map[string]any {
	return map[string]any{
		"prompt_tokens": prompt, "completion_tokens": completion,
		"total_tokens":          prompt + completion,
		"prompt_tokens_details": map[string]int{"cached_tokens": 0, "cache_write_tokens": prompt},
	}
}

func responseUsage(input, output int) map[string]any {
	return map[string]any{
		"input_tokens": input, "output_tokens": output, "total_tokens": input + output,
		"input_tokens_details":  map[string]int{"cached_tokens": 0, "cache_write_tokens": input},
		"output_tokens_details": map[string]int{"reasoning_tokens": 0},
	}
}

func messagesUsage(input, output int) map[string]any {
	return map[string]any{
		"input_tokens": 0, "output_tokens": output,
		"cache_read_input_tokens": 0, "cache_creation_input_tokens": input,
	}
}

// samplingRequest is the subset of upstream sampling controls that the
// host-side sampler can represent. An omitted control uses sampler.Default;
// an explicit temperature of zero selects greedy sampling. Thinking controls
// are handled by the optional ThinkingModel interface after request
// validation.
type samplingRequest struct {
	Temperature     *float32 `json:"temperature"`
	TopP            *float32 `json:"top_p"`
	MinP            *float32 `json:"min_p"`
	TopK            *int     `json:"top_k"`
	Seed            *uint64  `json:"seed"`
	Thinking        any      `json:"thinking"`
	Think           any      `json:"think"`
	ReasoningEffort any      `json:"reasoning_effort"`
}

func requestThinkMode(thinking, think, effort any) (ThinkMode, bool, error) {
	if (thinking != nil && think != nil) || (thinking != nil && effort != nil) || (think != nil && effort != nil) {
		return ThinkNone, false, errors.New("thinking, think and reasoning_effort conflict")
	}
	if effort != nil {
		s, ok := effort.(string)
		if !ok {
			return ThinkNone, false, errors.New("reasoning_effort must be a string")
		}
		switch strings.ToLower(s) {
		case "none", "off":
			return ThinkNone, true, nil
		case "low", "medium", "high":
			return ThinkHigh, true, nil
		case "max":
			return ThinkMax, true, nil
		default:
			return ThinkNone, false, errors.New("unsupported reasoning_effort")
		}
	}
	if think != nil {
		v, ok := think.(bool)
		if !ok {
			return ThinkNone, false, errors.New("think must be a boolean")
		}
		if v {
			return ThinkHigh, true, nil
		}
		return ThinkNone, true, nil
	}
	if thinking == nil {
		return ThinkNone, false, nil
	}
	switch v := thinking.(type) {
	case bool:
		if v {
			return ThinkHigh, true, nil
		}
		return ThinkNone, true, nil
	case string:
		switch strings.ToLower(v) {
		case "none", "off", "disabled":
			return ThinkNone, true, nil
		case "low", "medium", "high", "enabled":
			return ThinkHigh, true, nil
		case "max":
			return ThinkMax, true, nil
		}
	case map[string]any:
		t, ok := v["type"].(string)
		if !ok {
			return ThinkNone, false, errors.New("thinking.type must be a string")
		}
		switch strings.ToLower(t) {
		case "disabled":
			return ThinkNone, true, nil
		case "enabled":
			return ThinkHigh, true, nil
		}
	}
	return ThinkNone, false, errors.New("unsupported thinking mode")
}

func (r samplingRequest) sampler() (*sampler.Sampler, sampler.Params, *sampler.RNG, error) {
	p := sampler.Default
	if r.Temperature != nil {
		p.Temperature = *r.Temperature
	}
	if r.TopP != nil {
		p.TopP = *r.TopP
	}
	if r.MinP != nil {
		p.MinP = *r.MinP
	}
	if r.TopK != nil {
		p.TopK = *r.TopK
	}
	if r.Temperature != nil && (*r.Temperature < 0 || *r.Temperature > 100) {
		return nil, sampler.Params{}, nil, errors.New("temperature is outside 0 through 100")
	}
	if r.TopP != nil && (*r.TopP < 0 || *r.TopP > 1) {
		return nil, sampler.Params{}, nil, errors.New("top_p is outside 0 through 1")
	}
	if r.MinP != nil && (*r.MinP < 0 || *r.MinP > 1) {
		return nil, sampler.Params{}, nil, errors.New("min_p is outside 0 through 1")
	}
	if r.TopK != nil && (*r.TopK < 0 || *r.TopK > 1024) {
		return nil, sampler.Params{}, nil, errors.New("top_k is outside 0 through 1024")
	}
	if r.Seed != nil && *r.Seed == 0 {
		return nil, sampler.Params{}, nil, errors.New("seed must be positive")
	}
	if r.Seed != nil && (r.Temperature == nil || *r.Temperature <= 0) {
		return nil, sampler.Params{}, nil, errors.New("seed requires positive temperature")
	}
	rng := &sampler.RNG{}
	if r.Seed != nil {
		rng.State = *r.Seed
	} else if p.Temperature > 0 {
		var b [8]byte
		if _, err := cryptorand.Read(b[:]); err != nil {
			return nil, sampler.Params{}, nil, fmt.Errorf("generate sampling seed: %w", err)
		}
		for _, v := range b {
			rng.State = rng.State<<8 | uint64(v)
		}
	}
	return sampler.New(0), p, rng, nil
}

func sampleToken(logits []float32, smp *sampler.Sampler, p sampler.Params, rng *sampler.RNG) int {
	if p.Temperature <= 0 {
		return sampler.Argmax(logits)
	}
	return smp.Sample(logits, p, rng)
}

func (h *Handler) chat(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, h.cfg.MaxBody)
	var req chatRequest
	if !decodeRequest(w, r.Body, &req) {
		return
	}
	if !h.acceptsModel(req.Model) {
		writeError(w, http.StatusBadRequest, "unknown model")
		return
	}
	if req.StreamOptions != nil && !req.Stream || req.IgnoreEOS != nil {
		writeError(w, http.StatusBadRequest, "request field is unsupported")
		return
	}
	toolsDisabled, err := chatToolsDisabled(req.ToolChoice)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	stops, err := parseStop(req.Stop)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	think, thinkSet, err := requestThinkMode(req.Thinking, req.Think, req.ReasoningEffort)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	smp, params, rng, err := req.sampler()
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	tools, err := parseTools(req.Tools)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.MaxTokens != nil && req.MaxCompletion != nil {
		writeError(w, http.StatusBadRequest, "max_tokens and max_completion_tokens conflict")
		return
	}
	maxTokens := h.cfg.MaxTokens
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	if req.MaxCompletion != nil {
		maxTokens = *req.MaxCompletion
	}
	if maxTokens <= 0 || maxTokens > h.cfg.MaxTokens {
		writeError(w, http.StatusBadRequest, "max token limit exceeded")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "missing messages")
		return
	}
	for _, message := range req.Messages {
		if message.Role != "user" && message.Role != "system" && message.Role != "assistant" &&
			(message.Role != "tool" || len(tools) == 0) {
			writeError(w, http.StatusBadRequest, "unsupported message role")
			return
		}
	}
	toolsRequested := len(req.Tools) != 0 && string(req.Tools) != "null" && !toolsDisabled
	prompt, err := h.renderMode(req.Messages, tools, toolsRequested, think, thinkSet)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := h.contextError(len(prompt)); err != nil {
		limit := err.(*contextLimitError)
		writeContextError(w, "messages", limit.n, limit.limit)
		return
	}
	lease, err := h.pool.acquire(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer lease.Release()
	session := lease.session
	if err := session.Sync(prompt); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Stream {
		h.stream(w, r, session, prompt, maxTokens, requestModel(req.Model, h.cfg.Model), req.StreamOptions != nil && req.StreamOptions.IncludeUsage, smp, params, rng, think, thinkSet, stops)
		return
	}
	generated := make([]int, 0, maxTokens)
	stopped := false
	for len(generated) < maxTokens {
		if err := r.Context().Err(); err != nil {
			writeError(w, http.StatusRequestTimeout, "request canceled")
			return
		}
		id := sampleToken(session.Logits(), smp, params, rng)
		generated = append(generated, id)
		stop, err := session.Accept(id)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		if stop {
			generated = generated[:len(generated)-1]
			stopped = true
			break
		}
	}
	text, err := h.model.Decode(generated)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	text, stoppedBySequence := stops.cut(text)
	stopped = stopped || stoppedBySequence
	var output thinkingOutput
	if toolsRequested {
		output, err = h.classifyToolOutput(text)
	} else {
		output, err = h.classifyThinkingOutput(text, think, thinkSet)
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	message := map[string]any{"role": "assistant", "content": output.content}
	if output.present {
		message["reasoning_content"] = output.reasoning
	}
	id := "chatcmpl-" + strconv.FormatUint(h.next.Add(1), 10)
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "chat.completion", "created": time.Now().Unix(), "model": requestModel(req.Model, h.cfg.Model),
		"choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": openAIFinishReason(stopped, len(generated), maxTokens)}},
		"usage":   chatUsage(len(prompt), len(generated)),
	})
}

type completionRequest struct {
	Model         string          `json:"model"`
	Prompt        json.RawMessage `json:"prompt"`
	MaxTokens     *int            `json:"max_tokens"`
	MaxCompletion *int            `json:"max_completion_tokens"`
	samplingRequest
	Stream        bool           `json:"stream"`
	StreamOptions *streamOptions `json:"stream_options"`
	Stop          any            `json:"stop"`
}

func (h *Handler) completions(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, h.cfg.MaxBody)
	var req completionRequest
	if !decodeRequest(w, r.Body, &req) {
		return
	}
	if !h.acceptsModel(req.Model) {
		writeError(w, http.StatusBadRequest, "unknown model")
		return
	}
	if req.StreamOptions != nil && !req.Stream {
		writeError(w, http.StatusBadRequest, "request field is unsupported")
		return
	}
	stops, err := parseStop(req.Stop)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	think, thinkSet, err := requestThinkMode(req.Thinking, req.Think, req.ReasoningEffort)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	smp, params, rng, err := req.sampler()
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	promptText, ok := completionPrompt(req.Prompt)
	if !ok {
		writeError(w, http.StatusBadRequest, "missing prompt")
		return
	}
	if req.MaxTokens != nil && req.MaxCompletion != nil {
		writeError(w, http.StatusBadRequest, "max_tokens and max_completion_tokens conflict")
		return
	}
	maxTokens := h.cfg.MaxTokens
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	if req.MaxCompletion != nil {
		maxTokens = *req.MaxCompletion
	}
	if maxTokens <= 0 || maxTokens > h.cfg.MaxTokens {
		writeError(w, http.StatusBadRequest, "max token limit exceeded")
		return
	}
	prompt, err := h.renderMode([]Message{
		{Role: "system", Content: "You are a helpful assistant"},
		{Role: "user", Content: promptText},
	}, nil, false, think, thinkSet)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := h.contextError(len(prompt)); err != nil {
		limit := err.(*contextLimitError)
		writeContextError(w, "prompt", limit.n, limit.limit)
		return
	}
	if req.Stream {
		h.streamCompletions(w, r, prompt, maxTokens, requestModel(req.Model, h.cfg.Model), req.StreamOptions != nil && req.StreamOptions.IncludeUsage, smp, params, rng, stops)
		return
	}
	lease, err := h.pool.acquire(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer lease.Release()
	session := lease.session
	if err := session.Sync(prompt); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	generated := make([]int, 0, maxTokens)
	stopped := false
	for len(generated) < maxTokens {
		if err := r.Context().Err(); err != nil {
			writeError(w, http.StatusRequestTimeout, "request canceled")
			return
		}
		id := sampleToken(session.Logits(), smp, params, rng)
		stop, err := session.Accept(id)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		if stop {
			stopped = true
			break
		}
		generated = append(generated, id)
	}
	text, err := h.model.Decode(generated)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	text, stoppedBySequence := stops.cut(text)
	stopped = stopped || stoppedBySequence
	choice := map[string]any{"index": 0, "text": text, "finish_reason": openAIFinishReason(stopped, len(generated), maxTokens)}
	id := "cmpl-" + strconv.FormatUint(h.next.Add(1), 10)
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "text_completion", "created": time.Now().Unix(), "model": requestModel(req.Model, h.cfg.Model),
		"choices": []any{choice},
		"usage":   chatUsage(len(prompt), len(generated)),
	})
}

// completionPrompt accepts the upstream completion request forms: a string,
// or an array whose first element is a string. Other forms are rejected
// instead of being converted implicitly into a prompt.
func completionPrompt(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", false
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, text != ""
	}
	var values []json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil || len(values) == 0 {
		return "", false
	}
	if err := json.Unmarshal(values[0], &text); err != nil {
		return "", false
	}
	return text, text != ""
}

func (h *Handler) streamCompletions(w http.ResponseWriter, r *http.Request, prompt []int, maxTokens int, model string, includeUsage bool, smp *sampler.Sampler, params sampler.Params, rng *sampler.RNG, stops stopSequences) {
	lease, err := h.pool.acquire(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer lease.Release()
	if err := lease.session.Sync(prompt); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming is unsupported by this writer")
		return
	}
	id := "cmpl-" + strconv.FormatUint(h.next.Add(1), 10)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	if !writeSSE(w, map[string]any{"id": id, "object": "text_completion", "created": time.Now().Unix(), "model": model, "choices": []any{map[string]any{"index": 0, "text": "", "finish_reason": nil}}}) {
		return
	}
	flusher.Flush()
	generated := 0
	stopped := false
	var pending string
	for generated < maxTokens {
		if err := r.Context().Err(); err != nil {
			return
		}
		token := sampleToken(lease.session.Logits(), smp, params, rng)
		stop, err := lease.session.Accept(token)
		if err != nil {
			writeStreamError(w, err)
			flusher.Flush()
			return
		}
		if stop {
			stopped = true
			break
		}
		text, err := h.model.Decode([]int{token})
		if err != nil {
			writeStreamError(w, err)
			flusher.Flush()
			return
		}
		text, hit := stops.add(&pending, text)
		if hit {
			stopped = true
		}
		if text != "" && !writeSSE(w, map[string]any{"id": id, "object": "text_completion", "created": time.Now().Unix(), "model": model, "choices": []any{map[string]any{"index": 0, "text": text, "finish_reason": nil}}}) {
			return
		}
		flusher.Flush()
		generated++
		if stopped {
			break
		}
	}
	if !stopped {
		if text := stops.finish(&pending); text != "" && !writeSSE(w, map[string]any{"id": id, "object": "text_completion", "created": time.Now().Unix(), "model": model, "choices": []any{map[string]any{"index": 0, "text": text, "finish_reason": nil}}}) {
			return
		}
	}
	if !writeSSE(w, map[string]any{"id": id, "object": "text_completion", "created": time.Now().Unix(), "model": model, "choices": []any{map[string]any{"index": 0, "text": "", "finish_reason": openAIFinishReason(stopped, generated, maxTokens)}}}) {
		return
	}
	flusher.Flush()
	if includeUsage && !writeSSE(w, map[string]any{"id": id, "object": "text_completion", "created": time.Now().Unix(), "model": model, "choices": []any{}, "usage": chatUsage(len(prompt), generated)}) {
		return
	}
	if includeUsage {
		flusher.Flush()
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	flusher.Flush()
}

type responsesRequest struct {
	Model           string `json:"model"`
	Input           string `json:"input"`
	Instructions    string `json:"instructions"`
	MaxTokens       *int   `json:"max_tokens"`
	MaxOutputTokens *int   `json:"max_output_tokens"`
	samplingRequest
	Stream       bool            `json:"stream"`
	Tools        json.RawMessage `json:"tools"`
	ToolChoice   any             `json:"tool_choice"`
	Reasoning    any             `json:"reasoning"`
	PreviousID   any             `json:"previous_response_id"`
	Conversation any             `json:"conversation"`
}

func (h *Handler) responses(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, h.cfg.MaxBody)
	var req responsesRequest
	if !decodeRequest(w, r.Body, &req) {
		return
	}
	if !h.acceptsModel(req.Model) {
		writeError(w, http.StatusBadRequest, "unknown model")
		return
	}
	smp, params, rng, err := req.sampler()
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.ToolChoice != nil || req.PreviousID != nil || req.Conversation != nil {
		writeError(w, http.StatusBadRequest, "request field is unsupported")
		return
	}
	var reasoningEffort any = req.ReasoningEffort
	if req.Reasoning != nil {
		reasoning, ok := req.Reasoning.(map[string]any)
		if !ok || len(reasoning) != 1 || reasoning["effort"] == nil {
			writeError(w, http.StatusBadRequest, "unsupported reasoning request")
			return
		}
		if reasoningEffort != nil {
			writeError(w, http.StatusBadRequest, "reasoning and reasoning_effort conflict")
			return
		}
		reasoningEffort = reasoning["effort"]
	}
	think, thinkSet, err := requestThinkMode(nil, nil, reasoningEffort)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	tools, err := parseTools(req.Tools)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Input == "" {
		writeError(w, http.StatusBadRequest, "missing input")
		return
	}
	maxTokens, ok := boundedMaxTokens(h.cfg.MaxTokens, req.MaxTokens, req.MaxOutputTokens)
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid max token limit")
		return
	}
	messages := []Message{{Role: "user", Content: req.Input}}
	if req.Instructions != "" {
		messages = append([]Message{{Role: "system", Content: req.Instructions}}, messages...)
	}
	if req.Stream {
		if len(tools) != 0 {
			writeError(w, http.StatusBadRequest, "streaming tool calls are unsupported")
			return
		}
		h.streamResponses(w, r, messages, maxTokens, requestModel(req.Model, h.cfg.Model), smp, params, rng, think, thinkSet)
		return
	}
	toolsRequested := len(req.Tools) != 0 && string(req.Tools) != "null"
	prompt, generated, text, err := h.generateWithTools(r.Context(), messages, maxTokens, tools, toolsRequested, smp, params, rng, think, thinkSet)
	if err != nil {
		var limit *contextLimitError
		if errors.As(err, &limit) {
			writeContextError(w, "input", limit.n, limit.limit)
			return
		}
		if errors.Is(err, ErrUnsupported) {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	var output thinkingOutput
	if toolsRequested {
		output, err = h.classifyToolOutput(text)
	} else {
		output, err = h.classifyThinkingOutput(text, think, thinkSet)
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	id := "resp_" + strconv.FormatUint(h.next.Add(1), 10)
	status := responseStatus(len(generated), maxTokens)
	responseOutput := []any{}
	if output.content != "" {
		messageID := "msg_" + strconv.FormatUint(h.next.Add(1), 10)
		content := []any{map[string]any{"type": "output_text", "text": output.content, "annotations": []any{}}}
		responseOutput = append(responseOutput, map[string]any{"id": messageID, "type": "message", "status": status, "role": "assistant", "content": content})
	}
	if output.present && thinkSet && (status != "incomplete" || output.content != "") {
		responseOutput = append([]any{map[string]any{"id": "rs_" + strconv.FormatUint(h.next.Add(1), 10), "type": "reasoning", "summary": []any{map[string]string{"type": "summary_text", "text": output.reasoning}}}}, responseOutput...)
	}
	response := map[string]any{
		"id": id, "object": "response", "created_at": time.Now().Unix(), "status": status, "model": requestModel(req.Model, h.cfg.Model),
		"output": responseOutput,
		"usage":  responseUsage(len(prompt), len(generated)),
	}
	if status == "incomplete" {
		response["incomplete_details"] = map[string]string{"reason": "max_tokens"}
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *Handler) streamResponses(w http.ResponseWriter, r *http.Request, messages []Message, maxTokens int, model string, smp *sampler.Sampler, params sampler.Params, rng *sampler.RNG, think ThinkMode, thinkSet bool) {
	prompt, err := h.renderMode(messages, nil, false, think, thinkSet)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if promptErr := h.contextError(len(prompt)); promptErr != nil {
		limit := promptErr.(*contextLimitError)
		writeContextError(w, "input", limit.n, limit.limit)
		return
	}
	lease, err := h.pool.acquire(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer lease.Release()
	if err := lease.session.Sync(prompt); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming is unsupported by this writer")
		return
	}
	responseID := "resp_" + strconv.FormatUint(h.next.Add(1), 10)
	messageID := "msg_" + strconv.FormatUint(h.next.Add(1), 10)
	sequence := 0
	emit := func(event string, value map[string]any) bool {
		value["type"] = event
		value["sequence_number"] = sequence
		sequence++
		return writeEvent(w, event, value)
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	if !emit("response.created", map[string]any{"response": map[string]any{"id": responseID, "object": "response", "status": "in_progress", "model": model, "output": []any{}}}) {
		return
	}
	flusher.Flush()
	if !emit("response.output_item.added", map[string]any{"output_index": 0, "item": map[string]any{"id": messageID, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}}}) ||
		!emit("response.content_part.added", map[string]any{"item_id": messageID, "output_index": 0, "content_index": 0, "part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}}}) {
		return
	}
	flusher.Flush()
	textParts := make([]string, 0, maxTokens)
	reasoningParts := make([]string, 0, maxTokens)
	var thinking thinkingStream
	generated := 0
	stopped := false
	for generated < maxTokens {
		if err := r.Context().Err(); err != nil {
			return
		}
		token := sampleToken(lease.session.Logits(), smp, params, rng)
		stop, err := lease.session.Accept(token)
		if err != nil {
			emit("response.failed", map[string]any{"response": map[string]any{"id": responseID, "status": "failed", "error": map[string]string{"message": err.Error()}}})
			flusher.Flush()
			return
		}
		if stop {
			stopped = true
			break
		}
		part, err := h.model.Decode([]int{token})
		if err != nil {
			emit("response.failed", map[string]any{"response": map[string]any{"id": responseID, "status": "failed", "error": map[string]string{"message": err.Error()}}})
			flusher.Flush()
			return
		}
		chunks := []thinkingChunk{{content: part}}
		if thinkSet {
			chunks, err = thinking.feed(part)
			if err != nil {
				emit("response.failed", map[string]any{"response": map[string]any{"id": responseID, "status": "failed", "error": map[string]string{"message": err.Error()}}})
				flusher.Flush()
				return
			}
		}
		for _, chunk := range chunks {
			if chunk.reasoning != "" {
				reasoningParts = append(reasoningParts, chunk.reasoning)
				if !emit("response.reasoning_summary_text.delta", map[string]any{"item_id": messageID, "output_index": 0, "content_index": 0, "delta": chunk.reasoning}) {
					return
				}
			}
			if chunk.content != "" {
				textParts = append(textParts, chunk.content)
				if !emit("response.output_text.delta", map[string]any{"item_id": messageID, "output_index": 0, "content_index": 0, "delta": chunk.content}) {
					return
				}
			}
		}
		flusher.Flush()
		generated++
	}
	if thinkSet {
		chunks, err := thinking.done()
		if err != nil {
			emit("response.failed", map[string]any{"response": map[string]any{"id": responseID, "status": "failed", "error": map[string]string{"message": err.Error()}}})
			flusher.Flush()
			return
		}
		for _, chunk := range chunks {
			if chunk.reasoning != "" {
				reasoningParts = append(reasoningParts, chunk.reasoning)
				if !emit("response.reasoning_summary_text.delta", map[string]any{"item_id": messageID, "output_index": 0, "content_index": 0, "delta": chunk.reasoning}) {
					return
				}
			}
			if chunk.content != "" {
				textParts = append(textParts, chunk.content)
				if !emit("response.output_text.delta", map[string]any{"item_id": messageID, "output_index": 0, "content_index": 0, "delta": chunk.content}) {
					return
				}
			}
		}
		if len(reasoningParts) > 0 && !emit("response.reasoning_summary_text.done", map[string]any{"item_id": messageID, "output_index": 0, "content_index": 0, "text": strings.Join(reasoningParts, "")}) {
			return
		}
	}
	text := strings.Join(textParts, "")
	if !emit("response.output_text.done", map[string]any{"item_id": messageID, "output_index": 0, "content_index": 0, "text": text}) ||
		!emit("response.content_part.done", map[string]any{"item_id": messageID, "output_index": 0, "content_index": 0, "part": map[string]any{"type": "output_text", "text": text, "annotations": []any{}}}) ||
		!emit("response.output_item.done", map[string]any{"output_index": 0, "item": map[string]any{"id": messageID, "type": "message", "status": responseStatus(generated, maxTokens), "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}}}}) ||
		!emit(responseEvent(stopped), map[string]any{"response": map[string]any{"id": responseID, "object": "response", "status": responseStatus(generated, maxTokens), "model": model, "output": []any{map[string]any{"id": messageID, "type": "message", "status": responseStatus(generated, maxTokens), "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}}}}, "usage": responseUsage(len(prompt), generated)}}) {
		return
	}
	flusher.Flush()
}

type messagesRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	System    string    `json:"system"`
	MaxTokens *int      `json:"max_tokens"`
	samplingRequest
	Stream        bool            `json:"stream"`
	Tools         json.RawMessage `json:"tools"`
	ToolChoice    any             `json:"tool_choice"`
	StopSequences any             `json:"stop_sequences"`
	OutputConfig  any             `json:"output_config"`
}

func (h *Handler) messages(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, h.cfg.MaxBody)
	var req messagesRequest
	if !decodeRequest(w, r.Body, &req) {
		return
	}
	if !h.acceptsModel(req.Model) {
		writeError(w, http.StatusBadRequest, "unknown model")
		return
	}
	if req.ToolChoice != nil || req.OutputConfig != nil {
		writeError(w, http.StatusBadRequest, "request field is unsupported")
		return
	}
	stops, err := parseStop(req.StopSequences)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	think, thinkSet, err := requestThinkMode(req.Thinking, req.Think, req.ReasoningEffort)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	smp, params, rng, err := req.sampler()
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	tools, err := parseTools(req.Tools)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "missing messages")
		return
	}
	for _, message := range req.Messages {
		if message.Role != "user" {
			writeError(w, http.StatusBadRequest, "only user messages are supported")
			return
		}
	}
	maxTokens, ok := boundedMaxTokens(h.cfg.MaxTokens, req.MaxTokens, nil)
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid max token limit")
		return
	}
	messages := append([]Message(nil), req.Messages...)
	if req.System != "" {
		messages = append([]Message{{Role: "system", Content: req.System}}, messages...)
	}
	if req.Stream {
		if len(tools) != 0 {
			writeError(w, http.StatusBadRequest, "streaming tool calls are unsupported")
			return
		}
		h.streamMessages(w, r, messages, maxTokens, requestModel(req.Model, h.cfg.Model), smp, params, rng, think, thinkSet, stops)
		return
	}
	toolsRequested := len(req.Tools) != 0 && string(req.Tools) != "null"
	prompt, generated, text, err := h.generateWithTools(r.Context(), messages, maxTokens, tools, toolsRequested, smp, params, rng, think, thinkSet)
	if err != nil {
		var limit *contextLimitError
		if errors.As(err, &limit) {
			writeContextError(w, "messages", limit.n, limit.limit)
			return
		}
		if errors.Is(err, ErrUnsupported) {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	text, stoppedBySequence := stops.cut(text)
	var output thinkingOutput
	if toolsRequested {
		output, err = h.classifyToolOutput(text)
	} else {
		output, err = h.classifyThinkingOutput(text, think, thinkSet)
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	id := "msg_" + strconv.FormatUint(h.next.Add(1), 10)
	content := []any{map[string]string{"type": "text", "text": output.content}}
	if output.present {
		content = append([]any{map[string]string{"type": "thinking", "thinking": output.reasoning, "signature": id}}, content...)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "type": "message", "role": "assistant", "model": requestModel(req.Model, h.cfg.Model),
		"content": content, "stop_reason": anthropicStopReason(stoppedBySequence || len(generated) < maxTokens, len(generated), maxTokens), "stop_sequence": nil,
		"usage": messagesUsage(len(prompt), len(generated)),
	})
}

func (h *Handler) streamMessages(w http.ResponseWriter, r *http.Request, messages []Message, maxTokens int, model string, smp *sampler.Sampler, params sampler.Params, rng *sampler.RNG, think ThinkMode, thinkSet bool, stops stopSequences) {
	prompt, err := h.renderMode(messages, nil, false, think, thinkSet)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if promptErr := h.contextError(len(prompt)); promptErr != nil {
		limit := promptErr.(*contextLimitError)
		writeContextError(w, "messages", limit.n, limit.limit)
		return
	}
	lease, err := h.pool.acquire(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer lease.Release()
	if err := lease.session.Sync(prompt); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming is unsupported by this writer")
		return
	}
	id := "msg_" + strconv.FormatUint(h.next.Add(1), 10)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	if !writeEvent(w, "message_start", map[string]any{"type": "message_start", "message": map[string]any{
		"id": id, "type": "message", "role": "assistant", "model": model,
		"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
		"usage": map[string]int{"input_tokens": len(prompt), "output_tokens": 0},
	}}) {
		return
	}
	flusher.Flush()
	blockIndex := 0
	blockType := "text"
	if thinkSet {
		blockType = "thinking"
	}
	if !writeEvent(w, "content_block_start", map[string]any{"type": "content_block_start", "index": blockIndex, "content_block": map[string]string{"type": blockType, "text": ""}}) {
		return
	}
	flusher.Flush()
	generated := 0
	stopped := false
	var thinking thinkingStream
	var pending string
	thinkingBlock := thinkSet
	for generated < maxTokens {
		if err := r.Context().Err(); err != nil {
			return
		}
		token := sampleToken(lease.session.Logits(), smp, params, rng)
		stop, err := lease.session.Accept(token)
		if err != nil {
			writeEvent(w, "error", map[string]any{"type": "error", "error": map[string]string{"type": "api_error", "message": err.Error()}})
			flusher.Flush()
			return
		}
		if stop {
			stopped = true
			break
		}
		text, err := h.model.Decode([]int{token})
		if err != nil {
			writeEvent(w, "error", map[string]any{"type": "error", "error": map[string]string{"type": "api_error", "message": err.Error()}})
			flusher.Flush()
			return
		}
		text, hit := stops.add(&pending, text)
		if hit {
			stopped = true
		}
		chunks := []thinkingChunk{{content: text}}
		if thinkSet {
			chunks, err = thinking.feed(text)
			if err != nil {
				writeEvent(w, "error", map[string]any{"type": "error", "error": map[string]string{"type": "api_error", "message": err.Error()}})
				flusher.Flush()
				return
			}
		}
		for _, chunk := range chunks {
			if chunk.reasoning != "" {
				if !writeEvent(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": blockIndex, "delta": map[string]string{"type": "thinking_delta", "thinking": chunk.reasoning}}) {
					return
				}
			}
			if chunk.content != "" {
				if thinkingBlock {
					if !writeEvent(w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": blockIndex}) || !writeEvent(w, "content_block_start", map[string]any{"type": "content_block_start", "index": blockIndex + 1, "content_block": map[string]string{"type": "text", "text": ""}}) {
						return
					}
					blockIndex++
					thinkingBlock = false
				}
				if !writeEvent(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": blockIndex, "delta": map[string]string{"type": "text_delta", "text": chunk.content}}) {
					return
				}
			}
		}
		flusher.Flush()
		if stopped {
			break
		}
		generated++
	}
	if !stopped {
		_ = stops.finish(&pending)
	}
	if thinkSet {
		chunks, err := thinking.done()
		if err != nil {
			writeEvent(w, "error", map[string]any{"type": "error", "error": map[string]string{"type": "api_error", "message": err.Error()}})
			flusher.Flush()
			return
		}
		for _, chunk := range chunks {
			if chunk.reasoning != "" && !writeEvent(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": blockIndex, "delta": map[string]string{"type": "thinking_delta", "thinking": chunk.reasoning}}) {
				return
			}
			if chunk.content != "" {
				if thinkingBlock {
					if !writeEvent(w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": blockIndex}) || !writeEvent(w, "content_block_start", map[string]any{"type": "content_block_start", "index": blockIndex + 1, "content_block": map[string]string{"type": "text", "text": ""}}) {
						return
					}
					blockIndex++
					thinkingBlock = false
				}
				if !writeEvent(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": blockIndex, "delta": map[string]string{"type": "text_delta", "text": chunk.content}}) {
					return
				}
			}
		}
	}
	if !writeEvent(w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": blockIndex}) ||
		!writeEvent(w, "message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": anthropicStopReason(stopped, generated, maxTokens), "stop_sequence": nil}, "usage": map[string]int{"output_tokens": generated}}) ||
		!writeEvent(w, "message_stop", map[string]any{"type": "message_stop"}) {
		return
	}
	flusher.Flush()
}

func boundedMaxTokens(defaultMax int, first, second *int) (int, bool) {
	if first != nil && second != nil {
		return 0, false
	}
	max := defaultMax
	if first != nil {
		max = *first
	}
	if second != nil {
		max = *second
	}
	return max, max > 0 && max <= defaultMax
}

func parseTools(raw json.RawMessage) ([]dsv4tools.Schema, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var list []json.RawMessage
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("tools must be an array: %w", err)
	}
	schemas, err := dsv4tools.Validate(list)
	if err != nil {
		return nil, fmt.Errorf("invalid tool schema: %w", err)
	}
	return schemas, nil
}

// chatToolsDisabled recognizes the one explicit chat tool policy that changes
// prompt construction. tool_choice:none keeps the schemas valid but prevents
// them from entering the rendered prompt; any other policy remains unsupported
// rather than silently changing the model input.
func chatToolsDisabled(choice any) (bool, error) {
	if choice == nil {
		return false, nil
	}
	s, ok := choice.(string)
	if !ok || s != "none" {
		return false, errors.New("unsupported tool_choice")
	}
	return true, nil
}

func (h *Handler) render(messages []Message, tools []dsv4tools.Schema, toolsRequested bool) ([]int, error) {
	return h.renderMode(messages, tools, toolsRequested, ThinkNone, false)
}

func (h *Handler) renderMode(messages []Message, tools []dsv4tools.Schema, toolsRequested bool, think ThinkMode, thinkSet bool) ([]int, error) {
	if thinkSet {
		model, ok := h.model.(ThinkingModel)
		if !ok {
			return nil, fmt.Errorf("%w: thinking is unsupported by the current model", ErrUnsupported)
		}
		if toolsRequested {
			return nil, fmt.Errorf("%w: thinking with tools is unsupported", ErrUnsupported)
		}
		return model.RenderThinking(messages, think)
	}
	if !toolsRequested {
		return h.model.Render(messages)
	}
	model, ok := h.model.(ToolAwareModel)
	if !ok {
		return nil, fmt.Errorf("%w: tools are unsupported by the current model", ErrUnsupported)
	}
	return model.RenderTools(messages, tools)
}

func (h *Handler) classifyThinkingOutput(text string, requested ThinkMode, explicit bool) (thinkingOutput, error) {
	output, err := splitThinkingOutput(text)
	if err != nil || output.present || explicit || requested != ThinkNone {
		return output, err
	}
	model, ok := h.model.(DefaultThinkingModel)
	if !ok || model.DefaultThinkMode() == ThinkNone || text == "" {
		return output, nil
	}
	return thinkingOutput{reasoning: text, present: true}, nil
}

// classifyToolOutput treats the closing marker emitted before a tool result or
// final answer as an empty thinking block. Tool prompts deliberately end in a
// thinking boundary, so a model may begin its response with </think> without
// having produced reasoning content. That is ordinary content, not malformed
// reasoning output.
func (h *Handler) classifyToolOutput(text string) (thinkingOutput, error) {
	if strings.HasPrefix(text, thinkEnd) {
		text = strings.TrimPrefix(text, thinkEnd)
	}
	if strings.Contains(text, thinkStart) || strings.Contains(text, thinkEnd) {
		return splitThinkingOutput(text)
	}
	return thinkingOutput{content: text}, nil
}

func (h *Handler) generate(ctx context.Context, messages []Message, maxTokens int) (prompt, generated []int, text string, err error) {
	return h.generateWithTools(ctx, messages, maxTokens, nil, false, sampler.New(0), sampler.Params{}, &sampler.RNG{}, ThinkNone, false)
}

func (h *Handler) generateWithTools(ctx context.Context, messages []Message, maxTokens int, tools []dsv4tools.Schema, toolsRequested bool, smp *sampler.Sampler, params sampler.Params, rng *sampler.RNG, think ThinkMode, thinkSet bool) (prompt, generated []int, text string, err error) {
	prompt, err = h.renderMode(messages, tools, toolsRequested, think, thinkSet)
	if err != nil {
		return nil, nil, "", err
	}
	if promptErr := h.contextError(len(prompt)); promptErr != nil {
		return nil, nil, "", promptErr
	}
	lease, err := h.pool.acquire(ctx)
	if err != nil {
		return nil, nil, "", err
	}
	defer lease.Release()
	session := lease.session
	if err := session.Sync(prompt); err != nil {
		return nil, nil, "", err
	}
	generated = make([]int, 0, maxTokens)
	for len(generated) < maxTokens {
		if err := ctx.Err(); err != nil {
			return nil, nil, "", err
		}
		id := sampleToken(session.Logits(), smp, params, rng)
		stop, err := session.Accept(id)
		if err != nil {
			return nil, nil, "", err
		}
		if stop {
			break
		}
		generated = append(generated, id)
	}
	text, err = h.model.Decode(generated)
	return prompt, generated, text, err
}

func (h *Handler) stream(w http.ResponseWriter, r *http.Request, session *dsv4.Session, prompt []int, maxTokens int, model string, includeUsage bool, smp *sampler.Sampler, params sampler.Params, rng *sampler.RNG, think ThinkMode, thinkSet bool, stops stopSequences) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming is unsupported by this writer")
		return
	}
	id := "chatcmpl-" + strconv.FormatUint(h.next.Add(1), 10)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	if !writeSSE(w, map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": time.Now().Unix(), "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": map[string]string{"role": "assistant"}, "finish_reason": nil}},
	}) {
		return
	}
	flusher.Flush()
	generated := 0
	stopped := false
	var thinking thinkingStream
	defaultReasoning := false
	if !thinkSet && think == ThinkNone {
		if m, ok := h.model.(DefaultThinkingModel); ok {
			defaultReasoning = m.DefaultThinkMode() != ThinkNone
		}
	}
	classify := thinkSet || defaultReasoning
	tagged := false
	var pending string
	var buffered thinkingChunk
	var haveBuffered bool
	writeThinking := func(chunk thinkingChunk) bool {
		if chunk.reasoning == "" && chunk.content == "" {
			return true
		}
		if defaultReasoning && !tagged {
			chunk.reasoning, chunk.content = chunk.content, ""
		}
		if chunk.reasoning == "" {
			if haveBuffered {
				if !writeThinkingChunk(w, id, model, buffered) {
					return false
				}
				haveBuffered = false
				buffered = thinkingChunk{}
			}
			return writeThinkingChunk(w, id, model, chunk)
		}
		if haveBuffered {
			buffered.reasoning += chunk.reasoning
		} else {
			buffered = chunk
			haveBuffered = true
		}
		return true
	}
	for ; generated < maxTokens; generated++ {
		if err := r.Context().Err(); err != nil {
			return
		}
		idToken := sampleToken(session.Logits(), smp, params, rng)
		stop, err := session.Accept(idToken)
		if err != nil {
			writeStreamError(w, err)
			flusher.Flush()
			return
		}
		if stop {
			stopped = true
			break
		}
		text, err := h.model.Decode([]int{idToken})
		if err != nil {
			writeStreamError(w, err)
			flusher.Flush()
			return
		}
		text, hit := stops.add(&pending, text)
		if hit {
			stopped = true
		}
		chunks := []thinkingChunk{{content: text}}
		if classify {
			if thinking.state == streamUnknown && strings.HasPrefix(thinking.pending+text, thinkStart) {
				tagged = true
			}
			chunks, err = thinking.feed(text)
			if err != nil {
				writeStreamError(w, err)
				flusher.Flush()
				return
			}
		}
		for _, chunk := range chunks {
			if !writeThinking(chunk) {
				return
			}
		}
		flusher.Flush()
		if stopped {
			break
		}
	}
	if !stopped {
		if text := stops.finish(&pending); text != "" {
			chunks := []thinkingChunk{{content: text}}
			if classify {
				if thinking.state == streamUnknown && strings.HasPrefix(thinking.pending+text, thinkStart) {
					tagged = true
				}
				var err error
				chunks, err = thinking.feed(text)
				if err != nil {
					writeStreamError(w, err)
					flusher.Flush()
					return
				}
			}
			for _, chunk := range chunks {
				if !writeThinking(chunk) {
					return
				}
			}
		}
	}
	if classify {
		chunks, err := thinking.done()
		if err != nil {
			writeStreamError(w, err)
			flusher.Flush()
			return
		}
		for _, chunk := range chunks {
			if !writeThinking(chunk) {
				return
			}
		}
	}
	if haveBuffered && !writeThinkingChunk(w, id, model, buffered) {
		return
	}
	if !writeSSE(w, map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": time.Now().Unix(), "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": map[string]string{}, "finish_reason": openAIFinishReason(stopped, generated, maxTokens)}},
	}) {
		return
	}
	flusher.Flush()
	if includeUsage && !writeSSE(w, map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": time.Now().Unix(), "model": model,
		"choices": []any{},
		"usage":   chatUsage(len(prompt), generated),
	}) {
		return
	}
	if includeUsage {
		flusher.Flush()
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (h *Handler) models(w http.ResponseWriter) {
	ids := h.modelIDs()
	data := make([]any, 0, len(ids))
	for _, id := range ids {
		data = append(data, h.modelDescription(id))
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (h *Handler) modelByID(w http.ResponseWriter, id string) {
	// Aliases are request selectors, not model-list entries. The configured
	// engine is the only resource this handler can describe.
	if id == "" || !h.isListedModel(id) {
		writeError(w, http.StatusNotFound, "unknown endpoint")
		return
	}
	writeJSON(w, http.StatusOK, h.modelDescription(id))
}

func (h *Handler) modelDescription(id string) map[string]any {
	model := map[string]any{
		"id": id, "object": "model", "owned_by": h.modelOwner(),
		"context_length": h.cfg.ContextLength,
		"top_provider": map[string]any{
			"context_length":        h.cfg.ContextLength,
			"max_completion_tokens": h.modelMaxCompletion(),
			"is_moderated":          false,
		},
	}
	if h.cfg.ModelCreated != 0 {
		model["created"] = h.cfg.ModelCreated
	}
	if name := h.cfg.ModelName; name != "" {
		model["name"] = name
	}
	if params := h.cfg.ModelSupportedParameters; len(params) != 0 {
		model["supported_parameters"] = append([]string(nil), params...)
	}
	return model
}

func (h *Handler) modelIDs() []string {
	ids := []string{h.cfg.Model}
	for _, alias := range h.cfg.ModelAliases {
		if alias == "" || alias == h.cfg.Model || h.isModelID(ids, alias) {
			continue
		}
		ids = append(ids, alias)
	}
	return ids
}

func (h *Handler) isListedModel(id string) bool {
	return h.isModelID(h.modelIDs(), id)
}

func (h *Handler) isModelID(ids []string, id string) bool {
	for _, candidate := range ids {
		if candidate == id {
			return true
		}
	}
	return false
}

func (h *Handler) modelOwner() string {
	if h.cfg.ModelOwner != "" {
		return h.cfg.ModelOwner
	}
	return "ds4-min"
}

func (h *Handler) modelMaxCompletion() int {
	if h.cfg.ModelMaxCompletion > 0 {
		return minInt(h.cfg.ModelMaxCompletion, h.cfg.ContextLength)
	}
	return minInt(h.cfg.MaxTokens, h.cfg.ContextLength)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func openAIFinishReason(stopped bool, generated, maxTokens int) string {
	if stopped {
		return "stop"
	}
	if generated >= maxTokens {
		return "length"
	}
	return "stop"
}

func anthropicStopReason(stopped bool, generated, maxTokens int) string {
	if stopped {
		return "end_turn"
	}
	if generated >= maxTokens {
		return "max_tokens"
	}
	return "end_turn"
}

func responseStatus(generated, maxTokens int) string {
	if generated >= maxTokens {
		return "incomplete"
	}
	return "completed"
}

func responseEvent(stopped bool) string {
	if stopped {
		return "response.completed"
	}
	return "response.incomplete"
}

func (h *Handler) contextError(promptTokens int) error {
	if promptTokens < h.cfg.ContextLength {
		return nil
	}
	return &contextLimitError{n: promptTokens, limit: h.cfg.ContextLength}
}

func writeContextError(w http.ResponseWriter, param string, promptTokens, context int) {
	writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]any{
		"message":         "prompt exceeds context",
		"type":            "invalid_request_error",
		"param":           param,
		"code":            "context_length_exceeded",
		"n_prompt_tokens": promptTokens,
		"n_ctx":           context,
	}})
}

func requestModel(requested, configured string) string {
	if requested != "" {
		return requested
	}
	return configured
}

// acceptsModel reports whether requested selects this configured engine.
// The two upstream DeepSeek aliases are accepted as selectors, but an
// arbitrary model name must not be echoed in a successful response: doing so
// would make a request for a model that was never loaded look successful.
func (h *Handler) acceptsModel(requested string) bool {
	return requested == "" || h.isListedModel(requested) || requested == "deepseek-chat" || requested == "deepseek-reasoner"
}

// decodeRequest accepts exactly one JSON value. Decoder.Decode alone would
// accept a valid request followed by another JSON value, which makes a body
// containing two requests depend on decoder buffering rather than the HTTP
// request contract.
func decodeRequest(w http.ResponseWriter, r io.Reader, value any) bool {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(value); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON request")
		return false
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		writeError(w, http.StatusBadRequest, "invalid JSON request")
		return false
	}
	return true
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"message": message, "type": "invalid_request_error"}})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		return
	}
}

func writeSSE(w http.ResponseWriter, value any) bool {
	data, err := json.Marshal(value)
	if err != nil {
		return false
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", data)
	return err == nil
}

func writeThinkingChunk(w http.ResponseWriter, id, model string, chunk thinkingChunk) bool {
	delta := map[string]string{}
	if chunk.reasoning != "" {
		delta["reasoning_content"] = chunk.reasoning
	}
	if chunk.content != "" {
		delta["content"] = chunk.content
	}
	if len(delta) == 0 {
		return true
	}
	return writeSSE(w, map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": time.Now().Unix(), "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": nil}},
	})
}

func writeEvent(w io.Writer, event string, value any) bool {
	data, err := json.Marshal(value)
	if err != nil {
		return false
	}
	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
	return err == nil
}

func writeStreamError(w http.ResponseWriter, err error) {
	data, marshalErr := json.Marshal(map[string]any{"error": map[string]string{"message": err.Error(), "type": "server_error"}})
	if marshalErr == nil {
		_, _ = fmt.Fprintf(w, "event: error\ndata: %s\n\n", data)
	}
}
