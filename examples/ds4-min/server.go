package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4server"
	"github.com/tmc/lib/examples/ds4-min/dsv4tools"
	"github.com/tmc/lib/gguf"
)

// runServer starts the supported dsv4server HTTP front end without creating a
// second backend ownership path. The runtime and its immutable model data
// live for the lifetime of the process; each request gets an independent
// dsv4 state from that backend.
func runServer(args []string, stdout, stderr io.Writer) (err error) {
	fs := flag.NewFlagSet("ds4-min -server", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var model, listen, system string
	var contextLength, maxTokens, slots int
	var ballast float64
	var warmWeights, nothink bool
	var steering config
	var streaming streamingOptions
	addStreamingFlags(fs, &streaming)
	steeringFlags(fs, &steering)
	fs.StringVar(&model, "model", "", "path to model")
	fs.StringVar(&listen, "listen", "127.0.0.1:8080", "HTTP listen address")
	fs.StringVar(&system, "system", "", "system prompt")
	fs.IntVar(&contextLength, "ctx", 32768, "session context length")
	fs.IntVar(&maxTokens, "max-tokens", 16, "maximum generated tokens")
	fs.IntVar(&slots, "session-slots", 1, "maximum independent request sessions")
	fs.Float64Var(&ballast, "ballast-gib", 0, "allocate and touch CUDA ballast before serving")
	fs.BoolVar(&warmWeights, "warm-weights", false, "fault the CUDA model mapping into memory before serving")
	fs.BoolVar(&nothink, "nothink", false, "disable thinking mode")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("unexpected arguments")
	}
	if err := validateSteeringConfig(&steering, fs); err != nil {
		return err
	}
	if model == "" {
		return errors.New("model is required")
	}
	if contextLength <= 1 {
		return errors.New("ctx must exceed one token")
	}
	if maxTokens <= 0 {
		return errors.New("max-tokens must be positive")
	}
	if slots <= 0 || slots > 64 {
		return errors.New("session-slots must be in [1,64]")
	}

	f, err := os.Open(model)
	if err != nil {
		return fmt.Errorf("open model: %w", err)
	}
	info, err := f.Stat()
	if err != nil {
		return errors.Join(fmt.Errorf("stat model: %w", err), f.Close())
	}
	g, err := gguf.Read(f, info.Size())
	err = errors.Join(err, f.Close())
	if err != nil {
		return fmt.Errorf("read model: %w", err)
	}
	if err := validatePinnedProfile(g); err != nil {
		return fmt.Errorf("model profile: %w", err)
	}
	tok, err := newTokenizer(g)
	if err != nil {
		return fmt.Errorf("create tokenizer: %w", err)
	}
	backend, err := openConfiguredServerBackend(model, contextLength, ballast, warmWeights, streaming)
	if err != nil {
		return fmt.Errorf("open server backend: %w", err)
	}
	defer func() { err = finishBackendCleanup(err, backend) }()
	steered, err := newSteeringBackend(backend, steering)
	if err != nil {
		return err
	}
	backend = steered
	adapter := &serverModel{tok: tok, backend: backend, context: contextLength, system: system, think: !nothink, stops: dsv4.NewStopSet(stopIDs(tok.stops)...)}
	h, err := dsv4server.New(adapter, dsv4server.Config{
		Model:                    "deepseek-v4-flash",
		ModelAliases:             []string{"deepseek-v4-pro"},
		ModelCreated:             1767225600,
		ModelOwner:               "ds4.c",
		ModelName:                "DeepSeek V4 Flash",
		ModelMaxCompletion:       32768,
		ModelSupportedParameters: []string{"tools", "tool_choice", "max_tokens", "temperature", "top_p", "top_k", "min_p", "ignore_eos", "stop", "seed", "stream", "reasoning_effort"},
		MaxTokens:                maxTokens,
		ContextLength:            contextLength,
		SessionSlots:             slots,
	})
	if err != nil {
		return fmt.Errorf("create server: %w", err)
	}
	defer func() { err = errors.Join(err, h.Close()) }()

	srv := &http.Server{Addr: listen, Handler: h}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	errCh := make(chan error, 1)
	go func() { errCh <- srv.ListenAndServe() }()
	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return srv.Shutdown(shutdownCtx)
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	}
}

type serverModel struct {
	tok     *tokenizer
	backend dsv4.IndependentStates
	context int
	system  string
	think   bool
	stops   dsv4.StopSet
}

var _ dsv4server.Model = (*serverModel)(nil)
var _ dsv4server.ToolAwareModel = (*serverModel)(nil)
var _ dsv4server.ThinkingModel = (*serverModel)(nil)
var _ dsv4server.DefaultThinkingModel = (*serverModel)(nil)
var _ dsv4server.IndependentSessionModel = (*serverModel)(nil)

func (m *serverModel) Render(messages []dsv4server.Message) ([]int, error) {
	mode := ThinkNone
	if m.think {
		mode = ThinkHigh
	}
	return m.render(messages, mode)
}

func (m *serverModel) DefaultThinkMode() dsv4server.ThinkMode {
	if !m.think {
		return dsv4server.ThinkNone
	}
	return dsv4server.ThinkHigh
}

func (m *serverModel) RenderTools(messages []dsv4server.Message, tools []dsv4tools.Schema) ([]int, error) {
	if len(tools) == 0 {
		return m.Render(messages)
	}
	text, err := renderToolsText(messages, tools, m.think)
	if err != nil {
		return nil, err
	}
	return m.tok.encodeRendered("<｜begin▁of▁sentence｜>" + text)
}

func renderToolsText(messages []dsv4server.Message, tools []dsv4tools.Schema, think bool) (string, error) {
	var b strings.Builder
	b.WriteString("## Tools\n\nYou have access to a set of tools to help answer the user question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n")
	b.WriteString("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\n")
	b.WriteString("String parameters should be specified as raw text and set `string=\"true\"`. Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nWhen thinking mode is enabled, finish reasoning with </think> before any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n")
	for _, tool := range tools {
		b.Write(orderedToolSchema(tool.Raw))
		b.WriteString("\n\n")
	}
	b.WriteString("You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. Use the exact parameter names from the schemas.")
	for _, message := range messages {
		switch message.Role {
		case "user":
			b.WriteString("<｜User｜>")
			b.WriteString(message.Content)
		case "assistant":
			b.WriteString("<｜Assistant｜><think></think>")
			b.WriteString(message.Content)
			if len(message.ToolCalls) != 0 {
				b.WriteString(toolCallsText(message.ToolCalls))
			}
			b.WriteString("<｜end▁of▁sentence｜>")
		case "tool":
			b.WriteString("<｜User｜>")
			b.WriteString("<tool_result>")
			b.WriteString(message.Content)
			b.WriteString("</tool_result>")
		default:
			return "", fmt.Errorf("unsupported message role %q", message.Role)
		}
	}
	b.WriteString("<｜Assistant｜>")
	if think {
		b.WriteString("<think>")
	} else {
		b.WriteString("</think>")
	}
	return b.String(), nil
}

func orderedToolSchema(raw json.RawMessage) []byte {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return raw
	}
	var b strings.Builder
	b.WriteByte('{')
	first := true
	for _, name := range []string{"name", "description", "parameters"} {
		value, ok := fields[name]
		if !ok {
			continue
		}
		if !first {
			b.WriteByte(',')
		}
		first = false
		key, _ := json.Marshal(name)
		b.Write(key)
		b.WriteByte(':')
		b.Write(value)
	}
	extra := make([]string, 0, len(fields))
	for name := range fields {
		if name != "name" && name != "description" && name != "parameters" {
			extra = append(extra, name)
		}
	}
	sort.Strings(extra)
	for _, name := range extra {
		if !first {
			b.WriteByte(',')
		}
		first = false
		key, _ := json.Marshal(name)
		b.Write(key)
		b.WriteByte(':')
		b.Write(fields[name])
	}
	return []byte(append([]byte(b.String()), '}'))
}

func toolCallsText(calls []dsv4server.ToolCall) string {
	var b strings.Builder
	// These separators are part of the model prompt, not presentation.
	b.WriteString("\n\n<｜DSML｜tool_calls>\n")
	for _, call := range calls {
		b.WriteString("<｜DSML｜invoke name=\"")
		b.WriteString(call.Function.Name)
		b.WriteString("\">\n")
		var args map[string]json.RawMessage
		_ = json.Unmarshal([]byte(call.Function.Arguments), &args)
		names := make([]string, 0, len(args))
		for name := range args {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			value := args[name]
			var text string
			var stringValue string
			if json.Unmarshal(value, &stringValue) == nil {
				text, stringValue = stringValue, "true"
			} else {
				text, stringValue = string(value), "false"
			}
			b.WriteString("<｜DSML｜parameter name=\"")
			b.WriteString(name)
			b.WriteString("\" string=\"")
			b.WriteString(stringValue)
			b.WriteString("\">")
			b.WriteString(text)
			b.WriteString("</｜DSML｜parameter>\n")
		}
		b.WriteString("</｜DSML｜invoke>\n")
	}
	b.WriteString("</｜DSML｜tool_calls>")
	return b.String()
}

func (m *serverModel) RenderThinking(messages []dsv4server.Message, mode dsv4server.ThinkMode) ([]int, error) {
	if !m.think && mode != dsv4server.ThinkNone {
		return nil, errors.New("thinking is disabled")
	}
	think := ThinkHigh
	if mode == dsv4server.ThinkNone {
		think = ThinkNone
	} else if mode == dsv4server.ThinkMax {
		think = ThinkMax
	}
	return m.render(messages, think)
}

func (m *serverModel) render(messages []dsv4server.Message, mode ThinkMode) ([]int, error) {
	if len(messages) == 0 {
		return nil, errors.New("missing user message")
	}
	var turns []int
	system := m.system
	seenUser := false
	for _, message := range messages {
		switch message.Role {
		case "system":
			if seenUser {
				return nil, errors.New("system message must precede user messages")
			}
			system = message.Content
		case "user":
			seenUser = true
			turns = append(turns, m.tok.user)
			part, err := m.tok.Encode(message.Content)
			if err != nil {
				return nil, fmt.Errorf("encode user message: %w", err)
			}
			turns = append(turns, part...)
		case "assistant":
			if !seenUser {
				return nil, errors.New("assistant message precedes user message")
			}
			turns = append(turns, m.tok.assistant)
			if !strings.HasPrefix(message.Content, thinkTokenText) && !strings.HasPrefix(message.Content, thinkEndTokenText) {
				turns = append(turns, m.tok.thinkingEnd)
			}
			part, err := m.tok.encodeRendered(message.Content)
			if err != nil {
				return nil, fmt.Errorf("encode assistant message: %w", err)
			}
			turns = append(turns, part...)
			turns = append(turns, m.tok.eos)
		default:
			return nil, fmt.Errorf("unsupported message role %q", message.Role)
		}
	}
	if !seenUser {
		return nil, errors.New("missing user message")
	}
	ids := []int{m.tok.bos}
	if system != "" {
		part, err := m.tok.Encode(system)
		if err != nil {
			return nil, fmt.Errorf("encode system message: %w", err)
		}
		ids = append(ids, part...)
	}
	ids = append(ids, turns...)
	ids = append(ids, m.tok.assistant)
	if mode == ThinkNone {
		ids = append(ids, m.tok.thinkingEnd)
	} else {
		if mode == ThinkMax && m.context > 0 && m.context < maxThinkContext {
			mode = ThinkHigh
		}
		if mode == ThinkMax {
			part, err := m.tok.Encode(maxThinkPrefix)
			if err != nil {
				return nil, fmt.Errorf("encode maximum-thinking prefix: %w", err)
			}
			ids = append(ids, part...)
		}
		ids = append(ids, m.tok.thinking)
	}
	return ids, nil
}

func (m *serverModel) Decode(ids []int) (string, error) {
	b, err := m.tok.Decode(ids)
	return string(b), err
}

func (m *serverModel) NewSession() (*dsv4.Session, error) {
	return dsv4.NewSession(m.backend, m.context, m.stops)
}

func (m *serverModel) IndependentSessions() bool { return m != nil && m.backend != nil }

func stopIDs(stops stopSet) []int {
	ids := make([]int, 0, len(stops))
	for id := range stops {
		ids = append(ids, id)
	}
	return ids
}
