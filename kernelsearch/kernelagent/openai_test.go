package kernelagent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"
)

type openAIRoundTrip func(*http.Request) (*http.Response, error)

func (f openAIRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func openAIConfig() OpenAIConfig {
	return OpenAIConfig{APIKey: "inert-test-credential", Model: ModelIdentity{"model", "configured-model", "pinned-revision"}, Timeout: time.Second, MaxResponseBytes: 64 << 10, MaxOutputTokens: 1024}
}

func openAIRequest() Request {
	return Request{Epoch: "epoch-1", Operation: "add", Prompt: "select registered rules", Allowed: []string{"fast"}, Diagnostics: Summary{Classification: "unknown", Inferred: true, Reason: "unavailable"}}
}

func openAIResponse(edit string) string {
	b, _ := json.Marshal(map[string]any{
		"status": "completed", "model": "untrusted-reported-model",
		"output": []any{map[string]any{"type": "message", "role": "assistant", "status": "completed", "content": []any{map[string]any{"type": "output_text", "text": edit}}}},
	})
	return string(b)
}

func openAIHTTP(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}

type openAIRead func([]byte) (int, error)

func (f openAIRead) Read(p []byte) (int, error) { return f(p) }

type openAIBody struct {
	io.Reader
	close func() error
}

func (b openAIBody) Close() error { return b.close() }

func TestOpenAIResponsesRequest(t *testing.T) {
	c := openAIConfig()
	want := openAIRequest()
	calls := 0
	c.Transport = openAIRoundTrip(func(req *http.Request) (*http.Response, error) {
		defer req.Body.Close()
		calls++
		if req.Method != "POST" || req.URL.String() != "https://api.openai.com/v1/responses" || req.Header.Get("Authorization") != "Bearer "+c.APIKey || req.Header.Get("Content-Type") != "application/json" {
			t.Fatal("wrong Responses HTTP request")
		}
		if _, ok := req.Context().Deadline(); !ok {
			t.Fatal("request has no deadline")
		}
		if req.GetBody != nil || req.ContentLength <= 0 {
			t.Fatal("POST body can be replayed")
		}
		data, err := io.ReadAll(req.Body)
		if err != nil || strings.Contains(string(data), c.APIKey) {
			t.Fatal("credential in request body or unreadable body")
		}
		var wire struct {
			Model           string
			Store, Stream   bool
			Tools           []json.RawMessage
			ToolChoice      string `json:"tool_choice"`
			MaxOutputTokens int    `json:"max_output_tokens"`
			Input           []struct{ Role, Content string }
			Text            struct {
				Format struct {
					Type, Name string
					Strict     bool
					Schema     map[string]json.RawMessage
				}
			}
		}
		if err := json.Unmarshal(data, &wire); err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if json.Unmarshal(data, &fields) != nil || string(fields["store"]) != "false" || string(fields["stream"]) != "false" {
			t.Fatal("storage or streaming default was not disabled")
		}
		if wire.Model != c.Model.Name || wire.Store || wire.Stream || len(wire.Tools) != 0 || wire.ToolChoice != "none" || wire.MaxOutputTokens != c.MaxOutputTokens || wire.Text.Format.Type != "json_schema" || !wire.Text.Format.Strict || wire.Text.Format.Name != "kernel_rules" {
			t.Fatal("wrong Responses structured-output contract")
		}
		var context Request
		if len(wire.Input) != 2 || wire.Input[0].Role != "system" || wire.Input[1].Role != "user" || json.Unmarshal([]byte(wire.Input[1].Content), &context) != nil || !reflect.DeepEqual(context, want) {
			t.Fatal("prompt or diagnostic context changed")
		}
		var schema struct {
			Type                 string
			AdditionalProperties bool `json:"additionalProperties"`
			Required             []string
			Properties           map[string]struct {
				Type     string
				MaxItems int `json:"maxItems"`
				Items    struct {
					Type string
					Enum []string
				}
			}
		}
		raw, _ := json.Marshal(wire.Text.Format.Schema)
		if string(wire.Text.Format.Schema["additionalProperties"]) != "false" {
			t.Fatal("schema permits extra properties")
		}
		if json.Unmarshal(raw, &schema) != nil || schema.Type != "object" || schema.AdditionalProperties || !reflect.DeepEqual(schema.Required, []string{"rules"}) || len(schema.Properties) != 1 || schema.Properties["rules"].Type != "array" || schema.Properties["rules"].MaxItems != 1 || !reflect.DeepEqual(schema.Properties["rules"].Items.Enum, want.Allowed) {
			t.Fatal("rule schema is not closed and catalog-bound")
		}
		return openAIHTTP(200, openAIResponse(`{"rules":["fast"]}`)), nil
	})
	p, err := NewOpenAI(c)
	if err != nil {
		t.Fatal(err)
	}
	reply, err := p.Propose(context.Background(), want)
	if err != nil || reply.Model != c.Model || string(reply.Edit) != `{"rules":["fast"]}` || calls != 1 {
		t.Fatalf("reply=%+v calls=%d error=%v", reply, calls, err)
	}
}

func TestOpenAIRefusesBadResponses(t *testing.T) {
	valid := openAIResponse(`{"rules":["fast"]}`)
	for _, tc := range []struct {
		name, body string
		status     int
	}{
		{"HTTP429", valid, 429},
		{"redirect", valid, 307},
		{"redirect-to-get", valid, 302},
		{"malformed", `{`, 200},
		{"duplicate", `{"status":"incomplete","status":"completed","output":[]}`, 200},
		{"trailing", valid + `{}`, 200},
		{"incomplete", strings.Replace(valid, `"status":"completed"`, `"status":"incomplete"`, 1), 200},
		{"missing-status", `{"output":[]}`, 200},
		{"status-alias", strings.ReplaceAll(valid, `"status"`, `"Status"`), 200},
		{"output-alias", strings.Replace(valid, `"output"`, `"Output"`, 1), 200},
		{"refusal", `{"status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":"not permitted"}]}]}`, 200},
		{"tool", `{"status":"completed","output":[{"type":"function_call","name":"execute"}]}`, 200},
		{"no-message", `{"status":"completed","output":[{"type":"reasoning","summary":[]}]}`, 200},
		{"message-incomplete", strings.Replace(valid, `"role":"assistant","status":"completed"`, `"role":"assistant","status":"incomplete"`, 1), 200},
		{"invalid-edit", openAIResponse(`{`), 200},
		{"unknown-field", openAIResponse(`{"rules":[],"command":"execute"}`), 200},
		{"duplicate-edit", openAIResponse(`{"rules":[],"rules":["fast"]}`), 200},
		{"rules-alias", openAIResponse(`{"Rules":["fast"]}`), 200},
		{"null-rules", openAIResponse(`{"rules":null}`), 200},
		{"null-item", openAIResponse(`{"rules":[null]}`), 200},
		{"number-item", openAIResponse(`{"rules":[1]}`), 200},
		{"extra-value", openAIResponse(`{"rules":[]} {}`), 200},
		{"depth", strings.Repeat("[", 34) + "0" + strings.Repeat("]", 34), 200},
		{"oversized-edit", openAIResponse(strings.Repeat(" ", maxProposalBytes+1)), 200},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := openAIConfig()
			calls := 0
			c.Transport = openAIRoundTrip(func(*http.Request) (*http.Response, error) {
				calls++
				resp := openAIHTTP(tc.status, tc.body)
				resp.Header.Set("Location", "https://example.invalid/credential-leak")
				return resp, nil
			})
			p, err := NewOpenAI(c)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := p.Propose(context.Background(), openAIRequest()); err == nil || calls != 1 {
				t.Fatalf("calls=%d error=%v", calls, err)
			}
		})
	}
}

func TestOpenAIBoundsAndCancellation(t *testing.T) {
	c := openAIConfig()
	calls := 0
	c.Transport = openAIRoundTrip(func(req *http.Request) (*http.Response, error) {
		calls++
		<-req.Context().Done()
		return nil, req.Context().Err()
	})
	c.Timeout = 10 * time.Millisecond
	p, err := NewOpenAI(c)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Propose(context.Background(), openAIRequest()); !errors.Is(err, context.DeadlineExceeded) || calls != 1 {
		t.Fatalf("timeout calls=%d error=%v", calls, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Propose(ctx, openAIRequest()); !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatalf("cancellation calls=%d error=%v", calls, err)
	}
	for _, mutate := range []func(*Request){
		func(r *Request) { r.Prompt = strings.Repeat("x", maxProposalBytes+1) },
		func(r *Request) { r.Diagnostics.Counters = make([]StallCount, 65) },
		func(r *Request) { r.Diagnostics.Source = strings.Repeat("x", 1025) },
		func(r *Request) { r.Allowed = []string{"../../escape"} },
		func(r *Request) { r.Allowed = []string{"fast", "fast"} },
	} {
		r := openAIRequest()
		mutate(&r)
		if _, err := p.Propose(context.Background(), r); err == nil || calls != 1 {
			t.Fatal("unbounded input reached transport")
		}
	}
	c = openAIConfig()
	c.MaxResponseBytes = 31
	c.Transport = openAIRoundTrip(func(*http.Request) (*http.Response, error) { return openAIHTTP(200, strings.Repeat("x", 100)), nil })
	p, _ = NewOpenAI(c)
	if _, err := p.Propose(context.Background(), openAIRequest()); err == nil {
		t.Fatal("oversized response accepted")
	}
}

func TestOpenAIConfiguration(t *testing.T) {
	for _, mutate := range []func(*OpenAIConfig){
		func(c *OpenAIConfig) { c.APIKey = "" },
		func(c *OpenAIConfig) { c.APIKey = "secret\nheader" },
		func(c *OpenAIConfig) { c.APIKey = "secret\x01header" },
		func(c *OpenAIConfig) { c.Model.Kind = "fixed" },
		func(c *OpenAIConfig) { c.Model.Name = "" },
		func(c *OpenAIConfig) { c.Model.Version = "" },
		func(c *OpenAIConfig) { c.Timeout = 0 },
		func(c *OpenAIConfig) { c.MaxResponseBytes = 1<<20 + 1 },
		func(c *OpenAIConfig) { c.MaxOutputTokens = 0 },
	} {
		c := openAIConfig()
		mutate(&c)
		if _, err := NewOpenAI(c); err == nil {
			t.Fatal("invalid configuration accepted")
		}
	}
	var p *OpenAI
	if _, err := p.Propose(context.Background(), openAIRequest()); err == nil {
		t.Fatal("nil proposer accepted")
	}
}

func TestOpenAITransportOwnershipAndErrors(t *testing.T) {
	for _, kind := range []string{"transport", "read", "close", "cancel-read", "cancel-close"} {
		t.Run(kind, func(t *testing.T) {
			c := openAIConfig()
			c.Timeout = 10 * time.Millisecond
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls, closes := 0, 0
			c.Transport = openAIRoundTrip(func(req *http.Request) (*http.Response, error) {
				defer req.Body.Close()
				calls++
				if kind == "transport" {
					return nil, errors.New("secret-like-transport-detail")
				}
				var reader io.Reader = strings.NewReader(openAIResponse(`{"rules":["fast"]}`))
				if kind == "read" {
					reader = openAIRead(func([]byte) (int, error) { return 0, errors.New("secret-like-read-detail") })
				}
				if kind == "cancel-read" {
					reader = openAIRead(func([]byte) (int, error) { <-req.Context().Done(); return 0, req.Context().Err() })
				}
				resp := openAIHTTP(200, "")
				resp.Body = openAIBody{reader, func() error {
					closes++
					if kind == "close" {
						return errors.New("secret-like-close-detail")
					}
					if kind == "cancel-close" {
						cancel()
					}
					return nil
				}}
				return resp, nil
			})
			p, err := NewOpenAI(c)
			if err != nil {
				t.Fatal(err)
			}
			_, err = p.Propose(ctx, openAIRequest())
			if err == nil || strings.Contains(err.Error(), "secret-like") || calls != 1 || (kind == "transport" && closes != 0) || (kind != "transport" && closes != 1) {
				t.Fatalf("calls=%d closes=%d error=%v", calls, closes, err)
			}
			if kind == "cancel-read" && !errors.Is(err, context.DeadlineExceeded) {
				t.Fatal(err)
			}
			if kind == "cancel-close" && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
		})
	}
	consumed, closes := 0, 0
	c := openAIConfig()
	c.MaxResponseBytes = 31
	c.Transport = openAIRoundTrip(func(req *http.Request) (*http.Response, error) {
		defer req.Body.Close()
		resp := openAIHTTP(200, "")
		resp.Body = openAIBody{openAIRead(func(p []byte) (int, error) {
			for i := range p {
				p[i] = 'x'
			}
			consumed += len(p)
			return len(p), nil
		}), func() error { closes++; return nil }}
		return resp, nil
	})
	p, _ := NewOpenAI(c)
	if _, err := p.Propose(context.Background(), openAIRequest()); err == nil || consumed != 32 || closes != 1 {
		t.Fatalf("read=%d closed=%d error=%v", consumed, closes, err)
	}
}

func TestOpenAIUsesActualLoopArchiveAndAdmission(t *testing.T) {
	for _, tc := range []struct {
		edit       string
		closeError bool
	}{
		{`{"rules":["fast"]}`, false}, {`{"rules":["../../escape"]}`, false},
		{`{"rules":["sh -c execute"]}`, false}, {`{`, false}, {`{"rules":["fast"]}`, true},
	} {
		edit := tc.edit
		t.Run(edit, func(t *testing.T) {
			f, err := newFixture(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { f.archive.Close() })
			c := openAIConfig()
			calls := 0
			c.Transport = openAIRoundTrip(func(*http.Request) (*http.Response, error) {
				calls++
				resp := openAIHTTP(200, openAIResponse(edit))
				if tc.closeError {
					resp.Body = openAIBody{resp.Body, func() error { return errors.New("cleanup failure") }}
				}
				return resp, nil
			})
			p, err := NewOpenAI(c)
			if err != nil {
				t.Fatal(err)
			}
			f.cfg.Proposer, f.cfg.ExpectedModel = p, c.Model
			l, err := NewLoop(f.archive, f.cfg)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := l.Close(); err != nil {
					t.Error(err)
				}
			})
			result, runErr := l.RunEpoch(context.Background(), "model-epoch", f.state, nil, nil)
			if result != nil {
				t.Cleanup(result.Close)
			}
			valid := edit == `{"rules":["fast"]}` && !tc.closeError
			if valid && (runErr != nil || result.Search == nil || result.Search.Compilations == 0) {
				t.Fatalf("actual search not reached: %v", runErr)
			}
			if !valid && (runErr == nil || len(f.compiled) != 0) {
				t.Fatal("malicious/unparseable edit reached compiler")
			}
			found := false
			for _, record := range f.archive.Records() {
				if record.Kind != "proposal" {
					continue
				}
				found = true
				if record.Proposal.Model != c.Model || record.Proposal.Prompt != f.cfg.Prompt || record.Proposal.ReturnedEdit != edit || record.Proposal.ReportedModel == nil || *record.Proposal.ReportedModel != c.Model || valid != (record.Outcome == "accepted") {
					t.Fatalf("proposal archive mismatch: %+v", record)
				}
			}
			if !found || calls != 1 {
				t.Fatal("missing proposal or repeated request")
			}
		})
	}
}

func FuzzDecodeOpenAI(f *testing.F) {
	f.Add(openAIResponse(`{"rules":["fast"]}`))
	f.Add(`{"status":"completed","output":[]}`)
	f.Add(`{"status":"incomplete","status":"completed"}`)
	f.Add(openAIResponse(`{"rules":[null]}`))
	f.Fuzz(func(t *testing.T, data string) {
		if len(data) > 1<<20 {
			return
		}
		edit, err := decodeOpenAI([]byte(data))
		if len(edit) > maxProposalBytes {
			t.Fatal("unbounded edit")
		}
		if err == nil {
			if !json.Valid(edit) {
				t.Fatal("invalid accepted JSON")
			}
			var object map[string]json.RawMessage
			if json.Unmarshal(edit, &object) != nil || len(object) != 1 || object["rules"] == nil {
				t.Fatal("accepted open edit")
			}
		}
	})
}
