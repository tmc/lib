package dsv4server

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// matrixModel exercises host contracts only. Its tokens are not an upstream
// oracle, and this matrix does not qualify model or renderer parity.
type matrixModel struct{ thinkingTestModel }

func (*matrixModel) NewSession() (*dsv4.Session, error) {
	return dsv4.NewSession(testBackend{}, 128, dsv4.NewStopSet(9))
}

func TestChatRequestMatrix(t *testing.T) {
	prompts := []string{"hello", "Paris?", "1+1", "\"quoted\"", "line\nline", "日本語", "<think>literal", "tool result"}
	for p, prompt := range prompts {
		for _, mode := range []string{"none", "high"} {
			for _, limit := range []int{1, 64} {
				name := fmt.Sprintf("prompt%d/%s/max%d", p, mode, limit)
				t.Run(name, func(t *testing.T) {
					model := &matrixModel{}
					h, err := New(model, Config{Model: "test", MaxTokens: 64})
					if err != nil {
						t.Fatal(err)
					}
					request := map[string]any{"messages": []Message{{Role: "user", Content: prompt}}, "temperature": 0, "max_tokens": limit, "reasoning_effort": mode}
					body, _ := json.Marshal(request)
					rec := httptest.NewRecorder()
					h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(body))))
					if rec.Code != 200 {
						t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
					}
					var response struct {
						Choices []struct {
							Message struct {
								Content string `json:"content"`
							}
							Finish string `json:"finish_reason"`
						}
						Usage struct {
							Completion int `json:"completion_tokens"`
						}
					}
					if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
						t.Fatal(err)
					}
					if len(response.Choices) != 1 || response.Choices[0].Message.Content != strings.Repeat("c", limit) || response.Choices[0].Finish != "length" || response.Usage.Completion != limit {
						t.Fatalf("response: %s", rec.Body.String())
					}
					wantMode := ThinkNone
					if mode == "high" {
						wantMode = ThinkHigh
					}
					if model.mode != wantMode {
						t.Fatalf("render mode %v, want %v", model.mode, wantMode)
					}
					for _, usage := range []bool{false, true} {
						request["stream"] = true
						request["stream_options"] = map[string]bool{"include_usage": usage}
						body, _ = json.Marshal(request)
						rec = httptest.NewRecorder()
						h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(body))))
						if rec.Code != 200 {
							t.Fatalf("stream status %d: %s", rec.Code, rec.Body.String())
						}
						checkMatrixStream(t, rec.Body.String(), limit, usage)
					}
				})
			}
		}
	}
}

func checkMatrixStream(t *testing.T, body string, limit int, includeUsage bool) {
	t.Helper()
	if err := validateMatrixStream(body, limit, includeUsage); err != nil {
		t.Fatal(err)
	}
}

func validateMatrixStream(body string, limit int, includeUsage bool) error {
	var content strings.Builder
	roles, finishes, usages, done := 0, 0, 0, 0
	for _, line := range strings.Split(body, "\n") {
		if line == "" {
			continue
		}
		if line == "data: [DONE]" {
			if finishes != 1 || done != 0 || (includeUsage && usages != 1) {
				return fmt.Errorf("premature or duplicate DONE")
			}
			done++
			continue
		}
		if done != 0 {
			return fmt.Errorf("frame after DONE")
		}
		if !strings.HasPrefix(line, "data: ") {
			return fmt.Errorf("unexpected stream line %q", line)
		}
		var frame struct {
			Choices []struct {
				Delta  map[string]string
				Finish *string `json:"finish_reason"`
			}
			Usage *struct {
				Completion int `json:"completion_tokens"`
			}
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame); err != nil {
			return err
		}
		if frame.Usage != nil {
			usages++
			if finishes != 1 || !includeUsage || frame.Usage.Completion != limit || len(frame.Choices) != 0 {
				return fmt.Errorf("usage frame %s", line)
			}
		}
		for _, choice := range frame.Choices {
			if role, ok := choice.Delta["role"]; ok {
				roles++
				if role != "assistant" || content.Len() != 0 || finishes != 0 || roles != 1 {
					return fmt.Errorf("role frame %s", line)
				}
			}
			if choice.Finish != nil {
				finishes++
				if roles != 1 || finishes != 1 || *choice.Finish != "length" || len(choice.Delta) != 0 {
					return fmt.Errorf("finish frame %s", line)
				}
			}
			if _, ok := choice.Delta["content"]; ok && (roles != 1 || finishes != 0) {
				return fmt.Errorf("content outside generation")
			}
			content.WriteString(choice.Delta["content"])
		}
	}
	wantUsage := 0
	if includeUsage {
		wantUsage = 1
	}
	if roles != 1 || finishes != 1 || done != 1 || usages != wantUsage || content.String() != strings.Repeat("c", limit) {
		return fmt.Errorf("stream counts role=%d finish=%d done=%d usage=%d content=%q", roles, finishes, done, usages, content.String())
	}
	return nil
}

func TestChatStreamComparisonMutations(t *testing.T) {
	const role = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
	const reasoning = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thought\"},\"finish_reason\":null}]}\n\n"
	const finish = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n"
	const done = "data: [DONE]\n\n"
	want := CorpusCase{Status: "HTTP/1.1 200 OK", Request: []byte(`{"stream":true}`), Response: []byte(role + reasoning + finish + done), Tokens: []int{2}}
	for _, tc := range []struct{ name, response string }{
		{"missing role", reasoning + finish + done},
		{"reasoning as content", role + strings.Replace(reasoning, "reasoning_content", "content", 1) + finish + done},
		{"unexpected usage", role + reasoning + finish + "data: {\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\n" + done},
		{"finish changed", role + reasoning + strings.Replace(finish, "length", "stop", 1) + done},
		{"missing terminal", role + reasoning + finish},
		{"duplicate terminal", role + reasoning + finish + done + done},
		{"reordered events", reasoning + role + finish + done},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := CorpusObservation{Status: want.Status, Request: want.Request, Response: []byte(tc.response), Tokens: want.Tokens}
			if err := CompareCorpusCase(want, got); err == nil {
				t.Fatal("stream mutation accepted")
			}
		})
	}
}

func TestMatrixStreamOrderMutations(t *testing.T) {
	const role = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
	const content = "data: {\"choices\":[{\"delta\":{\"content\":\"c\"},\"finish_reason\":null}]}\n\n"
	const finish = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n"
	const usage = "data: {\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\n"
	const done = "data: [DONE]\n\n"
	if err := validateMatrixStream(role+content+finish+usage+done, 1, true); err != nil {
		t.Fatal(err)
	}
	for _, body := range []string{
		done + role + content + finish + usage,
		role + usage + content + finish + done,
		content + role + finish + usage + done,
		role + finish + content + usage + done,
		role + content + finish + usage + done + usage,
		role + content + finish + usage + done + done,
	} {
		if err := validateMatrixStream(body, 1, true); err == nil {
			t.Fatalf("accepted reordered stream %q", body)
		}
	}
}

func TestChatMatrixReasoningBuffers(t *testing.T) {
	model := &streamingThinkingModel{parts: []string{"<think>first", " second</think>", "answer"}}
	h, err := New(model, Config{Model: "test", MaxTokens: 3})
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"x"}],"reasoning_effort":"high","temperature":0,"max_tokens":3,"stream":true}`)))
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var reasoning, content strings.Builder
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var frame struct {
			Choices []struct{ Delta map[string]string }
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame); err != nil {
			t.Fatal(err)
		}
		for _, choice := range frame.Choices {
			reasoning.WriteString(choice.Delta["reasoning_content"])
			content.WriteString(choice.Delta["content"])
		}
	}
	if reasoning.String() != "first second" || content.String() != "answer" {
		t.Fatalf("reasoning=%q content=%q", reasoning.String(), content.String())
	}
}
