package cudatemplate_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatemplate"
)

func TestRenderLlama3(t *testing.T) {
	msgs := []cudatemplate.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Hello!"},
	}

	rendered := cudatemplate.Render(cudatemplate.FormatLlama3, msgs)
	if !strings.Contains(rendered, "<|begin_of_text|>") {
		t.Error("expected begin_of_text delimiter")
	}
	if !strings.Contains(rendered, "<|start_header_id|>system<|end_header_id|>") {
		t.Error("expected system header delimiter")
	}
	if !strings.Contains(rendered, "<|start_header_id|>assistant<|end_header_id|>") {
		t.Error("expected trailing assistant prompt")
	}
}

func TestRenderChatML(t *testing.T) {
	msgs := []cudatemplate.Message{
		{Role: "user", Content: "Hello!"},
	}

	rendered := cudatemplate.Render(cudatemplate.FormatChatML, msgs)
	if !strings.Contains(rendered, "<|im_start|>user\nHello!<|im_end|>") {
		t.Errorf("unexpected ChatML format:\n%s", rendered)
	}
}
