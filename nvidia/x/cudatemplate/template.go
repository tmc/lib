package cudatemplate

import (
	"fmt"
	"strings"
)

// Message represents a single turn in a chat conversation.
type Message struct {
	Role    string
	Content string
}

// Format identifies standard prompt formatting conventions.
type Format int

const (
	// FormatAuto attempts auto-detection or defaults to LLaMA-3.
	FormatAuto Format = iota
	// FormatLlama3 uses <|start_header_id|> and <|eot_id|> delimiters.
	FormatLlama3
	// FormatChatML uses <|im_start|> and <|im_end|> delimiters (Qwen, Yi, DeepSeek).
	FormatChatML
	// FormatGemma uses <start_of_turn> and <end_of_turn> delimiters.
	FormatGemma
	// FormatRaw formats with plain User/Assistant labels.
	FormatRaw
)

// Render formats messages according to the specified format template.
func Render(fmtType Format, messages []Message) string {
	switch fmtType {
	case FormatChatML:
		return RenderChatML(messages)
	case FormatGemma:
		return RenderGemma(messages)
	case FormatRaw:
		return RenderRaw(messages)
	case FormatLlama3, FormatAuto:
		fallthrough
	default:
		return RenderLlama3(messages)
	}
}

// RenderLlama3 renders messages in the LLaMA-3 format.
func RenderLlama3(messages []Message) string {
	var sb strings.Builder
	sb.WriteString("<|begin_of_text|>")
	for _, m := range messages {
		sb.WriteString(fmt.Sprintf("<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", m.Role, m.Content))
	}
	sb.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return sb.String()
}

// RenderChatML renders messages in the ChatML (<|im_start|>) format.
func RenderChatML(messages []Message) string {
	var sb strings.Builder
	for _, m := range messages {
		sb.WriteString(fmt.Sprintf("<|im_start|>%s\n%s<|im_end|>\n", m.Role, m.Content))
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}

// RenderGemma renders messages in the Gemma format.
func RenderGemma(messages []Message) string {
	var sb strings.Builder
	for _, m := range messages {
		sb.WriteString(fmt.Sprintf("<start_of_turn>%s\n%s<end_of_turn>\n", m.Role, m.Content))
	}
	sb.WriteString("<start_of_turn>model\n")
	return sb.String()
}

// RenderRaw renders messages with simple text labels.
func RenderRaw(messages []Message) string {
	var sb strings.Builder
	for _, m := range messages {
		sb.WriteString(fmt.Sprintf("%s: %s\n", strings.Title(m.Role), m.Content))
	}
	sb.WriteString("Assistant: ")
	return sb.String()
}
