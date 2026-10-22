package main

import (
	_ "embed"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4server"
	"github.com/tmc/lib/examples/ds4-min/dsv4tools"
)

//go:embed testdata/g5_upstream_prompt.txt
var g5UpstreamPrompt []byte

func TestRenderToolsMatchesG5UpstreamPrompt(t *testing.T) {
	tools := []dsv4tools.Schema{{Raw: json.RawMessage(`{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}`)}}
	messages := []dsv4server.Message{
		{Role: "user", Content: "What is the weather in Paris right now?"},
		{Role: "assistant", ToolCalls: []dsv4server.ToolCall{{Function: dsv4server.ToolFunction{Name: "get_weather", Arguments: `{"city": "Paris"}`}}}},
		{Role: "tool", Content: `{"temp_c": 18, "sky": "cloudy"}`},
	}
	got, err := renderToolsText(messages, tools, true)
	if err != nil {
		t.Fatal(err)
	}
	got = "<｜begin▁of▁sentence｜>" + got
	want := strings.TrimSuffix(string(g5UpstreamPrompt), "\n")
	if got != want {
		t.Fatalf("G5 prompt differs from upstream: got %d bytes, want %d", len(got), len(want))
	}
}

func TestWrappedToolsValidateBeforeG5Render(t *testing.T) {
	raw := []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}`)}
	tools, err := dsv4tools.Validate(raw)
	if err != nil {
		t.Fatal(err)
	}
	var gotSchema, wantSchema any
	if err := json.Unmarshal(tools[0].Raw, &gotSchema); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(`{"description":"Get the current weather for a city.","name":"get_weather","parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"}}`), &wantSchema); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotSchema, wantSchema) {
		t.Fatalf("validated schema = %s, want unwrapped function schema", tools[0].Raw)
	}
	messages := []dsv4server.Message{
		{Role: "user", Content: "What is the weather in Paris right now?"},
		{Role: "assistant", ToolCalls: []dsv4server.ToolCall{{Function: dsv4server.ToolFunction{Name: "get_weather", Arguments: `{"city": "Paris"}`}}}},
		{Role: "tool", Content: `{"temp_c": 18, "sky": "cloudy"}`},
	}
	got, err := renderToolsText(messages, tools, true)
	if err != nil {
		t.Fatal(err)
	}
	got = "<｜begin▁of▁sentence｜>" + got
	want := strings.TrimSuffix(string(g5UpstreamPrompt), "\n")
	if len(got) != 1852 || got != want {
		t.Fatalf("G5 prompt = %d bytes, want 1852-byte fixture", len(got))
	}
}

func TestRenderToolsHistoryUsesCanonicalRoleMarkers(t *testing.T) {
	tools := []dsv4tools.Schema{{Raw: json.RawMessage(`{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}`)}}
	messages := []dsv4server.Message{
		{Role: "user", Content: "What is the weather in Paris right now?"},
		{Role: "assistant", ToolCalls: []dsv4server.ToolCall{{Function: dsv4server.ToolFunction{Name: "get_weather", Arguments: `{"city": "Paris"}`}}}},
		{Role: "tool", Content: `{"temp_c": 18, "sky": "cloudy"}`},
	}
	got, err := renderToolsText(messages, tools, true)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(got, "<｜User｜>") != 2 {
		t.Fatalf("user markers = %d, want 2", strings.Count(got, "<｜User｜>"))
	}
	if strings.Contains(got, "<｜User｜><｜Assistant｜>") {
		t.Fatal("assistant turn has an extra user marker")
	}
	for _, want := range []string{
		"When thinking mode is enabled, finish reasoning with </think>",
		"You MUST strictly follow the above defined tool name and parameter schemas",
		"<｜Assistant｜><think></think>\n\n<｜DSML｜tool_calls>",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered tools prompt is missing %q", want)
		}
	}
}

func TestOrderedToolSchemaMatchesOracleOrder(t *testing.T) {
	raw := json.RawMessage(`{"parameters":{"type":"object"},"name":"get_weather","description":"Get the weather"}`)
	got := string(orderedToolSchema(raw))
	want := `{"name":"get_weather","description":"Get the weather","parameters":{"type":"object"}}`
	if got != want {
		t.Fatalf("ordered schema = %s, want %s", got, want)
	}
}

func TestToolCallsTextMatchesDSMLWireFormat(t *testing.T) {
	calls := []dsv4server.ToolCall{{Function: dsv4server.ToolFunction{
		Name: "lookup", Arguments: `{"city":"Paris","days":2}`,
	}}}
	got := toolCallsText(calls)
	want := "\n\n<｜DSML｜tool_calls>\n" +
		"<｜DSML｜invoke name=\"lookup\">\n" +
		"<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n" +
		"<｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter>\n" +
		"</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
	if got != want {
		t.Fatalf("tool DSML = %q, want %q", got, want)
	}
}
