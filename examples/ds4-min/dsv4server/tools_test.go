package dsv4server

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4tools"
)

type recordingExecutor struct {
	calls []dsv4tools.Call
}

func (e *recordingExecutor) Execute(_ context.Context, call dsv4tools.Call) (string, error) {
	e.calls = append(e.calls, call)
	return "result:" + call.Name, nil
}

func TestExecuteToolTextPreservesCallsAndOrder(t *testing.T) {
	text := `prefix <tool_calls><invoke name="one"><parameter name="x">1</parameter></invoke><invoke name="two"><parameter name="ok" string="true">yes</parameter></invoke></tool_calls> suffix`
	executor := &recordingExecutor{}
	got, err := ExecuteToolText(context.Background(), text, executor)
	if err != nil {
		t.Fatal(err)
	}
	if got.Content != "prefix  suffix" || len(got.Calls) != 2 || !reflect.DeepEqual(got.Outputs, []string{"result:one", "result:two"}) {
		t.Fatalf("execution = %+v", got)
	}
	if got.Calls[0].Name != "one" || string(got.Calls[0].Arguments) != `{"x":1}` || got.Calls[1].Name != "two" || string(got.Calls[1].Arguments) != `{"ok":"yes"}` {
		t.Fatalf("calls = %+v", got.Calls)
	}
	if !reflect.DeepEqual(executor.calls, got.Calls) {
		t.Fatalf("executor calls = %+v, parsed = %+v", executor.calls, got.Calls)
	}
}

func TestToolExecutionFailsClosedMutations(t *testing.T) {
	valid := `<tool_calls><invoke name="x"></invoke></tool_calls>`
	if got, err := ExecuteToolText(context.Background(), "plain text", nil); err != nil || got.Content != "plain text" {
		t.Fatalf("plain text = %+v, %v", got, err)
	}
	if _, err := ExecuteToolText(context.Background(), valid, nil); !errors.Is(err, ErrToolExecutorMissing) {
		t.Fatalf("missing executor error = %v", err)
	}
	for _, text := range []string{
		`<tool_calls><invoke name="x">`,
		`<invoke name="x"></invoke>`,
		`<tool_calls><invoke name="x"><parameter name="x">not json</parameter></invoke></tool_calls>`,
	} {
		if _, err := ExecuteToolText(context.Background(), text, &recordingExecutor{}); err == nil {
			t.Fatalf("malformed tool text accepted: %q", text)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ExecuteToolText(ctx, valid, &recordingExecutor{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled execution error = %v", err)
	}
}

func TestReplayToolTextUsesExactMemoryEntry(t *testing.T) {
	const text = `prefix <tool_calls><invoke name="x"><parameter name="n">1</parameter></invoke></tool_calls> suffix`
	memory := dsv4tools.NewMemory(1)
	ids, err := memory.Record(text)
	if err != nil || len(ids) != 1 {
		t.Fatalf("record = %v, %v", ids, err)
	}
	executor := &recordingExecutor{}
	got, err := ReplayToolText(context.Background(), memory, ids[0], executor)
	if err != nil || got.Content != "prefix  suffix" || len(got.Calls) != 1 {
		t.Fatalf("replay = %+v, %v", got, err)
	}
	if _, err := ReplayToolText(context.Background(), memory, "missing", executor); !errors.Is(err, ErrToolReplayMissing) {
		t.Fatalf("missing replay error = %v", err)
	}
	if _, err := ReplayToolText(context.Background(), nil, ids[0], executor); !errors.Is(err, ErrToolReplayMissing) {
		t.Fatalf("nil replay error = %v", err)
	}
}
