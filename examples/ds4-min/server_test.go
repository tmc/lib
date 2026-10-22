package main

import (
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4server"
)

func TestServerModelRequiresIndependentBackend(t *testing.T) {
	if (&serverModel{}).IndependentSessions() {
		t.Fatal("empty server model claims independent sessions")
	}
	if (*serverModel)(nil).IndependentSessions() {
		t.Fatal("nil server model claims independent sessions")
	}
}

func TestServerModelThinkingDisabledPrefix(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		m := &serverModel{tok: &tokenizer{bos: 0, user: 4, assistant: 5, thinking: 2, thinkingEnd: 3}, think: enabled}
		got, err := m.RenderThinking([]dsv4server.Message{{Role: "user"}}, dsv4server.ThinkNone)
		if err != nil {
			t.Fatal(err)
		}
		if want := []int{0, 4, 5, 3}; !reflect.DeepEqual(got, want) {
			t.Fatalf("thinking enabled=%v: prefix = %v, want %v", enabled, got, want)
		}
	}
}

func TestServerModelRejectsUnsupportedMessages(t *testing.T) {
	m := &serverModel{}
	_, err := m.render([]dsv4server.Message{{Role: "assistant", Content: "answer"}}, ThinkNone)
	if err == nil {
		t.Fatal("server model accepted unsupported assistant message")
	}
}

func TestStopIDs(t *testing.T) {
	got := stopIDs(stopSet{7: true, 11: true})
	if len(got) != 2 || (got[0] != 7 && got[0] != 11) || (got[1] != 7 && got[1] != 11) || got[0] == got[1] {
		t.Fatalf("stopIDs = %v", got)
	}
}
