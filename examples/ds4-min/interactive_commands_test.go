package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4chat"
)

type fixtureSteeringControl struct {
	loaded bool
	scale  float32
}

func (s *fixtureSteeringControl) steeringFFN() float32 { return s.scale }
func (s *fixtureSteeringControl) setSteeringFFN(scale float32) error {
	if !s.loaded && scale != 0 {
		return errors.New("directions not loaded")
	}
	s.scale = scale
	return nil
}

func TestInteractiveSteeringCommands(t *testing.T) {
	var out bytes.Buffer
	control := &fixtureSteeringControl{loaded: true, scale: 1}
	c := interactiveCommands{output: &out, steering: control}
	for _, line := range []string{"/steer", "/steer 0", "/steer -0.5", "/steer 1"} {
		action, err := c.command(line)
		if err != nil || !action.Handled {
			t.Fatalf("%s: %v", line, err)
		}
	}
	if got, want := out.String(), "Steering FFN: 1.\nSteering FFN: 0.\nSteering FFN: -0.5.\nSteering FFN: 1.\n"; got != want {
		t.Fatalf("output %q want %q", got, want)
	}
	for _, line := range []string{"/steer NaN", "/steer +Inf", "/steer 101", "/steer 1 2"} {
		if _, err := c.command(line); err == nil || control.scale != 1 {
			t.Fatalf("accepted/mutated on %s", line)
		}
	}
	control.loaded = false
	control.scale = 0
	if _, err := c.command("/steer 1"); err == nil || control.scale != 0 {
		t.Fatal("enabled unloaded directions")
	}
	if _, err := c.command("/steer 0"); err != nil {
		t.Fatal(err)
	}
}

func TestInteractiveCommands(t *testing.T) {
	var out bytes.Buffer
	c := interactiveCommands{mode: ThinkHigh, context: 4096, output: &out}
	for _, tt := range []struct {
		input     string
		mode      ThinkMode
		context   int
		wantError bool
	}{
		{"/nothink", ThinkNone, 0, false}, {"/think", ThinkHigh, 0, false},
		{"/think-max", ThinkMax, 0, false}, {"/ctx\t8192", ThinkMax, 8192, false},
		{"/ctx -1", ThinkMax, 0, true}, {"/think 50", ThinkMax, 0, true},
		{"/read", ThinkMax, 0, true}, {"/unknown", ThinkMax, 0, true},
	} {
		action, err := c.command(tt.input)
		if (err != nil) != tt.wantError || c.mode != tt.mode || action.Context != tt.context {
			t.Fatalf("%s: action=%+v mode=%v error=%v", tt.input, action, c.mode, err)
		}
	}
}

func TestInteractiveModeHistory(t *testing.T) {
	tok := maxPrefixTokenizer()
	tok.eos = 9
	turns := []dsv4chat.Message{{Role: "user", Text: "one"}, {Role: "assistant", Tokens: []int{700}, Prefix: []int{tok.assistant, tok.thinking}}, {Role: "user", Text: "two"}}
	got, err := tok.renderInteractive("", turns, ThinkNone, 4096)
	if err != nil {
		t.Fatal(err)
	}
	first, _ := tok.RenderChat("", "one", ThinkHigh, 4096)
	second, _ := tok.Encode("two")
	want := append(first, 700, 9, tok.user)
	want = append(want, second...)
	want = append(want, tok.assistant, tok.thinkingEnd)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens %v want %v", got, want)
	}
}

func TestInteractiveRead(t *testing.T) {
	c := interactiveCommands{output: new(bytes.Buffer), context: 4096}
	action, err := c.command("/read testdata/interactive-read.txt")
	if err != nil {
		t.Fatal(err)
	}
	if !action.Submit || action.Handled || action.Text != "first line\nsecond line\n" {
		t.Fatalf("action %+v", action)
	}
}

func TestInteractiveReadAdmission(t *testing.T) {
	if _, err := readInteractiveText("testdata/interactive-read.txt", 5); err == nil {
		t.Fatal("accepted oversized prompt file")
	}
	path := filepath.Join(t.TempDir(), "nul.txt")
	if err := os.WriteFile(path, []byte("ok\x00"+strings.Repeat("x", 32768)), 0600); err != nil {
		t.Fatal(err)
	}
	data, err := readInteractiveText(path, 2)
	if err != nil || string(data) != "ok" {
		t.Fatalf("data %q error %v", data, err)
	}
}
