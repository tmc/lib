package main

import (
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/tmc/lib/gguf"
)

func TestPinnedTokenizerOracle(t *testing.T) {
	modelPath := os.Getenv("DS4_GGUF_TOKENIZER_MODEL")
	if modelPath == "" {
		t.Skip("tokenizer model path not set")
	}
	f, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Read(f, info.Size())
	if err != nil {
		t.Fatal(err)
	}
	tok, err := newTokenizer(g)
	if err != nil {
		t.Fatal(err)
	}
	if want := (stopSet{1: true}); !reflect.DeepEqual(tok.stops, want) {
		t.Fatalf("stops = %v, want %v", tok.stops, want)
	}
	got, err := tok.RenderNonThinking("You are a helpful assistant", "Hello, 世界! 1234\nGo?\n")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 3476, 477, 260, 11502, 22896, 128803, 19923, 14, 223, 3427, 3, 223, 6895, 22, 201, 5188, 2755, 128804, 128822}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens = %v, want %v", got, want)
	}
	decodedPrompt, err := tok.Decode(got)
	if err != nil {
		t.Fatal(err)
	}
	wantPrompt := "<｜begin▁of▁sentence｜>You are a helpful assistant<｜User｜>Hello, 世界! 1234\nGo?\n<｜Assistant｜></think>"
	if string(decodedPrompt) != wantPrompt {
		t.Fatalf("decoded prompt = %q, want %q", decodedPrompt, wantPrompt)
	}

	textIDs := got[1:6]
	textIDs = append(textIDs, got[7:18]...)
	decoded, err := tok.Decode(textIDs)
	if err != nil {
		t.Fatal(err)
	}
	if want := "You are a helpful assistantHello, 世界! 1234\nGo?\n"; string(decoded) != want {
		t.Fatalf("decoded = %q, want %q", decoded, want)
	}
}

func FuzzJoyAIPieces(f *testing.F) {
	for _, text := range []string{"Hello, world", "世界 1234\n", "    int x;\n"} {
		f.Add(text)
	}
	f.Fuzz(func(t *testing.T, text string) {
		if !utf8.ValidString(text) {
			t.Skip()
		}
		if got := strings.Join(joyAIPieces(text), ""); got != text {
			t.Fatalf("pieces joined to %q, want %q", got, text)
		}
	})
}

func TestTokenizerNoByteFallback(t *testing.T) {
	tok := &tokenizer{
		tokenID:   map[string]int{"a": 0},
		mergeRank: map[string]int{},
	}
	for i := range 256 {
		tok.byteRunes[i] = gpt2ByteRune(byte(i))
	}
	if _, err := tok.Encode("b"); !errors.Is(err, errInvalidTokenizer) {
		t.Fatalf("error = %v, want invalid tokenizer", err)
	}
}

func TestJoyAIPieces(t *testing.T) {
	tests := []struct {
		text string
		want []string
	}{
		{"1234", []string{"123", "4"}},
		{"世界abc", []string{"世界", "abc"}},
		{"!Go?\n", []string{"!Go", "?\n"}},
		{"    int", []string{"   ", " int"}},
	}
	for _, tt := range tests {
		t.Run(tt.text, func(t *testing.T) {
			if got := joyAIPieces(tt.text); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("pieces = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestJoyAIPiecesKeepsPunctuationBeforeLetters(t *testing.T) {
	for _, test := range []struct {
		input string
		want  string
	}{
		{"x>b", ">b"},
		{">b", ">b"},
		{"a<｜User｜>b", ">b"},
		{"(abc", "(abc"},
		{"#tag", "#tag"},
		{"a:b", ":b"},
		{"1>b", ">b"},
		{"x>Paris", ">Paris"},
	} {
		pieces := joyAIPieces(test.input)
		if len(pieces) == 0 || pieces[len(pieces)-1] != test.want {
			t.Fatalf("joyAIPieces(%q) = %#v; want final %q piece", test.input, pieces, test.want)
		}
	}
	got := joyAIPieces("a<｜User｜>b")
	if got[len(got)-1] != ">b" {
		t.Fatalf("joyAIPieces marker boundary = %#v, want final >b piece", got)
	}
	got = joyAIPieces("x>世界")
	if len(got) != 3 || got[1] != ">" || got[2] != "世界" {
		t.Fatalf("joyAIPieces CJK punctuation boundary = %#v, want [x > 世界]", got)
	}
	got = joyAIPieces("x</think>y")
	if len(got) < 2 || got[0] != "x" || got[1] != "</" {
		t.Fatalf("joyAIPieces close marker = %#v, want x and </ prefix", got)
	}
}

func TestRenderedPromptAutoDetection(t *testing.T) {
	tok := maxPrefixTokenizer()
	got, err := tok.RenderChat("ignored", "<｜User｜>b", ThinkMax, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{tok.user, tok.tokenID[string(tok.byteRunes['b'])]}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("rendered prompt = %v, want %v", got, want)
	}
}

func TestRenderedDetectionMutation(t *testing.T) {
	for _, prompt := range []string{"<｜begin▁of▁sentence｜>x", "<｜User｜>x", "[gMASK]x", "<sop>x", "<|system|>x", "<|user|>x", "<|assistant|>x", "<|observation|>x"} {
		if !isRenderedChatPrompt(prompt) {
			t.Errorf("isRenderedChatPrompt(%q) = false", prompt)
		}
	}
	if isRenderedChatPrompt("ordinary text") {
		t.Fatal("ordinary prompt took rendered path")
	}
}

func TestThinkModeStopBoundary(t *testing.T) {
	tok := &tokenizer{thinking: 2, thinkingEnd: 3, stops: stopSet{1: true}}
	for _, id := range []int{1, 2, 3} {
		if !tok.IsStopForMode(id, ThinkNone) {
			t.Errorf("none mode id %d was not a stop", id)
		}
		if id != 1 && tok.IsStopForMode(id, ThinkHigh) {
			t.Errorf("thinking mode id %d was unexpectedly a stop", id)
		}
	}
	if tok.IsStopForMode(9, ThinkNone) || tok.IsStopForMode(9, ThinkHigh) {
		t.Fatal("ordinary token was classified as a stop")
	}
}
