package dsv4tokenizer

import (
	"reflect"
	"testing"
)

func testTokenizer() *Tokenizer {
	t := &Tokenizer{ids: make(map[string]int), merges: make(map[string]int), runes: make(map[rune]byte), bos: 0, user: 4, assistant: 5, thinking: 2, thinkingEnd: 3, eos: 1, stops: map[int]bool{1: true}}
	for i := range 256 {
		r := byteRune(byte(i))
		t.bytes[i] = r
		t.runes[r] = byte(i)
		t.ids[string(r)] = i + 10
	}
	t.tokens = make([]string, 10)
	for i := range t.tokens {
		t.tokens[i] = string(rune('a' + i))
	}
	return t
}

func TestRenderChatModes(t *testing.T) {
	tok := testTokenizer()
	for _, test := range []struct {
		name string
		mode ThinkMode
		want []int
	}{
		{"none", ThinkNone, []int{0, 4, 5, 3}},
		{"high", ThinkHigh, []int{0, 4, 5, 2}},
		{"max downgrades", ThinkMax, []int{0, 4, 5, 2}},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := tok.RenderChat("", "", test.mode, 4096)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("tokens = %v, want %v", got, test.want)
			}
		})
	}
}

func TestRenderPrefixRequiresAlternatingTranscript(t *testing.T) {
	if _, err := testTokenizer().RenderPrefix("", []Turn{{Role: "ASSISTANT", Content: "x"}}, ThinkNone); err == nil {
		t.Fatal("accepted transcript without USER first")
	}
	if _, err := testTokenizer().RenderPrefix("", []Turn{{Role: "USER", Content: "x"}}, ThinkNone); err == nil {
		t.Fatal("accepted transcript without ASSISTANT last")
	}
}

func TestPiecesPreserveInput(t *testing.T) {
	for _, input := range []string{"Hello, world", "世界 1234\n", "    int x;\n", "!Go?\n"} {
		var got string
		for _, piece := range pieces(input) {
			got += piece
		}
		if got != input {
			t.Fatalf("pieces(%q) joined to %q", input, got)
		}
	}
}

func TestPiecesKeepCloseMarkerPrefix(t *testing.T) {
	got := pieces("x</think>y")
	if len(got) < 2 || got[0] != "x" || got[1] != "</" {
		t.Fatalf("pieces close marker = %#v, want x and </ prefix", got)
	}
}

func TestPiecesKeepPunctuationBeforeLetters(t *testing.T) {
	for _, test := range []struct {
		input string
		want  []string
	}{
		{"x>b", []string{"x", ">b"}},
		{">b", []string{">b"}},
		{"(abc", []string{"(abc"}},
		{"\"word", []string{"\"word"}},
		{"#tag", []string{"#tag"}},
		{"a:b", []string{"a", ":b"}},
		{"1>b", []string{"1", ">b"}},
		{"x>世界", []string{"x", ">世界"}},
	} {
		if got := pieces(test.input); !reflect.DeepEqual(got, test.want) {
			t.Errorf("pieces(%q) = %#v, want %#v", test.input, got, test.want)
		}
	}
}

func TestDecodeRejectsInvalidID(t *testing.T) {
	if _, err := testTokenizer().Decode([]int{-1}); err == nil {
		t.Fatal("accepted negative token ID")
	}
}

func TestEncodeRenderedSplitsAdjacentMarkerText(t *testing.T) {
	tok := testTokenizer()
	got, err := tok.EncodeRendered("<｜User｜>b")
	if err != nil {
		t.Fatal(err)
	}
	// The marker is emitted as one control token; b is then ordinary BPE.
	want := []int{4, tok.ids[string(tok.bytes['b'])]}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("rendered IDs = %v, want %v", got, want)
	}
}

func TestRenderChatAutoDetectsRenderedPrompt(t *testing.T) {
	tok := testTokenizer()
	rendered := "<｜begin▁of▁sentence｜><｜User｜>b<｜Assistant｜></think>"
	for _, prompt := range []string{rendered, "<｜User｜>b"} {
		got, err := tok.RenderChat("ignored", prompt, ThinkMax, 1)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) == 0 || got[0] != tok.bos && prompt == rendered {
			t.Fatalf("prompt %q was not routed through rendered tokenization: %v", prompt, got)
		}
		if prompt == "<｜User｜>b" && got[0] != tok.user {
			t.Fatalf("prompt %q got %v, want user marker first", prompt, got)
		}
	}
}

func TestIsRenderedChatPrefixes(t *testing.T) {
	for _, prompt := range []string{"<｜begin▁of▁sentence｜>x", "<｜User｜>x", "[gMASK]x", "<sop>x", "<|system|>x", "<|user|>x", "<|assistant|>x", "<|observation|>x"} {
		if !IsRenderedChat(prompt) {
			t.Errorf("IsRenderedChat(%q) = false", prompt)
		}
	}
	if IsRenderedChat("plain prompt") {
		t.Fatal("plain prompt was classified as rendered")
	}
}
