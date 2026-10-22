package main

import (
	"reflect"
	"strings"
	"testing"
)

func maxPrefixTokenizer() *tokenizer {
	t := &tokenizer{bos: 0, user: 4, assistant: 5, thinking: 2, thinkingEnd: 3, tokenID: make(map[string]int), mergeRank: make(map[string]int)}
	for i := 0; i < 256; i++ {
		r := gpt2ByteRune(byte(i))
		t.byteRunes[i] = r
		t.tokenID[string(r)] = i + 10
	}
	return t
}

func TestRenderChatThinkModes(t *testing.T) {
	tok := &tokenizer{bos: 0, user: 4, assistant: 5, thinking: 2, thinkingEnd: 3}
	for _, tc := range []struct {
		name string
		mode ThinkMode
		ctx  int
		want []int
	}{
		{"none", ThinkNone, 4096, []int{0, 4, 5, 3}},
		{"high", ThinkHigh, 4096, []int{0, 4, 5, 2}},
		{"max downgrades", ThinkMax, 4096, []int{0, 4, 5, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tok.RenderChat("", "", tc.mode, tc.ctx)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("tokens = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestRenderChatUsesSystemAndMode(t *testing.T) {
	tok := maxPrefixTokenizer()
	without, err := tok.RenderChat("", "hello", ThinkNone, 4096)
	if err != nil {
		t.Fatal(err)
	}
	with, err := tok.RenderChat("system", "hello", ThinkHigh, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(without, with) || with[len(with)-1] != tok.thinking {
		t.Fatalf("system/mode were ignored: without=%v with=%v", without, with)
	}
}

func TestRenderChatMaxContext(t *testing.T) {
	tok := maxPrefixTokenizer()
	got, err := tok.RenderChat("", "", ThinkMax, maxThinkContext)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) <= 4 || got[len(got)-1] != tok.thinking {
		t.Fatalf("max prompt = %v", got)
	}
}

func TestEncodeRenderedG5DSMLMarker(t *testing.T) {
	// The F7 G5 receipt records 405 prompt IDs and 17 occurrences of the
	// vocabulary token ｜DSML｜ (128825). Keep the shape here so a marker that
	// falls back to ordinary BPE cannot silently pass the renderer gate.
	const wantIDs = 405
	const wantDSML = 17
	const receiptDSMLID = 128825
	tok := &tokenizer{
		tokenID:   map[string]int{"｜DSML｜": receiptDSMLID},
		mergeRank: make(map[string]int),
		dsml:      receiptDSMLID,
		dsmlSet:   true,
	}
	for i := range 256 {
		tok.byteRunes[i] = gpt2ByteRune(byte(i))
	}
	tok.tokenID[string(tok.byteRunes['a'])] = 7
	var text strings.Builder
	for i := 0; i < wantDSML; i++ {
		text.WriteString("a")
		text.WriteString("｜DSML｜")
	}
	text.WriteString(strings.Repeat("a", wantIDs-2*wantDSML))
	got, err := tok.encodeRendered(text.String())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != wantIDs {
		t.Fatalf("G5 prompt IDs = %d, want %d", len(got), wantIDs)
	}
	var markers int
	for _, id := range got {
		if id == receiptDSMLID {
			markers++
		}
	}
	if markers != wantDSML {
		t.Fatalf("G5 DSML markers = %d, want %d", markers, wantDSML)
	}
}
