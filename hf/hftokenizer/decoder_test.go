package hftokenizer

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestDecoderChunks(t *testing.T) {
	tok := readTokenizer(t, "testdata/qwen.json")
	for _, data := range [][]byte{
		[]byte("a€😀z"), {0xef, 0xbb}, {0xef, 0xbb, 'a'}, {0xff, 0xff}, {0xed, 0xa0, 0x80}, {0xe0, 0xa0, 0xff}, {0xf4, 0x90, 0x80, 0x80},
	} {
		var ids []int32
		for _, b := range data {
			ids = append(ids, tok.bytes[b])
		}
		want, err := tok.Decode(ids)
		if err != nil {
			t.Fatal(err)
		}
		for width := 1; width <= len(ids); width++ {
			d := tok.NewDecoder()
			var out strings.Builder
			for i := 0; i < len(ids); i += width {
				end := min(i+width, len(ids))
				text, err := d.Decode(ids[i:end], end == len(ids))
				if err != nil {
					t.Fatal(err)
				}
				if !utf8.ValidString(text) {
					t.Fatalf("invalid UTF-8 fragment %x", text)
				}
				out.WriteString(text)
			}
			if got := out.String(); got != want {
				t.Errorf("bytes %x width %d: %q, want %q", data, width, got, want)
			}
			if d.n != 0 {
				t.Fatal("final decode retained bytes")
			}
		}
	}
}

func TestDecoderErrorPreservesPending(t *testing.T) {
	tok := readTokenizer(t, "testdata/qwen.json")
	d := tok.NewDecoder()
	text, err := d.Decode([]int32{tok.bytes[0xe2], tok.bytes[0x82]}, false)
	if err != nil || text != "" || d.n != 2 {
		t.Fatalf("prefix: %q %v pending %d", text, err, d.n)
	}
	if _, err := d.Decode([]int32{-1}, true); err == nil {
		t.Fatal("accepted invalid ID")
	}
	text, err = d.Decode([]int32{tok.bytes[0xac]}, true)
	if err != nil || text != "€" {
		t.Fatalf("completion: %q %v", text, err)
	}
	text, err = d.Decode([]int32{tok.bytes['a']}, true)
	if err != nil || text != "a" {
		t.Fatalf("new sequence: %q %v", text, err)
	}
}

func TestDecoderFinalSpecialFlushes(t *testing.T) {
	tok := readTokenizer(t, "testdata/qwen.json")
	d := tok.NewDecoder()
	if _, err := d.Decode([]int32{tok.bytes[0xe2], tok.bytes[0x82]}, false); err != nil {
		t.Fatal(err)
	}
	var special int32 = -1
	for id, yes := range tok.special {
		if yes {
			special = id
			break
		}
	}
	if special < 0 {
		t.Fatal("fixture has no special token")
	}
	text, err := d.Decode([]int32{special}, true)
	if err != nil || text != "�" || d.n != 0 {
		t.Fatalf("flush: %q %v pending %d", text, err, d.n)
	}
}
