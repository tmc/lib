package hftokenizer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
)

type encodeCase struct {
	Text    string
	IDs     []int32
	Decoded string
}
type decodeCase struct {
	IDs     []int32
	Decoded string
	Bytes   []byte
}

func readTokenizer(t *testing.T, path string) *Tokenizer {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tok, err := New(f)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func checkEncoding(t *testing.T, tok *Tokenizer, cases []encodeCase) {
	t.Helper()
	for i, c := range cases {
		ids, err := tok.Encode(c.Text)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if !slices.Equal(ids, c.IDs) {
			t.Fatalf("case %d %q: IDs %v, want %v", i, c.Text, ids, c.IDs)
		}
		text, err := tok.Decode(ids)
		if err != nil || text != c.Decoded {
			t.Fatalf("case %d decode: %q, %v; want %q", i, text, err, c.Decoded)
		}
	}
}

func TestReference(t *testing.T) {
	data, err := os.ReadFile("testdata/reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var reference []struct {
		File     string
		Cases    []encodeCase
		Decoding []decodeCase
	}
	if err := json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	for _, r := range reference {
		t.Run(r.File, func(t *testing.T) {
			tok := readTokenizer(t, filepath.Join("testdata", r.File))
			checkEncoding(t, tok, r.Cases)
			for i, c := range r.Decoding {
				text, err := tok.Decode(c.IDs)
				if err != nil || text != c.Decoded {
					t.Fatalf("decode %d: %q, %v; want %q", i, text, err, c.Decoded)
				}
				raw, err := tok.DecodeBytes(c.IDs)
				if err != nil || !bytes.Equal(raw, c.Bytes) {
					t.Fatalf("decode bytes %d: %x, %v; want %x", i, raw, err, c.Bytes)
				}
			}
		})
	}
}

func TestQwenReference(t *testing.T) {
	dir := os.Getenv("HF_TOKENIZER_REFERENCE_DIR")
	if dir == "" {
		t.Skip("set HF_TOKENIZER_REFERENCE_DIR to pinned Qwen tokenizer and reference files")
	}
	tok := readTokenizer(t, filepath.Join(dir, "tokenizer.json"))
	data, err := os.ReadFile(filepath.Join(dir, "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []encodeCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("empty tokenizer oracle")
	}
	checkEncoding(t, tok, cases)
}

func TestConcurrentEncode(t *testing.T) {
	tok := readTokenizer(t, "testdata/llama.json")
	want, err := tok.Encode("abc café 😀 <tag>")
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				got, err := tok.Encode("abc café 😀 <tag>")
				if err != nil || !slices.Equal(got, want) {
					t.Errorf("Encode: %v, %v", got, err)
				}
				if _, err := tok.Decode(got); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Wait()
}

func TestRejectUnsupported(t *testing.T) {
	data, err := os.ReadFile("testdata/qwen.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name   string
		change func(map[string]any)
	}{
		{"normalizer", func(s map[string]any) { s["normalizer"] = map[string]any{"type": "NFKC"} }},
		{"padding", func(s map[string]any) { s["padding"] = map[string]any{} }},
		{"unknown option", func(s map[string]any) { s["unknown"] = true }},
		{"dropout", func(s map[string]any) { s["model"].(map[string]any)["dropout"] = 0.1 }},
		{"fallback", func(s map[string]any) { s["model"].(map[string]any)["byte_fallback"] = true }},
		{"missing byte", func(s map[string]any) { delete(s["model"].(map[string]any)["vocab"].(map[string]any), "a") }},
		{"ID collision", func(s map[string]any) { v := s["model"].(map[string]any)["vocab"].(map[string]any); v["b"] = v["a"] }},
		{"normalized added", func(s map[string]any) { s["added_tokens"].([]any)[0].(map[string]any)["normalized"] = true }},
		{"strip added", func(s map[string]any) { s["added_tokens"].([]any)[0].(map[string]any)["lstrip"] = true }},
		{"decoder", func(s map[string]any) { s["decoder"] = map[string]any{"type": "Metaspace"} }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var s map[string]any
			if err := json.Unmarshal(data, &s); err != nil {
				t.Fatal(err)
			}
			tt.change(s)
			b, err := json.Marshal(s)
			if err != nil {
				t.Fatal(err)
			}
			if tok, err := New(bytes.NewReader(b)); err == nil || tok != nil {
				t.Fatalf("accepted unsupported pipeline: %v", err)
			}
		})
	}
}

func TestInputErrors(t *testing.T) {
	tok := readTokenizer(t, "testdata/qwen.json")
	if _, err := tok.Encode(string([]byte{0xff})); err == nil {
		t.Fatal("accepted invalid UTF-8")
	}
	for _, id := range []int32{-1, tok.MaxTokenID() + 1} {
		if _, err := tok.Decode([]int32{id}); err == nil {
			t.Fatalf("accepted unknown ID %d", id)
		}
	}
	if _, err := New(strings.NewReader("{} {}")); err == nil {
		t.Fatal("accepted trailing JSON")
	}
	var zero Tokenizer
	if _, err := zero.Encode("abc"); err == nil {
		t.Fatal("accepted zero tokenizer")
	}
	if _, err := zero.Decode(nil); err == nil {
		t.Fatal("decoded with zero tokenizer")
	}
}

func TestLossyUTF8(t *testing.T) {
	for _, tt := range []struct {
		bytes []byte
		want  string
	}{
		{[]byte{0xff, 0xff}, "��"},
		{[]byte{0xef, 0xbb}, "�"},
		{[]byte{0xe0, 0x80, 0x80}, "���"},
		{[]byte{0xf0, 0x9f, 0x98, 0x80}, "😀"},
	} {
		t.Run(fmt.Sprintf("%x", tt.bytes), func(t *testing.T) {
			if got := lossyUTF8(tt.bytes); got != tt.want {
				t.Fatalf("%q, want %q", got, tt.want)
			}
		})
	}
}

func TestSplitterReference(t *testing.T) {
	path := os.Getenv("HF_SPLITTER_REFERENCE")
	if path == "" {
		t.Skip("set HF_SPLITTER_REFERENCE to the Unicode split oracle")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var reference []struct {
		Name  string
		Cases []struct {
			Text  string
			Parts []string
		}
	}
	if err := json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	for _, r := range reference {
		t.Run(r.Name, func(t *testing.T) {
			kind, ok := map[string]splitKind{"gpt2": splitGPT2, "qwen": splitQwen, "llama": splitLlama}[r.Name]
			if !ok || len(r.Cases) == 0 {
				t.Fatal("invalid split oracle")
			}
			for i, c := range r.Cases {
				if got := splitText(c.Text, kind); !slices.Equal(got, c.Parts) {
					t.Fatalf("case %d %q: split %q, want %q", i, c.Text, got, c.Parts)
				}
			}
		})
	}
}

func TestNormalizationReference(t *testing.T) {
	dir := os.Getenv("HF_NORMALIZATION_REFERENCE_DIR")
	if dir == "" {
		t.Skip("set HF_NORMALIZATION_REFERENCE_DIR to the full scalar normalization oracle")
	}
	input, err := os.ReadFile(filepath.Join(dir, "normalization-input.txt"))
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(filepath.Join(dir, "normalization-output.txt"))
	if err != nil {
		t.Fatal(err)
	}
	got := normalizeNFC(string(input))
	if got != string(want) {
		a, b := []rune(got), []rune(string(want))
		for i := 0; i < len(a) && i < len(b); i++ {
			if a[i] != b[i] {
				t.Fatalf("normalization rune %d = U+%04X, want U+%04X", i, a[i], b[i])
			}
		}
		t.Fatalf("normalization lengths differ: %d, want %d", len(a), len(b))
	}
}

func TestPinnedUnicode(t *testing.T) {
	for _, tt := range []struct{ text, want string }{
		{"cafe\u0301", "café"},
		{"\U00010041\u0301", "\U00010041\u0301"},
		{"\u1df6\u0301", "\u1df6\u0301"},
		{"\u1100\u1161\u11a8", "각"},
		{"a" + strings.Repeat("\u0301", 35), "á" + strings.Repeat("\u0301", 34)},
	} {
		if got := normalizeNFC(tt.text); got != tt.want {
			t.Errorf("NFC(%q) = %q, want %q", tt.text, got, tt.want)
		}
	}
	if !isLetter('\U00013f01') {
		t.Error("missing Unicode 16 letter")
	}
	if isLetter('\U00032b6c') {
		t.Error("accepted a Unicode 17 letter")
	}
}

func TestDuplicateJSONKeys(t *testing.T) {
	data, err := os.ReadFile("testdata/gpt2.json")
	if err != nil {
		t.Fatal(err)
	}
	input := strings.Replace(string(data), `"version": "1.0"`, `"version": "1.0", "version": "1.0"`, 1)
	if input == string(data) {
		t.Fatal("duplicate-key fixture did not apply")
	}
	if _, err := New(strings.NewReader(input)); err == nil || !strings.Contains(err.Error(), "duplicate key") {
		t.Fatalf("duplicate keys: %v", err)
	}
}

func TestNormalizationConformance(t *testing.T) {
	path := os.Getenv("HF_NORMALIZATION_TEST")
	if path == "" {
		t.Skip("set HF_NORMALIZATION_TEST to Unicode 9 NormalizationTest.txt")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for lineNo, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(strings.SplitN(line, "#", 2)[0])
		if line == "" || strings.HasPrefix(line, "@") {
			continue
		}
		fields := strings.Split(line, ";")
		if len(fields) < 5 {
			t.Fatalf("line %d: malformed conformance data", lineNo+1)
		}
		var text [5]string
		for i, field := range fields[:5] {
			for _, hex := range strings.Fields(field) {
				var r rune
				if _, err := fmt.Sscanf(hex, "%X", &r); err != nil {
					t.Fatal(err)
				}
				text[i] += string(r)
			}
		}
		for i, wantIndex := range []int{1, 1, 1, 3, 3} {
			if got := normalizeNFC(text[i]); got != text[wantIndex] {
				t.Fatalf("line %d column %d NFC = %U, want %U", lineNo+1, i+1, []rune(got), []rune(text[wantIndex]))
			}
		}
		count++
	}
	if count == 0 {
		t.Fatal("empty conformance corpus")
	}
	t.Logf("checked %d NFC conformance rows", count)
}

func TestJSONSurrogates(t *testing.T) {
	data, err := os.ReadFile("testdata/gpt2.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		content string
		valid   bool
	}{
		{`\ud800`, false},
		{`\udc00`, false},
		{`\ud800\u0041`, false},
		{`\ud83d\ude00`, true},
		{`\\ud800`, true},
	} {
		t.Run(tt.content, func(t *testing.T) {
			input := strings.Replace(string(data), `"content": "<BOS>"`, `"content": "`+tt.content+`"`, 1)
			if input == string(data) {
				t.Fatal("surrogate fixture did not apply")
			}
			_, err := New(strings.NewReader(input))
			if (err == nil) != tt.valid {
				t.Fatalf("valid=%v, want %v; error=%v", err == nil, tt.valid, err)
			}
		})
	}
}

func BenchmarkEncode(b *testing.B) {
	data, err := os.ReadFile("testdata/qwen.json")
	if err != nil {
		b.Fatal(err)
	}
	tok, err := New(bytes.NewReader(data))
	if err != nil {
		b.Fatal(err)
	}
	for _, tt := range []struct{ name, text string }{
		{"short", "abc 123 café 😀\n"},
		{"long", strings.Repeat("abc 123 café 😀\n", 256)},
	} {
		b.Run(tt.name, func(b *testing.B) {
			b.SetBytes(int64(len(tt.text)))
			b.ReportAllocs()
			for b.Loop() {
				ids, err := tok.Encode(tt.text)
				if err != nil || len(ids) == 0 {
					b.Fatalf("Encode = %v, %v", ids, err)
				}
			}
		})
	}
}
