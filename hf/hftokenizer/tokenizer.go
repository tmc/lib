package hftokenizer

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode/utf8"
)

// Tokenizer is an immutable byte-level BPE pipeline. Construct it with New;
// its zero value is not usable. Methods may be called concurrently.
type Tokenizer struct {
	bytes               [256]int32
	merges              map[pair]merge
	tokens              map[int32][]byte
	special             map[int32]bool
	added               []addedToken
	prefix, suffix      []int32
	split               splitKind
	nfc, addPrefixSpace bool
	maxID               int32
}

type addedToken struct {
	text string
	id   int32
}

// New reads a Hugging Face tokenizer.json and validates its supported pipeline.
// It retains vocabulary data, not the reader. Unsupported options return errors.
func New(r io.Reader) (*Tokenizer, error) {
	if r == nil {
		return nil, fmt.Errorf("tokenizer: nil reader")
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read: %w", err)
	}
	if !utf8.Valid(data) {
		return nil, fmt.Errorf("tokenizer: JSON is not valid UTF-8")
	}
	return parse(data)
}

// Encode returns IDs after added-token matching, normalization, BPE and the
// configured single-input post-processor. It never inserts an inferred BOS.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	if t == nil || t.tokens == nil {
		return nil, fmt.Errorf("tokenizer: uninitialized tokenizer")
	}
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("tokenizer: input is not valid UTF-8")
	}
	ids := append([]int32(nil), t.prefix...)
	for text != "" {
		pos := -1
		var match addedToken
		for _, a := range t.added {
			i := strings.Index(text, a.text)
			if i >= 0 && (pos < 0 || i < pos) {
				pos = i
				match = a
			}
		}
		if pos < 0 {
			ids = t.encodeChunk(text, ids)
			break
		}
		ids = t.encodeChunk(text[:pos], ids)
		ids = append(ids, match.id)
		text = text[pos+len(match.text):]
	}
	return append(ids, t.suffix...), nil
}

func (t *Tokenizer) encodeChunk(s string, ids []int32) []int32 {
	if s == "" {
		return ids
	}
	if t.nfc {
		s = normalizeNFC(s)
	}
	if t.addPrefixSpace && s[0] != ' ' {
		s = " " + s
	}
	for _, piece := range splitText(s, t.split) {
		ids = t.encodePiece(piece, ids)
	}
	return ids
}

// DecodeBytes returns the decoded bytes, omitting added tokens marked special.
// Bytes may contain incomplete UTF-8; concatenate adjacent token bytes before
// displaying them. An unknown token ID returns an error.
func (t *Tokenizer) DecodeBytes(ids []int32) ([]byte, error) {
	if t == nil || t.tokens == nil {
		return nil, fmt.Errorf("tokenizer: uninitialized tokenizer")
	}
	var out []byte
	for _, id := range ids {
		b, ok := t.tokens[id]
		if !ok {
			return nil, fmt.Errorf("tokenizer: unknown token ID %d", id)
		}
		if !t.special[id] {
			out = append(out, b...)
		}
	}
	return out, nil
}

// Decode returns decoded text, omitting added tokens marked special and replacing
// each invalid UTF-8 subsequence with U+FFFD, as the ByteLevel decoder does.
func (t *Tokenizer) Decode(ids []int32) (string, error) {
	b, err := t.DecodeBytes(ids)
	if err != nil {
		return "", err
	}
	return lossyUTF8(b), nil
}

// MaxTokenID returns the largest declared token ID, including added tokens.
// IDs need not be contiguous; an embedding table must have more rows than this ID.
func (t *Tokenizer) MaxTokenID() int32 { return t.maxID }

func byteAlphabet() ([256]rune, map[rune]byte) {
	var forward [256]rune
	reverse := make(map[rune]byte, 256)
	next := rune(256)
	for i := 0; i < 256; i++ {
		r := rune(i)
		if !(i >= 33 && i <= 126 || i >= 161 && i <= 172 || i >= 174) {
			r = next
			next++
		}
		forward[i] = r
		reverse[r] = byte(i)
	}
	return forward, reverse
}

func decodePiece(s string, alphabet map[rune]byte) []byte {
	var out []byte
	for _, r := range s {
		b, ok := alphabet[r]
		if !ok {
			return []byte(s)
		}
		out = append(out, b)
	}
	return out
}

func lossyUTF8(b []byte) string {
	var out strings.Builder
	for len(b) > 0 {
		r, n := utf8.DecodeRune(b)
		if r != utf8.RuneError || n > 1 {
			out.Write(b[:n])
			b = b[n:]
			continue
		}
		n = 1
		want := 0
		if b[0] >= 0xc2 && b[0] <= 0xdf {
			want = 2
		} else if b[0] >= 0xe0 && b[0] <= 0xef {
			want = 3
		} else if b[0] >= 0xf0 && b[0] <= 0xf4 {
			want = 4
		}
		for n < want && n < len(b) {
			c := b[n]
			if c < 0x80 || c > 0xbf || n == 1 && (b[0] == 0xe0 && c < 0xa0 || b[0] == 0xed && c > 0x9f || b[0] == 0xf0 && c < 0x90 || b[0] == 0xf4 && c > 0x8f) {
				break
			}
			n++
		}
		out.WriteRune(utf8.RuneError)
		b = b[n:]
	}
	return out.String()
}

func sortAdded(tokens []addedToken) {
	sort.Slice(tokens, func(i, j int) bool {
		if len(tokens[i].text) != len(tokens[j].text) {
			return len(tokens[i].text) > len(tokens[j].text)
		}
		return tokens[i].text < tokens[j].text
	})
}
