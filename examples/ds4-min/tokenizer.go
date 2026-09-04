package main

import (
	"errors"
	"fmt"
	"unicode"
	"unicode/utf8"
)

const (
	beginTokenText     = "<｜begin▁of▁sentence｜>"
	endTokenText       = "<｜end▁of▁sentence｜>"
	userTokenText      = "<｜User｜>"
	assistantTokenText = "<｜Assistant｜>"
	thinkEndTokenText  = "</think>"
)

var errInvalidTokenizer = errors.New("invalid DS4 tokenizer")

type tokenizer struct {
	tokens      []string
	tokenID     map[string]int
	mergeRank   map[string]int
	byteRunes   [256]rune
	runeBytes   map[rune]byte
	bos         int
	eos         int
	user        int
	assistant   int
	thinkingEnd int
}

func newTokenizer(g *ggufFile) (*tokenizer, error) {
	model, err := metadataString(g, "tokenizer.ggml.model")
	if err != nil || model != "gpt2" {
		return nil, fmt.Errorf("%w: model %q", errInvalidTokenizer, model)
	}
	pre, err := metadataString(g, "tokenizer.ggml.pre")
	if err != nil || pre != "joyai-llm" {
		return nil, fmt.Errorf("%w: pre-tokenizer %q", errInvalidTokenizer, pre)
	}
	tokens, err := metadataStrings(g, "tokenizer.ggml.tokens")
	if err != nil || len(tokens) != 129280 {
		return nil, fmt.Errorf("%w: token table", errInvalidTokenizer)
	}
	merges, err := metadataStrings(g, "tokenizer.ggml.merges")
	if err != nil || len(merges) != 127741 {
		return nil, fmt.Errorf("%w: merge table", errInvalidTokenizer)
	}
	t := &tokenizer{
		tokens:    tokens,
		tokenID:   make(map[string]int, len(tokens)),
		mergeRank: make(map[string]int, len(merges)),
		runeBytes: make(map[rune]byte, 256),
	}
	for id, token := range tokens {
		if _, exists := t.tokenID[token]; exists {
			return nil, fmt.Errorf("%w: duplicate token %q", errInvalidTokenizer, token)
		}
		t.tokenID[token] = id
	}
	for rank, merge := range merges {
		if _, exists := t.mergeRank[merge]; exists {
			return nil, fmt.Errorf("%w: duplicate merge %q", errInvalidTokenizer, merge)
		}
		t.mergeRank[merge] = rank
	}
	for i := range 256 {
		r := gpt2ByteRune(byte(i))
		t.byteRunes[i] = r
		t.runeBytes[r] = byte(i)
	}
	for _, special := range []struct {
		text string
		id   *int
	}{
		{beginTokenText, &t.bos},
		{endTokenText, &t.eos},
		{userTokenText, &t.user},
		{assistantTokenText, &t.assistant},
		{thinkEndTokenText, &t.thinkingEnd},
	} {
		id, ok := t.tokenID[special.text]
		if !ok {
			return nil, fmt.Errorf("%w: missing token %q", errInvalidTokenizer, special.text)
		}
		*special.id = id
	}
	if t.bos != 0 || t.eos != 1 {
		return nil, fmt.Errorf("%w: BOS/EOS ids %d/%d", errInvalidTokenizer, t.bos, t.eos)
	}
	return t, nil
}

func metadataString(g *ggufFile, key string) (string, error) {
	v, ok := g.metadata[key]
	if !ok || v.typeID != 8 {
		return "", fmt.Errorf("%w: missing string %q", errInvalidTokenizer, key)
	}
	s, ok := v.value.(string)
	if !ok {
		return "", fmt.Errorf("%w: string %q", errInvalidTokenizer, key)
	}
	return s, nil
}

func metadataStrings(g *ggufFile, key string) ([]string, error) {
	v, ok := g.metadata[key]
	if !ok || v.typeID != 9 {
		return nil, fmt.Errorf("%w: missing array %q", errInvalidTokenizer, key)
	}
	values, ok := v.value.([]any)
	if !ok {
		return nil, fmt.Errorf("%w: array %q", errInvalidTokenizer, key)
	}
	out := make([]string, len(values))
	for i, value := range values {
		var ok bool
		out[i], ok = value.(string)
		if !ok {
			return nil, fmt.Errorf("%w: %q element %d", errInvalidTokenizer, key, i)
		}
	}
	return out, nil
}

func (t *tokenizer) Encode(text string) ([]int, error) {
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("%w: input is not UTF-8", errInvalidTokenizer)
	}
	pieces := joyAIPieces(text)
	var ids []int
	for _, piece := range pieces {
		encoded := make([]rune, len([]byte(piece)))
		for i, b := range []byte(piece) {
			encoded[i] = t.byteRunes[b]
		}
		symbols := make([]string, len(encoded))
		for i, r := range encoded {
			symbols[i] = string(r)
		}
		for {
			best, rank := -1, int(^uint(0)>>1)
			for i := 0; i+1 < len(symbols); i++ {
				if r, ok := t.mergeRank[symbols[i]+" "+symbols[i+1]]; ok && r < rank {
					best, rank = i, r
				}
			}
			if best < 0 {
				break
			}
			symbols[best] += symbols[best+1]
			symbols = append(symbols[:best+1], symbols[best+2:]...)
		}
		for _, symbol := range symbols {
			id, ok := t.tokenID[symbol]
			if !ok {
				return nil, fmt.Errorf("%w: no token for %q", errInvalidTokenizer, symbol)
			}
			ids = append(ids, id)
		}
	}
	return ids, nil
}

func (t *tokenizer) Decode(ids []int) ([]byte, error) {
	var out []byte
	for _, id := range ids {
		if id < 0 || id >= len(t.tokens) {
			return nil, fmt.Errorf("%w: token id %d", errInvalidTokenizer, id)
		}
		token := t.tokens[id]
		decoded := make([]byte, 0, len(token))
		ordinary := true
		for _, r := range token {
			b, ok := t.runeBytes[r]
			if !ok {
				ordinary = false
				break
			}
			decoded = append(decoded, b)
		}
		if ordinary {
			out = append(out, decoded...)
		} else {
			out = append(out, token...)
		}
	}
	return out, nil
}

func (t *tokenizer) RenderNonThinking(system, prompt string) ([]int, error) {
	ids := []int{t.bos}
	if system != "" {
		part, err := t.Encode(system)
		if err != nil {
			return nil, err
		}
		ids = append(ids, part...)
	}
	ids = append(ids, t.user)
	part, err := t.Encode(prompt)
	if err != nil {
		return nil, err
	}
	ids = append(ids, part...)
	ids = append(ids, t.assistant, t.thinkingEnd)
	return ids, nil
}

func gpt2ByteRune(b byte) rune {
	if b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174 {
		return rune(b)
	}
	n := 0
	for x := 0; x < 256; x++ {
		if x >= 33 && x <= 126 || x >= 161 && x <= 172 || x >= 174 {
			continue
		}
		if byte(x) == b {
			return rune(256 + n)
		}
		n++
	}
	panic("unreachable")
}

func joyAIPieces(text string) []string {
	type char struct {
		start, end int
		r          rune
	}
	chars := make([]char, 0, utf8.RuneCountInString(text))
	for start, r := range text {
		_, n := utf8.DecodeRuneInString(text[start:])
		chars = append(chars, char{start, start + n, r})
	}
	var pieces []string
	for i := 0; i < len(chars); {
		start, j := i, i
		switch {
		case unicode.IsNumber(chars[i].r):
			for j < len(chars) && j-start < 3 && unicode.IsNumber(chars[j].r) {
				j++
			}
		case isCJK(chars[i].r):
			for j < len(chars) && isCJK(chars[j].r) {
				j++
			}
		case isPunctSymbol(chars[i].r) && i+1 < len(chars) && isASCIIAlpha(chars[i+1].r):
			j = i + 2
			for j < len(chars) && isASCIIAlpha(chars[j].r) {
				j++
			}
		case isLetterMark(chars[i].r) || i+1 < len(chars) && isLetterPrefix(chars[i].r) && isLetterMark(chars[i+1].r):
			if !isLetterMark(chars[i].r) {
				j++
			}
			for j < len(chars) && isLetterMark(chars[j].r) {
				j++
			}
		case chars[i].r == ' ' && i+1 < len(chars) && isPunctSymbol(chars[i+1].r):
			j = i + 1
			for j < len(chars) && isPunctSymbol(chars[j].r) {
				j++
			}
			for j < len(chars) && isNewline(chars[j].r) {
				j++
			}
		case isPunctSymbol(chars[i].r):
			for j < len(chars) && isPunctSymbol(chars[j].r) {
				j++
			}
			for j < len(chars) && isNewline(chars[j].r) {
				j++
			}
		case unicode.IsSpace(chars[i].r):
			j = i
			lastNewline := -1
			for j < len(chars) && unicode.IsSpace(chars[j].r) {
				if isNewline(chars[j].r) {
					lastNewline = j + 1
				}
				j++
			}
			if lastNewline >= 0 {
				j = lastNewline
			} else if j < len(chars) && j > i+1 {
				j--
			}
		default:
			j = i + 1
		}
		if j <= i {
			j = i + 1
		}
		pieces = append(pieces, text[chars[start].start:chars[j-1].end])
		i = j
	}
	return pieces
}

func isCJK(r rune) bool          { return r >= 0x4e00 && r <= 0x9fa5 || r >= 0x3040 && r <= 0x30ff }
func isPunctSymbol(r rune) bool  { return unicode.IsPunct(r) || unicode.IsSymbol(r) }
func isLetterMark(r rune) bool   { return unicode.IsLetter(r) || unicode.IsMark(r) }
func isLetterPrefix(r rune) bool { return !isNewline(r) && !unicode.IsLetter(r) && !isPunctSymbol(r) }
func isASCIIAlpha(r rune) bool   { return r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' }
func isNewline(r rune) bool      { return r == '\r' || r == '\n' }
