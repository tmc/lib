// Package dsv4tokenizer implements the host-side tokenizer and prompt
// renderer used by the DeepSeek V4 Flash example.
//
// The package deliberately has no model or backend dependency beyond GGUF
// metadata. It is suitable for front ends that need to prepare the exact
// token stream consumed by ds4-min.
package dsv4tokenizer

import (
	"errors"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/tmc/lib/gguf"
)

const (
	beginToken      = "<｜begin▁of▁sentence｜>"
	endToken        = "<｜end▁of▁sentence｜>"
	userToken       = "<｜User｜>"
	assistantToken  = "<｜Assistant｜>"
	thinkToken      = "<think>"
	thinkEndToken   = "</think>"
	maxThinkContext = 393216
)

// ThinkMode selects the reasoning marker appended to a chat prompt.
type ThinkMode uint8

const (
	ThinkNone ThinkMode = iota
	ThinkHigh
	ThinkMax
)

// Turn is one transcript turn accepted by RenderPrefix.
type Turn struct {
	Role    string
	Content string
}

// Tokenizer encodes, decodes, and renders a V4 tokenizer vocabulary.
type Tokenizer struct {
	tokens                                           []string
	ids                                              map[string]int
	merges                                           map[string]int
	bytes                                            [256]rune
	runes                                            map[rune]byte
	bos, eos, user, assistant, thinking, thinkingEnd int
	stops                                            map[int]bool
	dsml                                             int
}

var errInvalid = errors.New("invalid DS4 tokenizer")

// New constructs a tokenizer from a GGUF file's tokenizer metadata.
func New(g *gguf.File) (*Tokenizer, error) {
	if g == nil {
		return nil, fmt.Errorf("%w: nil GGUF", errInvalid)
	}
	model, ok := stringMetadata(g, "tokenizer.ggml.model")
	if !ok || model != "gpt2" {
		return nil, fmt.Errorf("%w: model %q", errInvalid, model)
	}
	pre, ok := stringMetadata(g, "tokenizer.ggml.pre")
	if !ok || pre != "joyai-llm" {
		return nil, fmt.Errorf("%w: pre-tokenizer %q", errInvalid, pre)
	}
	tokens, ok := stringArrayMetadata(g, "tokenizer.ggml.tokens")
	if !ok || len(tokens) != 129280 {
		return nil, fmt.Errorf("%w: token table", errInvalid)
	}
	merges, ok := stringArrayMetadata(g, "tokenizer.ggml.merges")
	if !ok || len(merges) != 127741 {
		return nil, fmt.Errorf("%w: merge table", errInvalid)
	}
	t := &Tokenizer{tokens: tokens, ids: make(map[string]int, len(tokens)), merges: make(map[string]int, len(merges)), runes: make(map[rune]byte, 256)}
	for i, s := range tokens {
		if _, exists := t.ids[s]; exists {
			return nil, fmt.Errorf("%w: duplicate token %q", errInvalid, s)
		}
		t.ids[s] = i
	}
	for i, s := range merges {
		if _, exists := t.merges[s]; exists {
			return nil, fmt.Errorf("%w: duplicate merge %q", errInvalid, s)
		}
		t.merges[s] = i
	}
	for i := range 256 {
		r := byteRune(byte(i))
		t.bytes[i] = r
		t.runes[r] = byte(i)
	}
	for _, x := range []struct {
		text string
		dst  *int
	}{{thinkToken, &t.thinking}, {beginToken, &t.bos}, {endToken, &t.eos}, {userToken, &t.user}, {assistantToken, &t.assistant}, {thinkEndToken, &t.thinkingEnd}} {
		id, exists := t.ids[x.text]
		if !exists {
			return nil, fmt.Errorf("%w: missing token %q", errInvalid, x.text)
		}
		*x.dst = id
	}
	if id, exists := t.ids["｜DSML｜"]; exists {
		t.dsml = id
	} else {
		t.dsml = -1
	}
	if t.bos != 0 || t.eos != 1 {
		return nil, fmt.Errorf("%w: BOS/EOS ids %d/%d", errInvalid, t.bos, t.eos)
	}
	t.stops = map[int]bool{t.eos: true}
	for _, key := range []string{"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"} {
		if v, ok := g.Metadata[key]; ok {
			id, ok := v.Uint()
			if !ok || id >= uint64(len(tokens)) {
				return nil, fmt.Errorf("%w: %s", errInvalid, key)
			}
			t.stops[int(id)] = true
		}
	}
	if types, ok := stringOrIntArray(g, "tokenizer.ggml.token_type"); ok {
		for id := range t.stops {
			if types[id] != 3 {
				return nil, fmt.Errorf("%w: stop token %d is not a control token", errInvalid, id)
			}
		}
	}
	return t, nil
}

func stringMetadata(g *gguf.File, key string) (string, bool) {
	v, ok := g.Metadata[key]
	if !ok || v.TypeID != gguf.TypeString {
		return "", false
	}
	s, ok := v.Value.(string)
	return s, ok
}
func stringArrayMetadata(g *gguf.File, key string) ([]string, bool) {
	v, ok := g.Metadata[key]
	if !ok || v.TypeID != gguf.TypeArray {
		return nil, false
	}
	a, ok := v.Value.([]any)
	if !ok {
		return nil, false
	}
	out := make([]string, len(a))
	for i, x := range a {
		out[i], ok = x.(string)
		if !ok {
			return nil, false
		}
	}
	return out, true
}
func stringOrIntArray(g *gguf.File, key string) ([]int, bool) {
	v, ok := g.Metadata[key]
	if !ok || v.TypeID != gguf.TypeArray {
		return nil, false
	}
	a, ok := v.Value.([]any)
	if !ok {
		return nil, false
	}
	out := make([]int, len(a))
	for i, x := range a {
		n, ok := gguf.Value{Value: x}.Int()
		if !ok {
			return nil, false
		}
		out[i] = int(n)
	}
	return out, true
}

// IsStop reports whether id is a generation stop token.
func (t *Tokenizer) IsStop(id int) bool { return t != nil && t.stops[id] }

// Encode converts UTF-8 text to token IDs.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if t == nil || !utf8.ValidString(text) {
		return nil, fmt.Errorf("%w: invalid UTF-8", errInvalid)
	}
	var ids []int
	for _, piece := range pieces(text) {
		runes := make([]string, 0, len([]byte(piece)))
		for _, b := range []byte(piece) {
			runes = append(runes, string(t.bytes[b]))
		}
		for {
			best, rank := -1, int(^uint(0)>>1)
			for i := 0; i+1 < len(runes); i++ {
				if r, ok := t.merges[runes[i]+" "+runes[i+1]]; ok && r < rank {
					best, rank = i, r
				}
			}
			if best < 0 {
				break
			}
			runes[best] += runes[best+1]
			runes = append(runes[:best+1], runes[best+2:]...)
		}
		for _, s := range runes {
			id, ok := t.ids[s]
			if !ok {
				return nil, fmt.Errorf("%w: no token for %q", errInvalid, s)
			}
			ids = append(ids, id)
		}
	}
	return ids, nil
}

// Decode converts token IDs to their byte representation.
func (t *Tokenizer) Decode(ids []int) ([]byte, error) {
	var out []byte
	for _, id := range ids {
		if t == nil || id < 0 || id >= len(t.tokens) {
			return nil, fmt.Errorf("%w: token id %d", errInvalid, id)
		}
		s := t.tokens[id]
		var b []byte
		ordinary := true
		for _, r := range s {
			x, ok := t.runes[r]
			if !ok {
				ordinary = false
				break
			}
			b = append(b, x)
		}
		if ordinary {
			out = append(out, b...)
		} else {
			out = append(out, s...)
		}
	}
	return out, nil
}

// RenderNonThinking renders one user turn without reasoning.
func (t *Tokenizer) RenderNonThinking(system, prompt string) ([]int, error) {
	return t.RenderChat(system, prompt, ThinkNone, 0)
}

// IsRenderedChat reports whether prompt begins with a rendered chat marker.
// The additional prefixes are accepted by upstream's CLI routing even when
// this V4 vocabulary does not assign them token IDs.
func IsRenderedChat(prompt string) bool {
	for _, prefix := range []string{
		beginToken, userToken, "[gMASK]", "<sop>", "<|system|>",
		"<|user|>", "<|assistant|>", "<|observation|>",
	} {
		if strings.HasPrefix(prompt, prefix) {
			return true
		}
	}
	return false
}

// EncodeRendered tokenizes already-rendered chat text. Unlike Encode, it
// recognizes active special-token spellings while BPE-encoding the spans
// between them. A marker followed immediately by ordinary text is split at
// the marker boundary.
func (t *Tokenizer) EncodeRendered(text string) ([]int, error) {
	if t == nil || !utf8.ValidString(text) {
		return nil, fmt.Errorf("%w: invalid UTF-8", errInvalid)
	}
	var ids []int
	span, p := 0, 0
	for p < len(text) {
		marker, id := t.renderedMarker(text[p:])
		if marker == "" {
			p++
			continue
		}
		part, err := t.Encode(text[span:p])
		if err != nil {
			return nil, err
		}
		ids = append(ids, part...)
		ids = append(ids, id)
		p += len(marker)
		span = p
	}
	part, err := t.Encode(text[span:])
	if err != nil {
		return nil, err
	}
	return append(ids, part...), nil
}

func (t *Tokenizer) renderedMarker(text string) (string, int) {
	markers := []struct {
		text string
		id   int
	}{
		{beginToken, t.bos},
		{endToken, t.eos},
		{userToken, t.user},
		{assistantToken, t.assistant},
		{thinkToken, t.thinking},
		{thinkEndToken, t.thinkingEnd},
	}
	if t.dsml >= 0 {
		markers = append(markers, struct {
			text string
			id   int
		}{"｜DSML｜", t.dsml})
	}
	for _, marker := range markers {
		if strings.HasPrefix(text, marker.text) {
			return marker.text, marker.id
		}
	}
	return "", 0
}

// RenderChat renders one V4 Flash user turn.
func (t *Tokenizer) RenderChat(system, prompt string, mode ThinkMode, context int) ([]int, error) {
	if IsRenderedChat(prompt) {
		return t.EncodeRendered(prompt)
	}
	if mode == ThinkMax && context > 0 && context < maxThinkContext {
		mode = ThinkHigh
	}
	ids := []int{t.bos}
	if mode == ThinkMax {
		p, err := t.Encode(maxThinkPrefix)
		if err != nil {
			return nil, err
		}
		ids = append(ids, p...)
	}
	if system != "" {
		p, err := t.Encode(system)
		if err != nil {
			return nil, err
		}
		ids = append(ids, p...)
	}
	ids = append(ids, t.user)
	p, err := t.Encode(prompt)
	if err != nil {
		return nil, err
	}
	ids = append(ids, p...)
	ids = append(ids, t.assistant)
	if mode == ThinkNone {
		ids = append(ids, t.thinkingEnd)
	} else {
		ids = append(ids, t.thinking)
	}
	return ids, nil
}

// RenderPrefix renders a validated USER/ASSISTANT transcript. The first turn
// must be USER and the last must be ASSISTANT.
func (t *Tokenizer) RenderPrefix(system string, turns []Turn, mode ThinkMode) ([]int, error) {
	if len(turns) == 0 || turns[0].Role != "USER" || turns[len(turns)-1].Role != "ASSISTANT" {
		return nil, fmt.Errorf("render prefix: turns must start with USER and end with ASSISTANT")
	}
	ids := []int{t.bos}
	if mode == ThinkMax {
		p, err := t.Encode(maxThinkPrefix)
		if err != nil {
			return nil, err
		}
		ids = append(ids, p...)
	}
	if system != "" {
		p, err := t.Encode(system)
		if err != nil {
			return nil, err
		}
		ids = append(ids, p...)
	}
	for _, turn := range turns {
		marker := t.user
		if turn.Role == "ASSISTANT" {
			marker = t.assistant
		}
		ids = append(ids, marker)
		if turn.Role == "ASSISTANT" && !strings.HasPrefix(turn.Content, "<think>") && !strings.HasPrefix(turn.Content, "</think>") {
			ids = append(ids, t.thinkingEnd)
		}
		p, err := t.Encode(turn.Content)
		if err != nil {
			return nil, err
		}
		ids = append(ids, p...)
		if turn.Role == "ASSISTANT" {
			ids = append(ids, t.eos)
		}
	}
	return ids, nil
}

const maxThinkPrefix = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"

func byteRune(b byte) rune {
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
	return 0
}
func pieces(s string) []string {
	type ch struct {
		start, end int
		r          rune
	}
	cs := make([]ch, 0, utf8.RuneCountInString(s))
	for start, r := range s {
		_, n := utf8.DecodeRuneInString(s[start:])
		cs = append(cs, ch{start, start + n, r})
	}
	var out []string
	for i := 0; i < len(cs); {
		start, j := i, i
		switch {
		case unicode.IsNumber(cs[i].r):
			for j < len(cs) && j-start < 3 && unicode.IsNumber(cs[j].r) {
				j++
			}
		case isCJK(cs[i].r):
			for j < len(cs) && isCJK(cs[j].r) {
				j++
			}
		case isPunct(cs[i].r) && i+1 < len(cs) && unicode.IsLetter(cs[i+1].r):
			j = i + 2
			for j < len(cs) && unicode.IsLetter(cs[j].r) {
				j++
			}
		case isLetter(cs[i].r) || i+1 < len(cs) && isPrefix(cs[i].r) && isLetter(cs[i+1].r):
			if !isLetter(cs[i].r) {
				j++
			}
			for j < len(cs) && isLetter(cs[j].r) {
				j++
			}
		case cs[i].r == ' ' && i+1 < len(cs) && isPunct(cs[i+1].r):
			j = i + 1
			for j < len(cs) && isPunct(cs[j].r) {
				j++
			}
			for j < len(cs) && isNL(cs[j].r) {
				j++
			}
		case isPunct(cs[i].r):
			for j < len(cs) && isPunct(cs[j].r) {
				if j > i && cs[j].r == '>' && j+1 < len(cs) && unicode.IsLetter(cs[j+1].r) {
					break
				}
				j++
			}
			for j < len(cs) && isNL(cs[j].r) {
				j++
			}
		case unicode.IsSpace(cs[i].r):
			j = i
			last := -1
			for j < len(cs) && unicode.IsSpace(cs[j].r) {
				if isNL(cs[j].r) {
					last = j + 1
				}
				j++
			}
			if last >= 0 {
				j = last
			} else if j > i+1 {
				j--
			}
		default:
			j = i + 1
		}
		if j <= i {
			j = i + 1
		}
		out = append(out, s[cs[start].start:cs[j-1].end])
		i = j
	}
	return out
}
func isCJK(r rune) bool    { return r >= 0x4e00 && r <= 0x9fa5 || r >= 0x3040 && r <= 0x30ff }
func isPunct(r rune) bool  { return unicode.IsPunct(r) || unicode.IsSymbol(r) }
func isLetter(r rune) bool { return unicode.IsLetter(r) || unicode.IsMark(r) }
func isPrefix(r rune) bool { return !isNL(r) && !unicode.IsLetter(r) && !isPunct(r) }
func isNL(r rune) bool     { return r == '\r' || r == '\n' }
