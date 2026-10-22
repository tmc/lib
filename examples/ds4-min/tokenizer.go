package main

import (
	"errors"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/tmc/lib/gguf"
)

const (
	beginTokenText     = "<｜begin▁of▁sentence｜>"
	endTokenText       = "<｜end▁of▁sentence｜>"
	userTokenText      = "<｜User｜>"
	assistantTokenText = "<｜Assistant｜>"
	thinkTokenText     = "<think>"
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
	thinking    int
	dsml        int
	dsmlSet     bool
	stops       stopSet
}

// A stopSet is the set of token ids that end generation. It is derived from
// tokenizer metadata, never from a hard-coded id.
type stopSet map[int]bool

// Contains reports whether id ends generation.
func (s stopSet) Contains(id int) bool { return s[id] }

// tokenTypeControl is the GGUF tokenizer.ggml.token_type value for a control
// token. Every stop token must carry it.
const tokenTypeControl = 3

// stopMetadataKeys are the optional metadata ids that name a stop token.
var stopMetadataKeys = []string{
	"tokenizer.ggml.eos_token_id",
	"tokenizer.ggml.eot_token_id",
	"tokenizer.ggml.eom_token_id",
}

func newTokenizer(g *gguf.File) (*tokenizer, error) {
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
		dsml:      -1,
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
		{thinkTokenText, &t.thinking},
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
	if id, ok := t.tokenID["｜DSML｜"]; ok {
		t.dsml = id
		t.dsmlSet = true
	}
	if t.bos != 0 || t.eos != 1 {
		return nil, fmt.Errorf("%w: BOS/EOS ids %d/%d", errInvalidTokenizer, t.bos, t.eos)
	}
	t.stops, err = t.generationStops(g)
	if err != nil {
		return nil, err
	}
	return t, nil
}

// generationStops derives the stop-token set. The vocabulary end-of-sentence
// token is always a stop; each optional end-of-sequence, end-of-turn, and
// end-of-message metadata id joins it. Every stop must name a control token in
// tokenizer.ggml.token_type.
func (t *tokenizer) generationStops(g *gguf.File) (stopSet, error) {
	stops := stopSet{t.eos: true}
	for _, key := range stopMetadataKeys {
		v, ok := g.Metadata[key]
		if !ok {
			continue
		}
		id, ok := v.Uint()
		if !ok || id >= uint64(len(t.tokens)) {
			return nil, fmt.Errorf("%w: %s", errInvalidTokenizer, key)
		}
		stops[int(id)] = true
	}
	types, err := tokenTypes(g, len(t.tokens))
	if err != nil {
		return nil, err
	}
	for id := range stops {
		if types != nil && types[id] != tokenTypeControl {
			return nil, fmt.Errorf("%w: stop token %d is not a control token", errInvalidTokenizer, id)
		}
	}
	return stops, nil
}

// IsStop reports whether id ends generation.
func (t *tokenizer) IsStop(id int) bool { return t.stops.Contains(id) }

// IsStopForMode reports whether id ends generation under the selected
// thinking mode. In non-thinking mode the reasoning boundary tokens are
// stops too; feeding either marker back into the model crosses the mode
// boundary and diverges from the upstream CLI.
func (t *tokenizer) IsStopForMode(id int, mode ThinkMode) bool {
	if t == nil {
		return false
	}
	if t.stops.Contains(id) {
		return true
	}
	return mode == ThinkNone && (id == t.thinking || id == t.thinkingEnd)
}

// tokenTypes returns the tokenizer.ggml.token_type array, or nil when the
// model omits it.
func tokenTypes(g *gguf.File, tokens int) ([]int, error) {
	v, ok := g.Metadata["tokenizer.ggml.token_type"]
	if !ok {
		return nil, nil
	}
	values, ok := v.Value.([]any)
	if !ok || v.TypeID != gguf.TypeArray || len(values) != tokens {
		return nil, fmt.Errorf("%w: token type array", errInvalidTokenizer)
	}
	out := make([]int, len(values))
	for i, value := range values {
		n, ok := gguf.Value{Value: value}.Int()
		if !ok {
			return nil, fmt.Errorf("%w: token type element %d", errInvalidTokenizer, i)
		}
		out[i] = int(n)
	}
	return out, nil
}

func metadataString(g *gguf.File, key string) (string, error) {
	v, ok := g.Metadata[key]
	if !ok || v.TypeID != gguf.TypeString {
		return "", fmt.Errorf("%w: missing string %q", errInvalidTokenizer, key)
	}
	s, ok := v.Value.(string)
	if !ok {
		return "", fmt.Errorf("%w: string %q", errInvalidTokenizer, key)
	}
	return s, nil
}

func metadataStrings(g *gguf.File, key string) ([]string, error) {
	v, ok := g.Metadata[key]
	if !ok || v.TypeID != gguf.TypeArray {
		return nil, fmt.Errorf("%w: missing array %q", errInvalidTokenizer, key)
	}
	values, ok := v.Value.([]any)
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
	return t.RenderChat(system, prompt, ThinkNone, 0)
}

// ThinkMode controls the reasoning marker appended to a chat prompt.
type ThinkMode uint8

const (
	ThinkNone ThinkMode = iota
	ThinkHigh
	ThinkMax
)

const maxThinkContext = 393216

// RenderChat renders one V4 Flash user turn. ThinkMax is downgraded to
// ThinkHigh below the upstream minimum context size. A zero context disables
// that downgrade, which is useful for token-only tests.
func (t *tokenizer) RenderChat(system, prompt string, mode ThinkMode, context int) ([]int, error) {
	if isRenderedChatPrompt(prompt) {
		return t.encodeRendered(prompt)
	}
	if mode == ThinkMax && context > 0 && context < maxThinkContext {
		mode = ThinkHigh
	}
	ids := []int{t.bos}
	if mode == ThinkMax {
		part, err := t.Encode(maxThinkPrefix)
		if err != nil {
			return nil, err
		}
		ids = append(ids, part...)
	}
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
	ids = append(ids, t.assistant)
	if mode == ThinkNone {
		ids = append(ids, t.thinkingEnd)
	} else {
		ids = append(ids, t.thinking)
	}
	return ids, nil
}

// isRenderedChatPrompt mirrors the upstream CLI's prompt routing. A rendered
// prompt is tokenized as-is; it must not be wrapped in another chat turn.
func isRenderedChatPrompt(prompt string) bool {
	for _, prefix := range []string{beginTokenText, userTokenText, "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"} {
		if strings.HasPrefix(prompt, prefix) {
			return true
		}
	}
	return false
}

func (t *tokenizer) encodeRendered(text string) ([]int, error) {
	markers := []struct {
		text string
		id   int
	}{
		{beginTokenText, t.bos},
		{endTokenText, t.eos},
		{userTokenText, t.user},
		{assistantTokenText, t.assistant},
		{thinkTokenText, t.thinking},
		{thinkEndTokenText, t.thinkingEnd},
	}
	if t.dsmlSet {
		markers = append(markers, struct {
			text string
			id   int
		}{"｜DSML｜", t.dsml})
	}
	var ids []int
	span, pos := 0, 0
	for pos < len(text) {
		var marker string
		markerID := 0
		for _, candidate := range markers {
			if strings.HasPrefix(text[pos:], candidate.text) {
				marker, markerID = candidate.text, candidate.id
				break
			}
		}
		if marker == "" {
			pos++
			continue
		}
		part, err := t.Encode(text[span:pos])
		if err != nil {
			return nil, err
		}
		ids = append(ids, part...)
		ids = append(ids, markerID)
		pos += len(marker)
		span = pos
	}
	part, err := t.Encode(text[span:])
	if err != nil {
		return nil, err
	}
	return append(ids, part...), nil
}

// RenderPrefix renders a validated multi-turn prefix transcript. Assistant
// turns are terminated with EOS; the final assistant turn is therefore a
// completed history rather than a generation prompt.
func (t *tokenizer) RenderPrefix(system string, turns []prefixTurn, mode ThinkMode) ([]int, error) {
	if len(turns) == 0 || turns[0].role != prefixUser || turns[len(turns)-1].role != prefixAssistant {
		return nil, fmt.Errorf("render prefix: turns must start with USER and end with ASSISTANT")
	}
	ids := []int{t.bos}
	if mode == ThinkMax {
		part, err := t.Encode(maxThinkPrefix)
		if err != nil {
			return nil, err
		}
		ids = append(ids, part...)
	}
	if system != "" {
		part, err := t.Encode(system)
		if err != nil {
			return nil, err
		}
		ids = append(ids, part...)
	}
	for _, turn := range turns {
		marker := t.user
		if turn.role == prefixAssistant {
			marker = t.assistant
		}
		ids = append(ids, marker)
		if turn.role == prefixAssistant && !strings.HasPrefix(turn.content, "<think>") && !strings.HasPrefix(turn.content, "</think>") {
			ids = append(ids, t.thinkingEnd)
		}
		part, err := t.Encode(turn.content)
		if err != nil {
			return nil, err
		}
		ids = append(ids, part...)
		if turn.role == prefixAssistant {
			ids = append(ids, t.eos)
		}
	}
	return ids, nil
}

const maxThinkPrefix = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"

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
		case isASCIIPunctSymbol(chars[i].r) && i+1 < len(chars) && isASCIIAlpha(chars[i+1].r):
			j = i + 2
			for j < len(chars) && isASCIIAlpha(chars[j].r) {
				j++
			}
		case isNonCJKLetterMark(chars[i].r) || i+1 < len(chars) && isLetterPrefix(chars[i].r) && isNonCJKLetterMark(chars[i+1].r):
			if !isNonCJKLetterMark(chars[i].r) {
				j++
			}
			for j < len(chars) && isNonCJKLetterMark(chars[j].r) {
				j++
			}
		case chars[i].r == ' ' && i+1 < len(chars) && isASCIIPunctSymbol(chars[i+1].r):
			j = i + 1
			for j < len(chars) && isASCIIPunctSymbol(chars[j].r) {
				j++
			}
			for j < len(chars) && isNewline(chars[j].r) {
				j++
			}
		case isASCIIPunctSymbol(chars[i].r):
			for j < len(chars) && isASCIIPunctSymbol(chars[j].r) {
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

func isCJK(r rune) bool              { return r >= 0x4e00 && r <= 0x9fa5 || r >= 0x3040 && r <= 0x30ff }
func isPunctSymbol(r rune) bool      { return unicode.IsPunct(r) || unicode.IsSymbol(r) }
func isLetterMark(r rune) bool       { return unicode.IsLetter(r) || unicode.IsMark(r) }
func isASCIIPunctSymbol(r rune) bool { return r < utf8.RuneSelf && isPunctSymbol(r) }
func isASCIIAlpha(r rune) bool       { return r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' }
func isNonCJKLetterMark(r rune) bool { return isLetterMark(r) && !isCJK(r) }
func isLetterPrefix(r rune) bool     { return !isNewline(r) && !unicode.IsLetter(r) && !isPunctSymbol(r) }
func isNewline(r rune) bool          { return r == '\r' || r == '\n' }
