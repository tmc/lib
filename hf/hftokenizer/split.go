package hftokenizer

import (
	"unicode/utf8"
)

const (
	gpt2Pattern  = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
	qwenPattern  = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	llamaPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
)

type splitKind int

const (
	splitNone splitKind = iota
	splitGPT2
	splitQwen
	splitLlama
)

// splitText implements the ordered alternatives of the supported patterns.
// Returning byte spans preserves original UTF-8 before byte-level conversion.
func splitText(text string, kind splitKind) []string {
	if text == "" {
		return nil
	}
	if kind == splitNone {
		return []string{text}
	}
	var out []string
	for text != "" {
		n := splitLength(text, kind)
		out = append(out, text[:n])
		text = text[n:]
	}
	return out
}

func splitLength(s string, kind splitKind) int {
	if s[0] == '\'' {
		for _, suffix := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
			// Case-insensitive contractions use Unicode simple folding, including long s.
			n := 1
			matched := true
			for _, want := range suffix {
				if n == len(s) {
					matched = false
					break
				}
				r, size := utf8.DecodeRuneInString(s[n:])
				if kind != splitGPT2 {
					if r >= 'A' && r <= 'Z' {
						r += 'a' - 'A'
					}
					if r == '\u017f' {
						r = 's'
					}
				}
				if r != want {
					matched = false
					break
				}
				n += size
			}
			if matched {
				return n
			}
		}
	}
	first, size := utf8.DecodeRuneInString(s)
	letterStart := 0
	if kind == splitGPT2 {
		if first == ' ' {
			letterStart = size
		}
	} else if first != '\r' && first != '\n' && !isLetter(first) && !isNumber(first) {
		letterStart = size
	}
	if n := scanRun(s, letterStart, isLetter); n > letterStart {
		return n
	}
	numberStart := 0
	if kind == splitGPT2 && first == ' ' {
		numberStart = size
	}
	n := numberStart
	limit := len(s)
	if kind == splitQwen {
		limit = 1
	} else if kind == splitLlama {
		limit = 3
	}
	for count := 0; n < len(s) && count < limit; count++ {
		r, size := utf8.DecodeRuneInString(s[n:])
		if !isNumber(r) {
			break
		}
		n += size
	}
	if n > numberStart {
		return n
	}
	punctStart := 0
	if first == ' ' {
		punctStart = size
	}
	n = scanRun(s, punctStart, func(r rune) bool { return !isSpace(r) && !isLetter(r) && !isNumber(r) })
	if n > punctStart {
		if kind != splitGPT2 {
			n = scanRun(s, n, func(r rune) bool { return r == '\r' || r == '\n' })
		}
		return n
	}
	n = scanRun(s, 0, isSpace)
	if kind != splitGPT2 {
		lastNewline := 0
		for i, r := range s[:n] {
			if r == '\r' || r == '\n' {
				lastNewline = i + 1
			}
		}
		if lastNewline != 0 {
			return lastNewline
		}
	}
	if n == len(s) {
		return n
	}
	// The negative lookahead allows backtracking over one whitespace rune.
	_, lastSize := utf8.DecodeLastRuneInString(s[:n])
	if n > lastSize {
		return n - lastSize
	}
	return n
}

func scanRun(s string, start int, match func(rune) bool) int {
	n := start
	for n < len(s) {
		r, size := utf8.DecodeRuneInString(s[n:])
		if !match(r) {
			break
		}
		n += size
	}
	return n
}
