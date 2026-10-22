package dsv4server

import (
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"
)

const (
	maxStopSequences = 4
	maxStopBytes     = 256
	maxStopTotal     = 1024
)

type stopSequences []string

// parseStop accepts the OpenAI string-or-array form and Anthropic's array
// form. Unknown JSON shapes are rejected rather than silently ignored.
func parseStop(v any) (stopSequences, error) {
	if v == nil {
		return nil, nil
	}
	var out []string
	switch x := v.(type) {
	case string:
		out = []string{x}
	case []any:
		if len(x) == 0 {
			return nil, errors.New("stop must not be empty")
		}
		for _, item := range x {
			s, ok := item.(string)
			if !ok {
				return nil, errors.New("stop entries must be strings")
			}
			out = append(out, s)
		}
	default:
		return nil, errors.New("stop must be a string or array of strings")
	}
	if len(out) == 0 || len(out) > maxStopSequences {
		return nil, fmt.Errorf("stop must contain 1 through %d sequences", maxStopSequences)
	}
	total := 0
	for _, s := range out {
		if s == "" || !utf8.ValidString(s) {
			return nil, errors.New("stop sequences must be non-empty valid UTF-8")
		}
		if len(s) > maxStopBytes {
			return nil, fmt.Errorf("stop sequence exceeds %d bytes", maxStopBytes)
		}
		total += len(s)
	}
	if total > maxStopTotal {
		return nil, fmt.Errorf("stop sequences exceed %d bytes", maxStopTotal)
	}
	return stopSequences(out), nil
}

func (s stopSequences) cut(text string) (string, bool) {
	if len(s) == 0 {
		return text, false
	}
	cut := len(text)
	for _, seq := range s {
		if i := strings.Index(text, seq); i >= 0 && i < cut {
			cut = i
		}
	}
	return text[:cut], cut != len(text)
}

// add returns the part that cannot still become part of a stop sequence.
// Keeping the possible prefix buffered prevents an SSE stream from leaking a
// stop marker when it is split across model tokens.
func (s stopSequences) add(buffer *string, fragment string) (safe string, stopped bool) {
	*buffer += fragment
	if cut, ok := s.cut(*buffer); ok {
		*buffer = ""
		return cut, true
	}
	keep := 0
	for _, seq := range s {
		for n := 1; n < len(seq) && n <= len(*buffer); n++ {
			if strings.HasSuffix(*buffer, seq[:n]) && n > keep {
				keep = n
			}
		}
	}
	if keep == 0 {
		safe, *buffer = *buffer, ""
		return safe, false
	}
	n := len(*buffer) - keep
	safe, *buffer = (*buffer)[:n], (*buffer)[n:]
	return safe, false
}

func (s stopSequences) finish(buffer *string) string {
	safe, _ := s.cut(*buffer)
	*buffer = ""
	return safe
}
