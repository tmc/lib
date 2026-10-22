package dsv4server

import (
	"errors"
	"strings"
)

const (
	thinkStart = "<think>"
	thinkEnd   = "</think>"
)

type thinkingOutput struct {
	reasoning string
	content   string
	present   bool
}

// thinkingChunk is one decoded stream fragment after the leading thinking
// block has been classified. At most one field is non-empty.
type thinkingChunk struct {
	reasoning string
	content   string
}

// thinkingStream recognizes the same format as splitThinkingOutput without
// requiring a token boundary to coincide with a marker boundary.
type thinkingStream struct {
	state   uint8
	pending string
}

const (
	streamUnknown uint8 = iota
	streamReasoning
	streamContent
)

func (s *thinkingStream) feed(text string) ([]thinkingChunk, error) {
	if text == "" {
		return nil, nil
	}
	s.pending += text
	var out []thinkingChunk
	for {
		switch s.state {
		case streamUnknown:
			if strings.HasPrefix(s.pending, thinkStart) {
				s.pending = s.pending[len(thinkStart):]
				s.state = streamReasoning
				continue
			}
			if strings.HasPrefix(thinkStart, s.pending) {
				return out, nil
			}
			if strings.Contains(s.pending, thinkStart) || strings.Contains(s.pending, thinkEnd) {
				return nil, errors.New("thinking output has a marker outside a leading block")
			}
			// A partial marker is held above; everything else is plain output.
			if s.pending != "" {
				out = append(out, thinkingChunk{content: s.pending})
				s.pending = ""
				s.state = streamContent
			}
			return out, nil
		case streamReasoning:
			i := strings.Index(s.pending, thinkEnd)
			if i < 0 {
				keep := markerSuffix(s.pending, thinkEnd)
				if len(s.pending)-keep > 0 {
					out = append(out, thinkingChunk{reasoning: s.pending[:len(s.pending)-keep]})
					s.pending = s.pending[len(s.pending)-keep:]
				}
				return out, nil
			}
			if i > 0 {
				out = append(out, thinkingChunk{reasoning: s.pending[:i]})
			}
			s.pending = s.pending[i+len(thinkEnd):]
			s.state = streamContent
			continue
		case streamContent:
			if strings.Contains(s.pending, thinkStart) || strings.Contains(s.pending, thinkEnd) {
				return nil, errors.New("thinking output has multiple or nested markers")
			}
			out = append(out, thinkingChunk{content: s.pending})
			s.pending = ""
			return out, nil
		}
	}
}

func (s *thinkingStream) done() ([]thinkingChunk, error) {
	if s.state == streamReasoning || (s.state == streamUnknown && s.pending != "") {
		if s.state == streamUnknown && !strings.HasPrefix(thinkStart, s.pending) {
			return []thinkingChunk{{content: s.pending}}, nil
		}
		return nil, errors.New("thinking output is unterminated")
	}
	if s.state == streamContent && s.pending != "" {
		return []thinkingChunk{{content: s.pending}}, nil
	}
	return nil, nil
}

func markerSuffix(text, marker string) int {
	for n := len(marker) - 1; n > 0; n-- {
		if strings.HasSuffix(text, marker[:n]) {
			return n
		}
	}
	return 0
}

// splitThinkingOutput separates the raw model representation used by the
// DeepSeek server from the fields exposed by the HTTP APIs. A thinking block
// is recognized only at the start of the output and must have exactly one
// closing marker. This deliberately rejects malformed output: treating an
// unterminated or embedded marker as ordinary content would silently put
// reasoning on the wrong side of an API boundary.
func splitThinkingOutput(text string) (thinkingOutput, error) {
	if !strings.Contains(text, thinkStart) && !strings.Contains(text, thinkEnd) {
		return thinkingOutput{content: text}, nil
	}
	if !strings.HasPrefix(text, thinkStart) {
		return thinkingOutput{}, errors.New("thinking output has a marker outside a leading block")
	}
	rest := text[len(thinkStart):]
	end := strings.Index(rest, thinkEnd)
	if end < 0 {
		return thinkingOutput{}, errors.New("thinking output is unterminated")
	}
	reasoning := rest[:end]
	content := rest[end+len(thinkEnd):]
	if strings.Contains(reasoning, thinkStart) || strings.Contains(reasoning, thinkEnd) ||
		strings.Contains(content, thinkStart) || strings.Contains(content, thinkEnd) {
		return thinkingOutput{}, errors.New("thinking output has multiple or nested markers")
	}
	return thinkingOutput{reasoning: reasoning, content: content, present: true}, nil
}
