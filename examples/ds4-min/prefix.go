package main

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
)

var errInvalidPrefix = errors.New("invalid prefix file")

type prefixRole uint8

const (
	prefixUser prefixRole = iota + 1
	prefixAssistant
)

type prefixTurn struct {
	role    prefixRole
	content string
}

// parsePrefixFile parses the USER:/ASSISTANT: transcript accepted by the
// upstream prefix-file path. It does not tokenize or render the turns.
func parsePrefixFile(data []byte) ([]prefixTurn, error) {
	if len(data) >= 3 && bytes.Equal(data[:3], []byte{0xef, 0xbb, 0xbf}) {
		data = data[3:]
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("%w: prefix file is empty", errInvalidPrefix)
	}
	if bytes.IndexByte(data, 0) >= 0 {
		return nil, fmt.Errorf("%w: prefix file contains a NUL byte", errInvalidPrefix)
	}
	lines := strings.Split(string(data), "\n")
	var turns []prefixTurn
	var role prefixRole
	var content []string
	markerLine := 0
	flush := func() error {
		if role == 0 {
			return nil
		}
		text := strings.Join(content, "\n")
		text = strings.TrimSuffix(text, "\n")
		text = strings.TrimSuffix(text, "\r")
		if text == "" {
			name := "USER"
			if role == prefixAssistant {
				name = "ASSISTANT"
			}
			return fmt.Errorf("%w: line %d has an empty %s turn", errInvalidPrefix, markerLine, name)
		}
		turns = append(turns, prefixTurn{role: role, content: text})
		content = nil
		return nil
	}
	for i, raw := range lines {
		lineNo := i + 1
		line := strings.TrimSuffix(raw, "\r")
		var next prefixRole
		switch {
		case strings.HasPrefix(line, "USER:"):
			next = prefixUser
		case strings.HasPrefix(line, "ASSISTANT:"):
			next = prefixAssistant
		}
		if next != 0 {
			if err := flush(); err != nil {
				return nil, err
			}
			want := prefixUser
			if len(turns)%2 == 1 {
				want = prefixAssistant
			}
			if next != want {
				name := "USER"
				if want == prefixAssistant {
					name = "ASSISTANT"
				}
				return nil, fmt.Errorf("%w: line %d: expected %s:", errInvalidPrefix, lineNo, name)
			}
			role, markerLine = next, lineNo
			text := line[strings.IndexByte(line, ':')+1:]
			if len(text) > 0 && (text[0] == ' ' || text[0] == '\t') {
				text = text[1:]
			}
			content = []string{text}
			continue
		}
		if role == 0 {
			return nil, fmt.Errorf("%w: line %d must start with USER: or ASSISTANT:", errInvalidPrefix, lineNo)
		}
		content = append(content, line)
	}
	if err := flush(); err != nil {
		return nil, err
	}
	if len(turns) == 0 || turns[len(turns)-1].role != prefixAssistant {
		return nil, fmt.Errorf("%w: prefix file must end with an ASSISTANT turn", errInvalidPrefix)
	}
	return turns, nil
}
