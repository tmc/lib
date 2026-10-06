package httpapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"unicode"
)

// validateJSON limits nested work and rejects ambiguous duplicate members.
func validateJSON(body []byte) error {
	d := json.NewDecoder(bytes.NewReader(body))
	d.UseNumber()
	tokens := 0
	var value func(int) error
	next := func() (json.Token, error) {
		tokens++
		if tokens > 100000 {
			return nil, errors.New("decode request: token limit exceeded")
		}
		return d.Token()
	}
	value = func(depth int) error {
		if depth > 64 {
			return errors.New("decode request: nesting limit exceeded")
		}
		t, err := next()
		if err != nil {
			return err
		}
		start, compound := t.(json.Delim)
		if !compound {
			return nil
		}
		switch start {
		case '{':
			seen := make(map[string]bool)
			for d.More() {
				key, err := next()
				if err != nil {
					return err
				}
				name, ok := key.(string)
				name = foldJSONName(name)
				if !ok || seen[name] {
					return errors.New("decode request: duplicate or invalid member")
				}
				seen[name] = true
				if err := value(depth + 1); err != nil {
					return err
				}
			}
		case '[':
			for d.More() {
				if err := value(depth + 1); err != nil {
					return err
				}
			}
		default:
			return errors.New("decode request: invalid delimiter")
		}
		_, err = next()
		return err
	}
	if err := value(0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("decode request: trailing JSON")
	}
	return nil
}

// encoding/json matches struct fields using Unicode case folding. Use the
// smallest rune in each folding cycle so aliases share one map entry.
func foldJSONName(name string) string {
	return strings.Map(func(r rune) rune {
		for {
			next := unicode.SimpleFold(r)
			if next <= r {
				return next
			}
			r = next
		}
	}, name)
}
