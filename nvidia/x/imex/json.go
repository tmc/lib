package imex

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode"
	"unicode/utf8"
)

func unique(b []byte, depth int) error {
	if !utf8.Valid(b) {
		return errors.New("imex: invalid UTF-8")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := value(d, 0, depth); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("imex: trailing JSON data")
	}
	return nil
}
func value(d *json.Decoder, depth, max int) error {
	if depth > max {
		return errors.New("imex: JSON depth exceeded")
	}
	t, err := d.Token()
	if err != nil {
		return fmt.Errorf("imex JSON: %w", err)
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		keys := make(map[string]bool)
		for d.More() {
			t, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := t.(string)
			if !ok {
				return errors.New("imex: nonstring key")
			}
			// Fold aliases too: consumers must not disagree about known-field spelling.
			folded := foldKey(key)
			if keys[folded] {
				return fmt.Errorf("imex: duplicate field %q", key)
			}
			keys[folded] = true
			if err := value(d, depth+1, max); err != nil {
				return err
			}
		}
		t, err = d.Token()
		if err != nil || t != json.Delim('}') {
			return errors.New("imex: unclosed object")
		}
	case '[':
		for d.More() {
			if err := value(d, depth+1, max); err != nil {
				return err
			}
		}
		t, err = d.Token()
		if err != nil || t != json.Delim(']') {
			return errors.New("imex: unclosed array")
		}
	default:
		return errors.New("imex: unexpected delimiter")
	}
	return nil
}
func foldKey(s string) string {
	var b strings.Builder
	for _, r := range s {
		min := r
		for next := unicode.SimpleFold(r); next != r; next = unicode.SimpleFold(next) {
			if next < min {
				min = next
			}
		}
		b.WriteRune(min)
	}
	return b.String()
}
