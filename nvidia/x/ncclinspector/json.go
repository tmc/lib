package ncclinspector

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode"
)

func requiredObject(b []byte, keys ...string) (map[string]json.RawMessage, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil || m == nil {
		return nil, errors.New("required object is missing or malformed")
	}
	for _, k := range keys {
		if v, ok := m[k]; !ok || bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
			return nil, fmt.Errorf("missing %s", k)
		}
	}
	return m, nil
}

func uniqueJSON(b []byte) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := jsonValue(d, 0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing or malformed JSON")
	}
	return nil
}

func jsonValue(d *json.Decoder, depth int) error {
	if depth > 32 {
		return errors.New("JSON nesting limit exceeded")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	if delim != '{' && delim != '[' {
		return errors.New("unexpected JSON delimiter")
	}
	seen := make(map[string]bool)
	for d.More() {
		if delim == '{' {
			t, err := d.Token()
			if err != nil {
				return err
			}
			k, ok := t.(string)
			if !ok {
				return errors.New("invalid JSON key")
			}
			k = fold(k)
			if seen[k] {
				return errors.New("duplicate or case-aliased JSON key")
			}
			seen[k] = true
		}
		if err := jsonValue(d, depth+1); err != nil {
			return err
		}
	}
	_, err = d.Token()
	return err
}

func fold(s string) string {
	return strings.Map(func(r rune) rune {
		m := r
		for n := unicode.SimpleFold(r); n != r; n = unicode.SimpleFold(n) {
			if n < m {
				m = n
			}
		}
		return m
	}, s)
}
