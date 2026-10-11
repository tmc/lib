package cudashadow

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"unicode"
)

func uniqueJSON(data []byte) error {
	d := json.NewDecoder(bytes.NewReader(data))
	var value func(int) error
	value = func(depth int) error {
		if depth > 32 {
			return errors.New("cudashadow: json nesting limit exceeded")
		}
		token, err := d.Token()
		if err != nil {
			return err
		}
		delim, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		if delim != '{' && delim != '[' {
			return errors.New("cudashadow: unexpected json delimiter")
		}
		seen := make(map[string]bool)
		for d.More() {
			if delim == '{' {
				token, err := d.Token()
				if err != nil {
					return err
				}
				key, ok := token.(string)
				if !ok {
					return errors.New("cudashadow: invalid json key")
				}
				key = strings.Map(func(r rune) rune {
					smallest := r
					for next := unicode.SimpleFold(r); next != r; next = unicode.SimpleFold(next) {
						if next < smallest {
							smallest = next
						}
					}
					return smallest
				}, key)
				if seen[key] {
					return errors.New("cudashadow: duplicate or case-aliased json field")
				}
				seen[key] = true
			}
			if err := value(depth + 1); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	}
	if err := value(0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("cudashadow: trailing json data")
	}
	return nil
}
