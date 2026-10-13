package nsightsystems

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"unicode/utf8"
)

func uniqueRow(raw []byte) error {
	if !utf8.Valid(raw) {
		return errors.New("invalid UTF-8 row")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	start, err := d.Token()
	if err != nil || start != json.Delim('{') {
		return errors.New("row object required")
	}
	var keys []string
	for d.More() {
		t, err := d.Token()
		if err != nil {
			return err
		}
		key, ok := t.(string)
		if !ok {
			return errors.New("invalid row key")
		}
		for _, old := range keys {
			if strings.EqualFold(old, key) {
				return errors.New("duplicate or case-aliased row key")
			}
		}
		keys = append(keys, key)
		if len(keys) > 64 {
			return errors.New("row column limit exceeded")
		}
		var value json.RawMessage
		if err := d.Decode(&value); err != nil {
			return err
		}
	}
	if _, err := d.Token(); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing row document")
	}
	return nil
}
