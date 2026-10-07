package cudainject

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
)

// correlationPayload keeps unknown producer fields available in the original
// wire payload, but refuses ambiguous spellings of fields the typed join uses.
func correlationPayload(payload []byte, value any, required ...string) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if err := correlationKeys(decoder, 0); err != nil {
		return err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return errors.New("cudainject: trailing correlation JSON")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(payload, &fields); err != nil {
		return err
	}
	if fields == nil {
		return errors.New("cudainject: correlation object required")
	}
	typ := reflect.TypeOf(value).Elem()
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		name := strings.Split(field.Tag.Get("json"), ",")[0]
		if name == "" {
			name = field.Name
		}
		if name == "-" {
			continue
		}
		for key, raw := range fields {
			if strings.EqualFold(key, name) {
				if key != name {
					return fmt.Errorf("cudainject: noncanonical correlation field %q", key)
				}
				if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
					return fmt.Errorf("cudainject: null correlation field %q", key)
				}
			}
		}
	}
	for _, key := range required {
		if _, ok := fields[key]; !ok {
			return fmt.Errorf("cudainject: correlation field %q required", key)
		}
	}
	return json.Unmarshal(payload, value)
}

func correlationKeys(d *json.Decoder, depth int) error {
	if depth > 64 {
		return errors.New("cudainject: correlation nesting limit exceeded")
	}
	token, err := d.Token()
	if err != nil {
		return err
	}
	delimiter, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delimiter {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[name] {
				return errors.New("cudainject: duplicate correlation JSON key")
			}
			seen[name] = true
			if err := correlationKeys(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := correlationKeys(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return errors.New("cudainject: correlation JSON delimiter refused")
	}
	_, err = d.Token()
	return err
}
