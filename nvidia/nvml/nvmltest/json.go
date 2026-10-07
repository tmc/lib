package nvmltest

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"unicode"
)

// decode rejects ambiguous field identities before encoding/json can fold or
// overwrite them. Struct field spellings follow their JSON tags exactly.
func decode(b []byte, dst any) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := jsonValue(d, 0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing JSON input")
	}
	if err := jsonFields(b, reflect.TypeOf(dst)); err != nil {
		return err
	}
	d = json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	return d.Decode(dst)
}

func jsonValue(d *json.Decoder, depth int) error {
	// Each recorded error cause adds an object and an array. Allow the
	// supported 32-cause chain plus the recording envelope, while bounding
	// malformed input before typed allocation.
	if depth > 96 {
		return errors.New("JSON nesting exceeds limit")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	switch t {
	case json.Delim('{'):
		seen := make(map[string]bool)
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok {
				return errors.New("invalid JSON field name")
			}
			fold := foldName(name)
			if seen[fold] {
				return fmt.Errorf("duplicate or aliased JSON field %q", name)
			}
			seen[fold] = true
			if err := jsonValue(d, depth+1); err != nil {
				return err
			}
		}
		if t, err := d.Token(); err != nil || t != json.Delim('}') {
			return errors.New("incomplete JSON object")
		}
	case json.Delim('['):
		for d.More() {
			if err := jsonValue(d, depth+1); err != nil {
				return err
			}
		}
		if t, err := d.Token(); err != nil || t != json.Delim(']') {
			return errors.New("incomplete JSON array")
		}
	case json.Delim('}'), json.Delim(']'):
		return errors.New("unexpected JSON delimiter")
	}
	return nil
}

func foldName(s string) string {
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

func jsonFields(b []byte, typ reflect.Type) error {
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
		return nil
	}
	switch typ.Kind() {
	case reflect.Struct:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(b, &fields); err != nil {
			return err
		}
		allowed := make(map[string]reflect.Type)
		for i := 0; i < typ.NumField(); i++ {
			f := typ.Field(i)
			if f.PkgPath != "" {
				continue
			}
			name := strings.Split(f.Tag.Get("json"), ",")[0]
			if name == "-" {
				continue
			}
			if name == "" {
				name = f.Name
			}
			allowed[name] = f.Type
		}
		for name, value := range fields {
			field, ok := allowed[name]
			if !ok {
				return fmt.Errorf("unknown JSON field %q", name)
			}
			if err := jsonFields(value, field); err != nil {
				return err
			}
		}
	case reflect.Array, reflect.Slice:
		if typ == reflect.TypeFor[json.RawMessage]() {
			return nil
		}
		var values []json.RawMessage
		if err := json.Unmarshal(b, &values); err != nil {
			return err
		}
		for _, value := range values {
			if err := jsonFields(value, typ.Elem()); err != nil {
				return err
			}
		}
	}
	return nil
}
