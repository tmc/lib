package advisecorpus

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// canonical uses the sidecar's sorted, ASCII JSON identity. Float rendering
// follows Python's shortest round-trip JSON spelling, including .0 on fixed
// integral floats. This is a corpus identity, never a source-upload digest.
func canonical(raw []byte) ([]byte, error) {
	if err := validUnicode(raw); err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	value, err := readValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err = dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing JSON input")
	}
	var b strings.Builder
	if err := writeValue(&b, value); err != nil {
		return nil, err
	}
	return []byte(b.String()), nil
}

// encoding/json replaces malformed Unicode with U+FFFD. Reject it before
// decoding so distinct corpus identities cannot silently collapse. Other JSON
// syntax remains the decoder's responsibility.
func validUnicode(raw []byte) error {
	if !utf8.Valid(raw) {
		return fmt.Errorf("invalid UTF-8 in JSON")
	}
	inString := false
	for i := 0; i < len(raw); {
		if raw[i] == '"' {
			inString = !inString
			i++
			continue
		}
		if !inString || raw[i] != '\\' {
			i++
			continue
		}
		if i+1 >= len(raw) {
			return fmt.Errorf("truncated JSON escape")
		}
		if raw[i+1] != 'u' {
			i += 2
			continue
		}
		if i+6 > len(raw) {
			return fmt.Errorf("truncated Unicode escape")
		}
		n, err := strconv.ParseUint(string(raw[i+2:i+6]), 16, 16)
		if err != nil {
			return fmt.Errorf("invalid Unicode escape")
		}
		if n >= 0xd800 && n <= 0xdbff {
			if i+12 > len(raw) || raw[i+6] != '\\' || raw[i+7] != 'u' {
				return fmt.Errorf("unpaired Unicode surrogate")
			}
			low, err := strconv.ParseUint(string(raw[i+8:i+12]), 16, 16)
			if err != nil || low < 0xdc00 || low > 0xdfff {
				return fmt.Errorf("unpaired Unicode surrogate")
			}
			i += 12
			continue
		}
		if n >= 0xdc00 && n <= 0xdfff {
			return fmt.Errorf("unpaired Unicode surrogate")
		}
		i += 6
	}
	return nil
}

func readValue(d *json.Decoder) (any, error) {
	t, err := d.Token()
	if err != nil {
		return nil, err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return t, nil
	}
	switch delim {
	case '{':
		m := make(map[string]any)
		for d.More() {
			t, err := d.Token()
			if err != nil {
				return nil, err
			}
			key, ok := t.(string)
			if !ok {
				return nil, fmt.Errorf("JSON key is not a string")
			}
			if _, exists := m[key]; exists {
				return nil, fmt.Errorf("duplicate JSON field: %s", key)
			}
			v, err := readValue(d)
			if err != nil {
				return nil, err
			}
			m[key] = v
		}
		if t, err := d.Token(); err != nil || t != json.Delim('}') {
			return nil, fmt.Errorf("invalid JSON object")
		}
		return m, nil
	case '[':
		a := []any{}
		for d.More() {
			v, err := readValue(d)
			if err != nil {
				return nil, err
			}
			a = append(a, v)
		}
		if t, err := d.Token(); err != nil || t != json.Delim(']') {
			return nil, fmt.Errorf("invalid JSON array")
		}
		return a, nil
	default:
		return nil, fmt.Errorf("unexpected JSON delimiter")
	}
}
func writeValue(b *strings.Builder, v any) error {
	switch v := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		b.WriteString(strconv.FormatBool(v))
	case string:
		var out bytes.Buffer
		e := json.NewEncoder(&out)
		e.SetEscapeHTML(false)
		if err := e.Encode(v); err != nil {
			return err
		}
		s := strings.TrimSuffix(out.String(), "\n")
		for len(s) > 0 {
			r, n := utf8.DecodeRuneInString(s)
			s = s[n:]
			if r < 127 {
				b.WriteRune(r)
			} else if r <= 0xffff {
				fmt.Fprintf(b, "\\u%04x", r)
			} else {
				a, c := utf16.EncodeRune(r)
				fmt.Fprintf(b, "\\u%04x\\u%04x", a, c)
			}
		}
	case json.Number:
		s := string(v)
		if !strings.ContainsAny(s, ".eE") {
			if s == "-0" {
				s = "0"
			}
			b.WriteString(s)
			break
		}
		f, err := v.Float64()
		if err != nil || math.IsInf(f, 0) || math.IsNaN(f) {
			return fmt.Errorf("non-finite JSON number")
		}
		expForm := strconv.FormatFloat(f, 'e', -1, 64)
		i := strings.IndexByte(expForm, 'e')
		exp, _ := strconv.Atoi(expForm[i+1:])
		if exp >= -4 && exp < 16 {
			s = strconv.FormatFloat(f, 'f', -1, 64)
			if !strings.Contains(s, ".") {
				s += ".0"
			}
		} else {
			sign := "+"
			if exp < 0 {
				sign = "-"
				exp = -exp
			}
			s = fmt.Sprintf("%se%s%02d", expForm[:i], sign, exp)
		}
		b.WriteString(s)
	case []any:
		b.WriteByte('[')
		for i, x := range v {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := writeValue(b, x); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case map[string]any:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := writeValue(b, k); err != nil {
				return err
			}
			b.WriteByte(':')
			if err := writeValue(b, v[k]); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		return fmt.Errorf("unsupported JSON value")
	}
	return nil
}
