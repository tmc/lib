package main

// Ordered JSON values with Python-compatible serialization.
//
// The model was trained on prompts assembled by Python, so json.dumps output is
// part of the input text: object key order, the ", " and ": " separators and
// repr-style float formatting all change the token stream. encoding/json's
// map[string]any loses key order and reformats numbers, so requests are parsed
// into this representation instead.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

type jsonKind int

const (
	jNull jsonKind = iota
	jBool
	jNumber
	jString
	jArray
	jObject
)

type jsonValue struct {
	kind jsonKind
	b    bool
	num  float64
	lit  string // number as written, to tell 3 from 3.0
	str  string
	arr  []*jsonValue
	keys []string
	vals []*jsonValue
}

// get returns the member named k, or nil. A nil receiver returns nil so callers
// can chain lookups on optional objects.
func (v *jsonValue) get(k string) *jsonValue {
	if v == nil || v.kind != jObject {
		return nil
	}
	for i, key := range v.keys {
		if key == k {
			return v.vals[i]
		}
	}
	return nil
}

// len reports the number of array elements or object members.
func (v *jsonValue) length() int {
	switch v.kind {
	case jArray:
		return len(v.arr)
	case jObject:
		return len(v.vals)
	}
	return 0
}

// at returns the i'th array element or object value.
func (v *jsonValue) at(i int) *jsonValue {
	if v.kind == jArray {
		return v.arr[i]
	}
	return v.vals[i]
}

func parseJSON(data []byte) (*jsonValue, error) {
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	v, err := parseValue(d)
	if err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing data after JSON value")
	}
	return v, nil
}

func parseValue(d *json.Decoder) (*jsonValue, error) {
	t, err := d.Token()
	if err != nil {
		return nil, err
	}
	return parseFrom(d, t)
}

func parseFrom(d *json.Decoder, t json.Token) (*jsonValue, error) {
	switch t := t.(type) {
	case nil:
		return &jsonValue{kind: jNull}, nil
	case bool:
		return &jsonValue{kind: jBool, b: t}, nil
	case string:
		return &jsonValue{kind: jString, str: t}, nil
	case json.Number:
		f, err := t.Float64()
		if err != nil {
			return nil, fmt.Errorf("bad number %q", t)
		}
		return &jsonValue{kind: jNumber, num: f, lit: string(t)}, nil
	case json.Delim:
		switch t {
		case '[':
			v := &jsonValue{kind: jArray}
			for d.More() {
				e, err := parseValue(d)
				if err != nil {
					return nil, err
				}
				v.arr = append(v.arr, e)
			}
			_, err := d.Token() // ']'
			return v, err
		case '{':
			v := &jsonValue{kind: jObject}
			for d.More() {
				k, err := d.Token()
				if err != nil {
					return nil, err
				}
				e, err := parseValue(d)
				if err != nil {
					return nil, err
				}
				v.keys = append(v.keys, k.(string))
				v.vals = append(v.vals, e)
			}
			_, err := d.Token() // '}'
			return v, err
		}
	}
	return nil, fmt.Errorf("unexpected JSON token %v", t)
}

// dumps writes v the way Python's json.dumps does. ascii selects
// ensure_ascii=True, which escapes every non-ASCII rune.
func (v *jsonValue) dumps(b *strings.Builder, ascii bool) {
	switch v.kind {
	case jNull:
		b.WriteString("null")
	case jBool:
		if v.b {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case jNumber:
		writeNumber(b, v)
	case jString:
		writeJSONString(b, v.str, ascii)
	case jArray:
		b.WriteByte('[')
		for i, e := range v.arr {
			if i > 0 {
				b.WriteString(", ")
			}
			e.dumps(b, ascii)
		}
		b.WriteByte(']')
	case jObject:
		b.WriteByte('{')
		for i, k := range v.keys {
			if i > 0 {
				b.WriteString(", ")
			}
			writeJSONString(b, k, ascii)
			b.WriteString(": ")
			v.vals[i].dumps(b, ascii)
		}
		b.WriteByte('}')
	}
}

func (v *jsonValue) dumpString(ascii bool) string {
	var b strings.Builder
	v.dumps(&b, ascii)
	return b.String()
}

// writeNumber prints an integer literal back unchanged, as Python does, and
// formats anything else like repr(float).
func writeNumber(b *strings.Builder, v *jsonValue) {
	s := v.lit
	i := 0
	if strings.HasPrefix(s, "-") {
		i = 1
	}
	j := i
	for j < len(s) && s[j] >= '0' && s[j] <= '9' {
		j++
	}
	if j > i && j == len(s) {
		if s == "-0" {
			b.WriteString("0")
		} else {
			b.WriteString(s)
		}
		return
	}
	b.WriteString(pyFloat(v.num))
}

// pyFloat formats x like Python's repr: the shortest round-tripping decimal,
// always with a fractional part or an exponent.
func pyFloat(x float64) string {
	switch {
	case math.IsNaN(x):
		return "NaN"
	case math.IsInf(x, 1):
		return "Infinity"
	case math.IsInf(x, -1):
		return "-Infinity"
	case x == 0:
		if math.Signbit(x) {
			return "-0.0"
		}
		return "0.0"
	}

	s := strconv.FormatFloat(x, 'e', -1, 64)
	neg := s[0] == '-'
	if neg {
		s = s[1:]
	}
	mant, exp, _ := strings.Cut(s, "e")
	e10, err := strconv.Atoi(exp)
	if err != nil {
		return s
	}
	dg := strings.Replace(mant, ".", "", 1)
	for len(dg) > 1 && dg[len(dg)-1] == '0' {
		dg = dg[:len(dg)-1]
	}

	var b strings.Builder
	if neg {
		b.WriteByte('-')
	}
	switch {
	case e10 >= 0 && e10 < 16:
		for i := 0; i <= e10; i++ {
			if i < len(dg) {
				b.WriteByte(dg[i])
			} else {
				b.WriteByte('0')
			}
		}
		b.WriteByte('.')
		if len(dg) > e10+1 {
			b.WriteString(dg[e10+1:])
		} else {
			b.WriteByte('0')
		}
	case e10 >= -4 && e10 < 0:
		b.WriteString("0.")
		b.WriteString(strings.Repeat("0", -e10-1))
		b.WriteString(dg)
	default:
		b.WriteByte(dg[0])
		if len(dg) > 1 {
			b.WriteByte('.')
			b.WriteString(dg[1:])
		}
		sign := byte('+')
		if e10 < 0 {
			sign, e10 = '-', -e10
		}
		fmt.Fprintf(&b, "e%c%02d", sign, e10)
	}
	return b.String()
}

// writeJSONString quotes s with Python's escape set.
func writeJSONString(b *strings.Builder, s string, ascii bool) {
	b.WriteByte('"')
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '"':
			b.WriteString(`\"`)
		case c == '\\':
			b.WriteString(`\\`)
		case c == '\n':
			b.WriteString(`\n`)
		case c == '\r':
			b.WriteString(`\r`)
		case c == '\t':
			b.WriteString(`\t`)
		case c == '\b':
			b.WriteString(`\b`)
		case c == '\f':
			b.WriteString(`\f`)
		case c < 0x20:
			fmt.Fprintf(b, `\u%04x`, c)
		case c < 0x80 || !ascii:
			b.WriteByte(c)
		default:
			r, n := utf8.DecodeRuneInString(s[i:])
			if r > 0xFFFF {
				r -= 0x10000
				fmt.Fprintf(b, `\u%04x\u%04x`, 0xD800+(r>>10), 0xDC00+(r&0x3FF))
			} else {
				fmt.Fprintf(b, `\u%04x`, r)
			}
			i += n
			continue
		}
		i++
	}
	b.WriteByte('"')
}
