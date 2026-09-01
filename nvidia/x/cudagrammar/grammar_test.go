// SPDX-License-Identifier: Apache-2.0

package cudagrammar_test

import (
	"encoding/json"
	"math/rand"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagrammar"
)

// personSchema exercises every supported node kind, with an object nested
// inside an array nested inside an object.
const personSchema = `{
	"type": "object",
	"properties": {
		"name":   {"type": "string"},
		"age":    {"type": "integer"},
		"score":  {"type": "number"},
		"active": {"type": "boolean"},
		"note":   {"type": "null"},
		"role":   {"enum": ["admin", "adept", "guest"]},
		"tags":   {"type": "array", "items": {"type": "string"}},
		"pets": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"kind": {"enum": ["cat", "dog"]},
					"legs": {"type": "integer"}
				}
			}
		}
	}
}`

func compile(t *testing.T, schema string) *cudagrammar.Grammar {
	t.Helper()
	g, err := cudagrammar.CompileString(schema)
	if err != nil {
		t.Fatalf("CompileString: %v", err)
	}
	return g
}

// accept feeds s to a fresh matcher and reports the outcome.
func accept(g *cudagrammar.Grammar, s string) error {
	m := g.Match()
	if err := m.AcceptString(s); err != nil {
		return err
	}
	return m.Finish()
}

func TestMatcherAccepts(t *testing.T) {
	g := compile(t, personSchema)

	tests := []struct {
		name string
		doc  string
	}{
		{"minimal", `{"name":"a","age":1,"score":0,"active":true,"note":null,"role":"admin","tags":[],"pets":[]}`},
		{"whitespace", `{ "name" : "a" , "age" : 1 , "score" : 0 , "active" : true , "note" : null , "role" : "admin" , "tags" : [ ] , "pets" : [ ] }`},
		{"negative and exponent", `{"name":"a","age":-12,"score":-1.5e-3,"active":false,"note":null,"role":"guest","tags":["x","y"],"pets":[]}`},
		{"nested objects", `{"name":"a","age":0,"score":1.25,"active":true,"note":null,"role":"adept","tags":["t"],"pets":[{"kind":"cat","legs":4},{"kind":"dog","legs":4}]}`},
		{"escapes", `{"name":"a\"b\\cé\n","age":0,"score":0,"active":true,"note":null,"role":"admin","tags":[],"pets":[]}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := accept(g, tt.doc); err != nil {
				t.Errorf("accepting %s: %v", tt.doc, err)
			}
			// The document is also valid JSON, so a decoder driven by
			// this grammar produces something a caller can unmarshal.
			var v any
			if err := json.Unmarshal([]byte(tt.doc), &v); err != nil {
				t.Errorf("the test document is not valid JSON: %v", err)
			}
		})
	}
}

// TestMatcherRejectsAtOffset checks that invalid documents are refused at the
// exact byte that first cannot be part of a conforming document, rather than
// at the end or not at all.
func TestMatcherRejectsAtOffset(t *testing.T) {
	const simple = `{"type":"object","properties":{"n":{"type":"integer"},"s":{"type":"string"}}}`
	g := compile(t, simple)

	tests := []struct {
		name string
		doc  string
		off  int // offset of the first byte that cannot be matched
	}{
		{"not an object", `["n"]`, 0},
		{"unquoted key", `{n:1,"s":"x"}`, 1},
		{"wrong key", `{"m":1}`, 2},
		{"wrong value type", `{"n":"1"}`, 5},
		{"integer with a fraction", `{"n":1.5}`, 6},
		{"missing required key", `{"n":1}`, 6},
		{"value for a missing key", `{"n":1,"s":true}`, 11},
		{"trailing garbage", `{"n":1,"s":"x"}}`, 15},
		{"trailing text", `{"n":1,"s":"x"} oops`, 16},
		{"unterminated escape", `{"n":1,"s":"\q"}`, 13},
		{"raw control character in a string", "{\"n\":1,\"s\":\"\n\"}", 12},
		{"leading zero run", `{"n":01,"s":"x"}`, 6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := g.Match()
			var err error
			for i := 0; i < len(tt.doc); i++ {
				if err = m.AcceptByte(tt.doc[i]); err != nil {
					break
				}
			}
			if err == nil {
				err = m.Finish()
				if err == nil {
					t.Fatalf("%s was accepted", tt.doc)
				}
				// An incomplete document is refused at its end.
				if m.Pos() != tt.off {
					t.Errorf("stopped at offset %d, want %d (%v)", m.Pos(), tt.off, err)
				}
				return
			}
			if m.Pos() != tt.off {
				t.Errorf("rejected at offset %d, want %d: %v", m.Pos(), tt.off, err)
			}
			if !strings.Contains(err.Error(), "cudagrammar:") {
				t.Errorf("error does not name the package: %v", err)
			}
		})
	}
}

// TestMatcherRejectsWrongEnum checks that an enum admits its values and only
// its values, including a prefix of one.
func TestMatcherRejectsWrongEnum(t *testing.T) {
	g := compile(t, `{"type":"object","properties":{"role":{"enum":["admin","adept"]}}}`)

	for _, ok := range []string{`{"role":"admin"}`, `{"role":"adept"}`} {
		if err := accept(g, ok); err != nil {
			t.Errorf("accepting %s: %v", ok, err)
		}
	}
	for _, bad := range []string{`{"role":"root"}`, `{"role":"ad"}`, `{"role":"admins"}`, `{"role":""}`} {
		if err := accept(g, bad); err == nil {
			t.Errorf("%s was accepted", bad)
		}
	}
}

func TestCompileUnsupported(t *testing.T) {
	tests := []struct {
		name   string
		schema string
		want   string
	}{
		{"oneOf", `{"oneOf":[{"type":"string"}]}`, "oneOf"},
		{"ref", `{"$ref":"#/defs/x"}`, "$ref"},
		{"no type", `{}`, "schema without a type"},
		{"unknown type", `{"type":"integerish"}`, "type integerish"},
		{"union type", `{"type":["string","null"]}`, "type that is not a single type name"},
		{"array without items", `{"type":"array"}`, "array without items"},
		{"object without properties", `{"type":"object"}`, "object without properties"},
		{"open object", `{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":true}`, "additionalProperties"},
		{"optional property", `{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["a"]}`, "optional property b"},
		{"non-string enum", `{"enum":[1,2]}`, "enum whose values are not all strings"},
		{"nested unsupported", `{"type":"object","properties":{"a":{"type":"string","minLength":2}}}`, "minLength at a"},
		{"constrained item", `{"type":"array","items":{"type":"string","pattern":"x"}}`, "pattern at items"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := cudagrammar.CompileString(tt.schema)
			if err == nil {
				t.Fatalf("compiled a schema using %s", tt.want)
			}
			if !strings.Contains(err.Error(), "cudagrammar: unsupported schema feature: "+tt.want) {
				t.Errorf("error = %v, want it to name %q as unsupported", err, tt.want)
			}
		})
	}
}

// testVocab is a small tokenizer vocabulary: every printable non-space ASCII
// byte as its own token, plus multi-byte pieces a real tokenizer would have
// merged. Whitespace is left out so the generation driver terminates: the
// grammar admits whitespace between tokens forever, which is correct and
// would let a random sampler emit padding without end.
var testVocab = func() []string {
	var v []string
	for c := byte(0x21); c < 0x7f; c++ {
		v = append(v, string(c))
	}
	return append(v,
		`"name"`, `"age"`, `"role"`, `"tags"`, `"n"`, `"s"`,
		"true", "false", "null", "admin", "guest", `":`, `","`,
		"12", "0.5", "-3", "hello", "world", "[]", "{}", "",
	)
}()

func vocabView(id int) []byte {
	if id < 0 || id >= len(testVocab) {
		return nil
	}
	return []byte(testVocab[id])
}

func TestTokenMaskAdmitsOnlyValidTokens(t *testing.T) {
	g := compile(t, `{"type":"object","properties":{"n":{"type":"integer"},"s":{"type":"string"}}}`)
	m := g.Match()

	// At the start only '{' can continue the document.
	ids := m.TokenMask(vocabView, len(testVocab), nil)
	got := map[string]bool{}
	for _, id := range ids {
		got[testVocab[id]] = true
	}
	// "{}" is not admissible: the object has a required key, so nothing can
	// follow the brace but that key.
	want := map[string]bool{"{": true}
	for tok := range want {
		if !got[tok] {
			t.Errorf("token %q is not admitted at the start of the document", tok)
		}
	}
	for tok := range got {
		if !want[tok] {
			t.Errorf("token %q is admitted at the start of the document", tok)
		}
	}

	// After the opening brace the only admissible key is "n".
	if err := m.AcceptString("{"); err != nil {
		t.Fatalf("AcceptString: %v", err)
	}
	for _, tok := range []string{`"s"`, "true", "hello"} {
		id := tokenID(t, tok)
		if m.Admits(vocabView, id) {
			t.Errorf("token %q admitted where only the key %q fits", tok, "n")
		}
	}
	if !m.Admits(vocabView, tokenID(t, `"n"`)) {
		t.Errorf("the required key token %q was not admitted", `"n"`)
	}
}

func TestApplyMask(t *testing.T) {
	g := compile(t, `{"type":"object","properties":{"n":{"type":"integer"}}}`)
	m := g.Match()

	logits := make([]float32, len(testVocab))
	for i := range logits {
		logits[i] = 1
	}
	m.ApplyMask(vocabView, logits)

	if got := logits[tokenID(t, "{")]; got != 1 {
		t.Errorf("logit for %q = %g, want it untouched", "{", got)
	}
	for _, tok := range []string{"hello", "true", "["} {
		if got := logits[tokenID(t, tok)]; got != cudagrammar.MaskedLogit {
			t.Errorf("logit for %q = %g, want %g", tok, got, cudagrammar.MaskedLogit)
		}
	}
}

func tokenID(t *testing.T, tok string) int {
	t.Helper()
	for i, v := range testVocab {
		if v == tok {
			return i
		}
	}
	t.Fatalf("token %q is not in the test vocabulary", tok)
	return -1
}

// TestGenerationDriver decodes with the grammar the way a sampler would: at
// every step it takes the admissible token ids and picks one, and it never
// consults the schema itself. Whatever comes out has to be JSON that conforms,
// which is checked by unmarshalling it and inspecting the result.
func TestGenerationDriver(t *testing.T) {
	g := compile(t, personSchema)

	for seed := int64(0); seed < 25; seed++ {
		rng := rand.New(rand.NewSource(seed))
		m := g.Match()
		var out strings.Builder
		var ids []int
		for step := 0; !m.Done(); step++ {
			if step > 4000 {
				t.Fatalf("seed %d: no complete document after %d tokens: %q", seed, step, out.String())
			}
			ids = m.TokenMask(vocabView, len(testVocab), ids[:0])
			if len(ids) == 0 {
				t.Fatalf("seed %d: no admissible token after %q", seed, out.String())
			}
			tok := testVocab[ids[rng.Intn(len(ids))]]
			if err := m.AcceptString(tok); err != nil {
				t.Fatalf("seed %d: token %q was admitted but not accepted: %v", seed, tok, err)
			}
			out.WriteString(tok)
		}

		doc := out.String()
		// The numbers are decoded as json.Number: the grammar admits any
		// JSON number, including exponents past the range of a float64.
		var v struct {
			Name   *string      `json:"name"`
			Age    *json.Number `json:"age"`
			Score  *json.Number `json:"score"`
			Active *bool        `json:"active"`
			Note   any          `json:"note"`
			Role   *string      `json:"role"`
			Tags   *[]string    `json:"tags"`
			Pets   *[]struct {
				Kind *string      `json:"kind"`
				Legs *json.Number `json:"legs"`
			} `json:"pets"`
		}
		if err := json.Unmarshal([]byte(doc), &v); err != nil {
			t.Fatalf("seed %d: generated %q, which is not valid JSON: %v", seed, doc, err)
		}
		if v.Name == nil || v.Age == nil || v.Score == nil || v.Active == nil || v.Role == nil || v.Tags == nil || v.Pets == nil {
			t.Fatalf("seed %d: generated %q, which is missing a required property or gave it the wrong type", seed, doc)
		}
		if v.Note != nil {
			t.Errorf("seed %d: note = %v, want null", seed, v.Note)
		}
		if !isIntegerLiteral(v.Age.String()) {
			t.Errorf("seed %d: age = %v, want an integer", seed, *v.Age)
		}
		switch *v.Role {
		case "admin", "adept", "guest":
		default:
			t.Errorf("seed %d: role = %q, want an enumerated value", seed, *v.Role)
		}
		for i, p := range *v.Pets {
			if p.Kind == nil || p.Legs == nil {
				t.Errorf("seed %d: pet %d is missing a property in %q", seed, i, doc)
				continue
			}
			if *p.Kind != "cat" && *p.Kind != "dog" {
				t.Errorf("seed %d: pet %d kind = %q, want an enumerated value", seed, i, *p.Kind)
			}
			if !isIntegerLiteral(p.Legs.String()) {
				t.Errorf("seed %d: pet %d legs = %v, want an integer", seed, i, *p.Legs)
			}
		}
	}
}

// isIntegerLiteral reports whether s is a JSON number with no fractional or
// exponent part. It is a syntactic check rather than a conversion, because an
// integer schema admits values wider than any Go integer type.
func isIntegerLiteral(s string) bool {
	if strings.HasPrefix(s, "-") {
		s = s[1:]
	}
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// TestFinishRejectsIncomplete covers documents that run out mid-value, where
// nothing has been mismatched but nothing is complete either.
func TestFinishRejectsIncomplete(t *testing.T) {
	g := compile(t, `{"type":"object","properties":{"n":{"type":"integer"}}}`)
	for _, doc := range []string{"", "{", `{"n"`, `{"n":`, `{"n":1`, `{"n":-`} {
		m := g.Match()
		if err := m.AcceptString(doc); err != nil {
			t.Fatalf("accepting the prefix %q: %v", doc, err)
		}
		if err := m.Finish(); err == nil {
			t.Errorf("the incomplete document %q was reported complete", doc)
		}
	}
	// A bare number is only complete once Finish ends it.
	g2 := compile(t, `{"type":"number"}`)
	m := g2.Match()
	if err := m.AcceptString("-1.5e2"); err != nil {
		t.Fatalf("accepting a number: %v", err)
	}
	if err := m.Finish(); err != nil {
		t.Errorf("Finish after a complete number: %v", err)
	}
}
