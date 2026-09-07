package hftokenizer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

type schema struct {
	Version       string          `json:"version"`
	Truncation    json.RawMessage `json:"truncation"`
	Padding       json.RawMessage `json:"padding"`
	Normalizer    json.RawMessage `json:"normalizer"`
	PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
	PostProcessor json.RawMessage `json:"post_processor"`
	Decoder       json.RawMessage `json:"decoder"`
	Model         struct {
		Type                    string            `json:"type"`
		Vocab                   map[string]int32  `json:"vocab"`
		Merges                  []json.RawMessage `json:"merges"`
		Dropout                 *float64          `json:"dropout"`
		UnkToken                *string           `json:"unk_token"`
		ContinuingSubwordPrefix string            `json:"continuing_subword_prefix"`
		EndOfWordSuffix         string            `json:"end_of_word_suffix"`
		FuseUnk                 bool              `json:"fuse_unk"`
		ByteFallback            bool              `json:"byte_fallback"`
		IgnoreMerges            bool              `json:"ignore_merges"`
	} `json:"model"`
	Added []struct {
		ID         *int32 `json:"id"`
		Content    string `json:"content"`
		SingleWord bool   `json:"single_word"`
		LStrip     bool   `json:"lstrip"`
		RStrip     bool   `json:"rstrip"`
		Normalized *bool  `json:"normalized"`
		Special    bool   `json:"special"`
	} `json:"added_tokens"`
}

func strictJSON(data []byte, dst any) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	return nil
}

func present(raw json.RawMessage) bool { return len(raw) > 0 && string(bytes.TrimSpace(raw)) != "null" }

func parse(data []byte) (*Tokenizer, error) {
	var s schema
	if err := strictJSON(data, &s); err != nil {
		return nil, fmt.Errorf("tokenizer: JSON: %w", err)
	}
	if err := validSurrogates(data); err != nil {
		return nil, fmt.Errorf("tokenizer: JSON: %w", err)
	}
	if err := uniqueJSON(json.NewDecoder(bytes.NewReader(data)), 0); err != nil {
		return nil, fmt.Errorf("tokenizer: JSON: %w", err)
	}
	if s.Version != "1.0" {
		return nil, fmt.Errorf("tokenizer: unsupported version %q", s.Version)
	}
	if present(s.Truncation) || present(s.Padding) {
		return nil, fmt.Errorf("tokenizer: truncation and padding are unsupported")
	}
	m := s.Model
	if m.Type != "BPE" || m.Dropout != nil && *m.Dropout != 0 || m.UnkToken != nil || m.ContinuingSubwordPrefix != "" || m.EndOfWordSuffix != "" || m.FuseUnk || m.ByteFallback || m.IgnoreMerges {
		return nil, fmt.Errorf("tokenizer: unsupported BPE model options")
	}
	t := &Tokenizer{merges: make(map[pair]merge), tokens: make(map[int32][]byte), special: make(map[int32]bool), maxID: -1}
	var err error
	if present(s.Normalizer) {
		var n struct {
			Type string `json:"type"`
		}
		if err := strictJSON(s.Normalizer, &n); err != nil || n.Type != "NFC" {
			return nil, fmt.Errorf("tokenizer: unsupported normalizer")
		}
		t.nfc = true
	}
	t.split, t.addPrefixSpace, err = parsePreTokenizer(s.PreTokenizer)
	if err != nil {
		return nil, err
	}
	var decoder byteLevel
	if err := strictJSON(s.Decoder, &decoder); err != nil || decoder.Type != "ByteLevel" {
		return nil, fmt.Errorf("tokenizer: unsupported decoder")
	}
	alphabet, inverse := byteAlphabet()
	names := make(map[int32]string)
	for name, id := range m.Vocab {
		if name == "" || id < 0 {
			return nil, fmt.Errorf("tokenizer: empty token or negative ID")
		}
		if old, ok := names[id]; ok {
			return nil, fmt.Errorf("tokenizer: duplicate ID %d for %q and %q", id, old, name)
		}
		names[id] = name
		t.tokens[id] = decodePiece(name, inverse)
		if id > t.maxID {
			t.maxID = id
		}
	}
	for i, r := range alphabet {
		id, ok := m.Vocab[string(r)]
		if !ok {
			return nil, fmt.Errorf("tokenizer: missing byte token for 0x%02x", i)
		}
		t.bytes[i] = id
	}
	addedNames := make(map[string]bool)
	for _, a := range s.Added {
		if a.Content == "" || a.ID == nil || *a.ID < 0 || a.SingleWord || a.LStrip || a.RStrip || a.Normalized == nil || *a.Normalized {
			return nil, fmt.Errorf("tokenizer: unsupported added token %q", a.Content)
		}
		addedID := *a.ID
		if addedNames[a.Content] {
			return nil, fmt.Errorf("tokenizer: duplicate added token %q", a.Content)
		}
		addedNames[a.Content] = true
		if old, ok := names[addedID]; ok && old != a.Content {
			return nil, fmt.Errorf("tokenizer: conflicting added token ID %d", addedID)
		}
		if id, ok := m.Vocab[a.Content]; ok && id != addedID {
			return nil, fmt.Errorf("tokenizer: conflicting added token %q", a.Content)
		}
		names[addedID] = a.Content
		t.tokens[addedID] = decodePiece(a.Content, inverse)
		t.special[addedID] = a.Special
		t.added = append(t.added, addedToken{a.Content, addedID})
		if addedID > t.maxID {
			t.maxID = addedID
		}
	}
	sortAdded(t.added)
	for rank, raw := range m.Merges {
		var parts []string
		if len(raw) > 0 && raw[0] == '"' {
			var s string
			if err := json.Unmarshal(raw, &s); err != nil {
				return nil, err
			}
			parts = strings.Split(s, " ")
		} else if err := json.Unmarshal(raw, &parts); err != nil {
			return nil, fmt.Errorf("tokenizer: merge %d: %w", rank, err)
		}
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			return nil, fmt.Errorf("tokenizer: invalid merge %d", rank)
		}
		left, lok := m.Vocab[parts[0]]
		right, rok := m.Vocab[parts[1]]
		joined, jok := m.Vocab[parts[0]+parts[1]]
		if !lok || !rok || !jok {
			return nil, fmt.Errorf("tokenizer: merge %d references missing vocabulary", rank)
		}
		key := pair{left, right}
		if _, ok := t.merges[key]; ok {
			return nil, fmt.Errorf("tokenizer: duplicate merge %d", rank)
		}
		t.merges[key] = merge{rank, joined}
	}
	if err := t.parsePostProcessor(s.PostProcessor, names); err != nil {
		return nil, err
	}
	return t, nil
}

type byteLevel struct {
	Type           string `json:"type"`
	AddPrefixSpace *bool  `json:"add_prefix_space"`
	TrimOffsets    bool   `json:"trim_offsets"`
	UseRegex       *bool  `json:"use_regex"`
}

func parsePreTokenizer(raw json.RawMessage) (splitKind, bool, error) {
	var kind struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &kind); err != nil {
		return 0, false, fmt.Errorf("tokenizer: invalid pre-tokenizer: %w", err)
	}
	if kind.Type == "ByteLevel" {
		var b byteLevel
		if err := strictJSON(raw, &b); err != nil {
			return 0, false, fmt.Errorf("tokenizer: byte-level pre-tokenizer: %w", err)
		}
		if b.AddPrefixSpace == nil {
			return 0, false, fmt.Errorf("tokenizer: missing add_prefix_space")
		}
		k := splitGPT2
		if b.UseRegex != nil && !*b.UseRegex {
			k = splitNone
		}
		return k, *b.AddPrefixSpace, nil
	}
	var seq struct {
		Type  string            `json:"type"`
		Parts []json.RawMessage `json:"pretokenizers"`
	}
	if err := strictJSON(raw, &seq); err != nil || seq.Type != "Sequence" || len(seq.Parts) != 2 {
		return 0, false, fmt.Errorf("tokenizer: unsupported pre-tokenizer")
	}
	var split struct {
		Type    string `json:"type"`
		Pattern struct {
			Regex string `json:"Regex"`
		} `json:"pattern"`
		Behavior string `json:"behavior"`
		Invert   bool   `json:"invert"`
	}
	if err := strictJSON(seq.Parts[0], &split); err != nil || split.Type != "Split" || split.Behavior != "Isolated" || split.Invert {
		return 0, false, fmt.Errorf("tokenizer: unsupported split operation")
	}
	var b byteLevel
	if err := strictJSON(seq.Parts[1], &b); err != nil || b.Type != "ByteLevel" || b.AddPrefixSpace == nil || *b.AddPrefixSpace || b.UseRegex == nil || *b.UseRegex {
		return 0, false, fmt.Errorf("tokenizer: unsupported byte-level split conversion")
	}
	switch split.Pattern.Regex {
	case qwenPattern:
		return splitQwen, false, nil
	case llamaPattern:
		return splitLlama, false, nil
	case gpt2Pattern:
		return splitGPT2, false, nil
	default:
		return 0, false, fmt.Errorf("tokenizer: unsupported split pattern %q", split.Pattern.Regex)
	}
}

func (t *Tokenizer) parsePostProcessor(raw json.RawMessage, names map[int32]string) error {
	if !present(raw) {
		return nil
	}
	var kind struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &kind); err != nil {
		return fmt.Errorf("tokenizer: invalid post-processor: %w", err)
	}
	if kind.Type == "ByteLevel" {
		var b byteLevel
		if err := strictJSON(raw, &b); err != nil {
			return fmt.Errorf("tokenizer: byte-level post-processor: %w", err)
		}
		return nil
	}
	var p struct {
		Type    string                       `json:"type"`
		Single  []map[string]json.RawMessage `json:"single"`
		Pair    json.RawMessage              `json:"pair"`
		Special map[string]struct {
			ID     string   `json:"id"`
			IDs    []int32  `json:"ids"`
			Tokens []string `json:"tokens"`
		} `json:"special_tokens"`
	}
	if err := strictJSON(raw, &p); err != nil || p.Type != "TemplateProcessing" {
		return fmt.Errorf("tokenizer: unsupported post-processor")
	}
	seen := false
	for _, part := range p.Single {
		if len(part) != 1 {
			return fmt.Errorf("tokenizer: invalid template element")
		}
		for kind, raw := range part {
			var value struct {
				ID     string `json:"id"`
				TypeID int    `json:"type_id"`
			}
			if err := strictJSON(raw, &value); err != nil {
				return fmt.Errorf("tokenizer: template element: %w", err)
			}
			switch kind {
			case "Sequence":
				if value.ID != "A" || seen {
					return fmt.Errorf("tokenizer: template must contain one input A")
				}
				seen = true
			case "SpecialToken":
				token, ok := p.Special[value.ID]
				if !ok || len(token.IDs) == 0 || len(token.IDs) != len(token.Tokens) || token.ID != value.ID {
					return fmt.Errorf("tokenizer: invalid template special token %q", value.ID)
				}
				for i, id := range token.IDs {
					if name, ok := names[id]; !ok || name != token.Tokens[i] {
						return fmt.Errorf("tokenizer: template token ID %d does not match vocabulary", id)
					}
				}
				if seen {
					t.suffix = append(t.suffix, token.IDs...)
				} else {
					t.prefix = append(t.prefix, token.IDs...)
				}
			default:
				return fmt.Errorf("tokenizer: unsupported template element %q", kind)
			}
		}
	}
	if !seen {
		return fmt.Errorf("tokenizer: template has no input A")
	}
	return nil
}

// uniqueJSON rejects duplicate object keys, including vocabulary and pipeline
// fields, rather than depending on whichever duplicate the decoder retains.
func uniqueJSON(d *json.Decoder, depth int) error {
	if depth > 256 {
		return fmt.Errorf("JSON nesting exceeds tokenizer schema")
	}
	token, err := d.Token()
	if err != nil {
		return err
	}
	switch token {
	case json.Delim('{'):
		seen := make(map[string]bool)
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok {
				return fmt.Errorf("invalid object key")
			}
			if seen[name] {
				return fmt.Errorf("duplicate key %q", name)
			}
			seen[name] = true
			if err := uniqueJSON(d, depth+1); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	case json.Delim('['):
		for d.More() {
			if err := uniqueJSON(d, depth+1); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	}
	return nil
}

// encoding/json replaces unpaired UTF-16 escapes with U+FFFD. Reject them so
// loading a malformed file cannot silently change an added token or vocabulary.
// The ordinary JSON decoder has already checked escape syntax before this call.
func validSurrogates(data []byte) error {
	for i := 0; i < len(data); i++ {
		if data[i] != '\\' {
			continue
		}
		i++
		if i >= len(data) || data[i] != 'u' {
			continue
		}
		if i+4 >= len(data) {
			return fmt.Errorf("incomplete Unicode escape")
		}
		r, err := strconv.ParseUint(string(data[i+1:i+5]), 16, 16)
		if err != nil {
			return err
		}
		i += 4
		if r >= 0xdc00 && r <= 0xdfff {
			return fmt.Errorf("unpaired low surrogate")
		}
		if r < 0xd800 || r > 0xdbff {
			continue
		}
		if i+6 >= len(data) || data[i+1] != '\\' || data[i+2] != 'u' {
			return fmt.Errorf("unpaired high surrogate")
		}
		low, err := strconv.ParseUint(string(data[i+3:i+7]), 16, 16)
		if err != nil || low < 0xdc00 || low > 0xdfff {
			return fmt.Errorf("unpaired high surrogate")
		}
		i += 6
	}
	return nil
}
