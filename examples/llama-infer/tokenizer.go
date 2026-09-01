package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Tokenizer represents an interface for encoding text and decoding token arrays.
type Tokenizer interface {
	Encode(text string) ([]int32, error)
	Decode(tokens []int32) (string, error)
}

// BpeTokenizer implements a Byte-Pair Encoding tokenizer compatible with Hugging Face tokenizer.json.
type BpeTokenizer struct {
	vocab      map[string]int32
	invVocab   map[int32]string
	merges     map[string]int
	byteTokens map[byte]string
}

type hfTokenizerSchema struct {
	Model struct {
		Vocab  map[string]int32 `json:"vocab"`
		Merges []string         `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
}

// LoadHFTokenizer parses a Hugging Face tokenizer.json file.
func LoadHFTokenizer(path string) (*BpeTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read file: %w", err)
	}

	var schema hfTokenizerSchema
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, fmt.Errorf("tokenizer: unmarshal json: %w", err)
	}

	vocab := schema.Model.Vocab
	if vocab == nil {
		vocab = make(map[string]int32)
	}
	invVocab := make(map[int32]string, len(vocab))
	for k, v := range vocab {
		invVocab[v] = k
	}

	for _, at := range schema.AddedTokens {
		vocab[at.Content] = at.ID
		invVocab[at.ID] = at.Content
	}

	merges := make(map[string]int, len(schema.Model.Merges))
	for rank, m := range schema.Model.Merges {
		merges[m] = rank
	}

	byteTokens := make(map[byte]string, 256)
	for b := 0; b < 256; b++ {
		byteTokens[byte(b)] = fmt.Sprintf("<0x%02X>", b)
	}

	return &BpeTokenizer{
		vocab:      vocab,
		invVocab:   invVocab,
		merges:     merges,
		byteTokens: byteTokens,
	}, nil
}

// NewByteTokenizer creates a default byte-level fallback tokenizer mapping raw bytes 0..255.
func NewByteTokenizer() *BpeTokenizer {
	vocab := make(map[string]int32, 256)
	invVocab := make(map[int32]string, 256)
	byteTokens := make(map[byte]string, 256)

	for b := 0; b < 256; b++ {
		str := string([]byte{byte(b)})
		vocab[str] = int32(b)
		invVocab[int32(b)] = str
		byteTokens[byte(b)] = str
	}

	return &BpeTokenizer{
		vocab:      vocab,
		invVocab:   invVocab,
		merges:     make(map[string]int),
		byteTokens: byteTokens,
	}
}

// Encode encodes a text string into BPE token IDs.
func (t *BpeTokenizer) Encode(text string) ([]int32, error) {
	if text == "" {
		return nil, nil
	}

	var tokens []int32
	var word []string

	for i := 0; i < len(text); i++ {
		b := text[i]
		s, ok := t.byteTokens[b]
		if !ok {
			s = string([]byte{b})
		}
		word = append(word, s)
	}

	word = t.bpeMerge(word)

	for _, piece := range word {
		if id, ok := t.vocab[piece]; ok {
			tokens = append(tokens, id)
		} else {
			for i := 0; i < len(piece); i++ {
				if id, ok := t.vocab[string([]byte{piece[i]})]; ok {
					tokens = append(tokens, id)
				}
			}
		}
	}

	return tokens, nil
}

// Decode decodes an array of token IDs back into a UTF-8 text string.
func (t *BpeTokenizer) Decode(tokens []int32) (string, error) {
	var sb strings.Builder
	for _, id := range tokens {
		piece, ok := t.invVocab[id]
		if !ok {
			continue
		}
		if strings.HasPrefix(piece, "<0x") && strings.HasSuffix(piece, ">") && len(piece) == 6 {
			hexStr := piece[3:5]
			if val, err := strconv.ParseUint(hexStr, 16, 8); err == nil {
				sb.WriteByte(byte(val))
				continue
			}
		}
		piece = strings.ReplaceAll(piece, "Ġ", " ")
		piece = strings.ReplaceAll(piece, " ", " ")
		sb.WriteString(piece)
	}

	raw := sb.String()
	if !utf8.ValidString(raw) {
		return strings.ToValidUTF8(raw, ""), nil
	}
	return raw, nil
}

func (t *BpeTokenizer) bpeMerge(word []string) []string {
	if len(word) <= 1 {
		return word
	}

	for {
		bestRank := -1
		bestIdx := -1

		for i := 0; i < len(word)-1; i++ {
			pair := word[i] + " " + word[i+1]
			if rank, ok := t.merges[pair]; ok {
				if bestRank == -1 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}

		if bestIdx == -1 {
			break
		}

		merged := word[bestIdx] + word[bestIdx+1]
		newWord := make([]string, 0, len(word)-1)
		newWord = append(newWord, word[:bestIdx]...)
		newWord = append(newWord, merged)
		newWord = append(newWord, word[bestIdx+2:]...)
		word = newWord
	}

	return word
}
