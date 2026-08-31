package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
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

	// Add special/added tokens
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

// Encode converts a string prompt into an array of token IDs.
func (tok *BpeTokenizer) Encode(text string) ([]int32, error) {
	if len(tok.merges) == 0 {
		// Fast-path byte tokenizer
		bytes := []byte(text)
		tokens := make([]int32, len(bytes))
		for i, b := range bytes {
			if id, ok := tok.vocab[string([]byte{b})]; ok {
				tokens[i] = id
			} else {
				tokens[i] = int32(b)
			}
		}
		return tokens, nil
	}

	// Split by space/words for BPE
	words := strings.Fields(text)
	var tokens []int32

	for _, w := range words {
		// Initial character pieces
		runes := []rune(w)
		pieces := make([]string, len(runes))
		for i, r := range runes {
			pieces[i] = string(r)
		}

		// Iteratively merge BPE pairs with lowest rank
		for len(pieces) > 1 {
			bestRank := 1 << 30
			bestIdx := -1

			for i := 0; i < len(pieces)-1; i++ {
				pair := pieces[i] + " " + pieces[i+1]
				if rank, exists := tok.merges[pair]; exists {
					if rank < bestRank {
						bestRank = rank
						bestIdx = i
					}
				}
			}

			if bestIdx == -1 {
				break
			}

			// Merge best pair
			newPieces := make([]string, 0, len(pieces)-1)
			newPieces = append(newPieces, pieces[:bestIdx]...)
			newPieces = append(newPieces, pieces[bestIdx]+pieces[bestIdx+1])
			newPieces = append(newPieces, pieces[bestIdx+2:]...)
			pieces = newPieces
		}

		for _, p := range pieces {
			if id, ok := tok.vocab[p]; ok {
				tokens = append(tokens, id)
			} else {
				// Byte fallback
				for _, b := range []byte(p) {
					hexKey := fmt.Sprintf("<0x%02X>", b)
					if id, ok := tok.vocab[hexKey]; ok {
						tokens = append(tokens, id)
					} else {
						tokens = append(tokens, int32(b))
					}
				}
			}
		}
	}

	return tokens, nil
}

// Decode converts token IDs back into a reconstructed text string.
func (tok *BpeTokenizer) Decode(tokens []int32) (string, error) {
	var sb strings.Builder
	for _, id := range tokens {
		if s, ok := tok.invVocab[id]; ok {
			// Handle byte fallback <0xNN>
			if strings.HasPrefix(s, "<0x") && strings.HasSuffix(s, ">") && len(s) == 6 {
				if b, err := strconv.ParseUint(s[3:5], 16, 8); err == nil {
					sb.WriteByte(byte(b))
					continue
				}
			}
			// Replace sentencepiece   prefix with space, and GPT-2 Ġ with space, Ċ with newline
			clean := strings.ReplaceAll(s, " ", " ")
			clean = strings.ReplaceAll(clean, "Ġ", " ")
			clean = strings.ReplaceAll(clean, "Ċ", "\n")
			sb.WriteString(clean)
		} else if id >= 0 && id < 256 {
			sb.WriteByte(byte(id))
		} else {
			sb.WriteString(fmt.Sprintf("<unk:%d>", id))
		}
	}
	return sb.String(), nil
}


// VocabSize returns the total vocabulary size.
func (tok *BpeTokenizer) VocabSize() int {
	return len(tok.vocab)
}

// SpecialTokens returns sorted special token IDs.
func (tok *BpeTokenizer) SpecialTokens() []int32 {
	var res []int32
	for _, id := range tok.vocab {
		res = append(res, id)
	}
	sort.Slice(res, func(i, j int) bool { return res[i] < res[j] })
	return res
}
