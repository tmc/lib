package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Tokenizer defines text encoding and token decoding.
type Tokenizer interface {
	Encode(text string) ([]int32, error)
	Decode(tokens []int32) (string, error)
	VocabSize() int
}

// BpeTokenizer implements a Byte-Pair Encoding tokenizer for HuggingFace models.
type BpeTokenizer struct {
	vocab      map[string]int32
	invVocab   map[int32]string
	merges     map[string]int
	byteTokens map[byte]string
	bosID      int32
	eosID      int32
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
		return nil, fmt.Errorf("read tokenizer: %w", err)
	}

	var schema hfTokenizerSchema
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, fmt.Errorf("unmarshal tokenizer: %w", err)
	}

	vocab := schema.Model.Vocab
	if vocab == nil {
		vocab = make(map[string]int32)
	}
	invVocab := make(map[int32]string, len(vocab)+len(schema.AddedTokens))
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

	bos := int32(-1)
	if id, ok := vocab["<|im_start|>"]; ok {
		bos = id
	} else if id, ok := vocab["<s>"]; ok {
		bos = id
	}

	eos := int32(-1)
	if id, ok := vocab["<|im_end|>"]; ok {
		eos = id
	} else if id, ok := vocab["<|endoftext|>"]; ok {
		eos = id
	} else if id, ok := vocab["</s>"]; ok {
		eos = id
	}

	return &BpeTokenizer{
		vocab:      vocab,
		invVocab:   invVocab,
		merges:     merges,
		byteTokens: byteTokens,
		bosID:      bos,
		eosID:      eos,
	}, nil
}

// Encode converts text into token IDs.
func (tok *BpeTokenizer) Encode(text string) ([]int32, error) {
	var tokens []int32

	// Check prefix convention
	spPrefix := ""
	if _, ok := tok.vocab["\u2581The"]; ok {
		spPrefix = "\u2581"
	} else if _, ok := tok.vocab["ĠThe"]; ok {
		spPrefix = "Ġ"
	}

	words := strings.Fields(text)
	for i, w := range words {
		if spPrefix != "" {
			w = spPrefix + w
		} else if i > 0 {
			w = " " + w
		}

		runes := []rune(w)
		pieces := make([]string, len(runes))
		for j, r := range runes {
			pieces[j] = string(r)
		}

		for len(pieces) > 1 {
			bestRank := 1 << 30
			bestIdx := -1
			for j := 0; j < len(pieces)-1; j++ {
				pair := pieces[j] + " " + pieces[j+1]
				if rank, exists := tok.merges[pair]; exists {
					if rank < bestRank {
						bestRank = rank
						bestIdx = j
					}
				}
			}
			if bestIdx == -1 {
				break
			}
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

// Decode converts token IDs back into string.
func (tok *BpeTokenizer) Decode(tokens []int32) (string, error) {
	var sb strings.Builder
	for _, id := range tokens {
		if s, ok := tok.invVocab[id]; ok {
			if s == "<|im_start|>" || s == "<|im_end|>" || s == "<|endoftext|>" || s == "<s>" || s == "</s>" {
				continue
			}
			if strings.HasPrefix(s, "<0x") && strings.HasSuffix(s, ">") && len(s) == 6 {
				if b, err := strconv.ParseUint(s[3:5], 16, 8); err == nil {
					sb.WriteByte(byte(b))
					continue
				}
			}
			clean := strings.ReplaceAll(s, "\u2581", " ")
			clean = strings.ReplaceAll(clean, " ", " ")
			clean = strings.ReplaceAll(clean, "Ġ", " ")
			clean = strings.ReplaceAll(clean, "Ċ", "\n")
			sb.WriteString(clean)
		} else if id >= 0 && id < 256 {
			sb.WriteByte(byte(id))
		}
	}
	return sb.String(), nil
}

func (tok *BpeTokenizer) VocabSize() int {
	return len(tok.vocab)
}
