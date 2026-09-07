package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/x/hftokenizer"
)

// loadPJRTTokenizer validates the pipeline before it can produce model inputs.
func loadPJRTTokenizer(dir string, vocab int) (*hftokenizer.Tokenizer, error) {
	path := filepath.Join(dir, "tokenizer.json")
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("pjrt tokenizer: %w", err)
	}
	tok, err := hftokenizer.New(f)
	if err = errors.Join(err, f.Close()); err != nil {
		return nil, fmt.Errorf("pjrt tokenizer %s: %w", path, err)
	}
	if int64(tok.MaxTokenID()) >= int64(vocab) {
		return nil, fmt.Errorf("pjrt tokenizer: maximum token ID %d exceeds vocabulary size %d", tok.MaxTokenID(), vocab)
	}
	return tok, nil
}

// preflightPJRTText checks text assets before loading weights or opening PJRT.
func preflightPJRTText(dir string) (*hftokenizer.Tokenizer, error) {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("pjrt config: %w", err)
	}
	p, err := readPJRTConfig(data)
	if err != nil {
		return nil, err
	}
	return loadPJRTTokenizer(dir, p.config.VocabSize)
}

func (p *pjrtModel) generateText(ctx context.Context, tok *hftokenizer.Tokenizer, prompt string, maxTokens int, eos []int32, emit func(int32, string) bool) error {
	if emit == nil {
		return fmt.Errorf("pjrt: token callback is required")
	}
	ids, err := tok.Encode(prompt)
	if err != nil {
		return err
	}
	dec := tok.NewDecoder()
	var decodeErr error
	err = p.generateFinal(ctx, ids, maxTokens, eos, func(id int32, final bool) bool {
		text, err := dec.Decode([]int32{id}, final)
		if err != nil {
			decodeErr = err
			return false
		}
		return emit(id, text)
	})
	return errors.Join(err, decodeErr)
}
