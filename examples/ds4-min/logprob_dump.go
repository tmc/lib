package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

type logprobStep struct {
	Step     int                  `json:"step"`
	Selected int                  `json:"selected"`
	Scores   []sampler.TokenScore `json:"top_logprobs"`
}

type logprobToken struct {
	ID    int    `json:"id"`
	Text  string `json:"text"`
	Bytes []int  `json:"bytes"`
}

type logprobJSONScore struct {
	Token   logprobToken `json:"token"`
	Logit   float32      `json:"logit"`
	Logprob float32      `json:"logprob"`
}

type logprobJSONStep struct {
	Step        int                `json:"step"`
	Selected    logprobToken       `json:"selected"`
	TopLogprobs []logprobJSONScore `json:"top_logprobs"`
}

type logprobDump struct {
	Source       string            `json:"source"`
	PromptTokens int               `json:"prompt_tokens"`
	Context      int               `json:"ctx"`
	TopK         int               `json:"top_k"`
	Steps        []logprobJSONStep `json:"steps"`
}

func dumpDecodeLogits(dir string, step int, logits []float32) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create decode logits directory: %w", err)
	}
	data := make([]byte, 4*len(logits))
	for i, value := range logits {
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(value))
	}
	path := filepath.Join(dir, fmt.Sprintf("step-%04d.f32", step))
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("write decode logits: %w", err)
	}
	return nil
}

func writeLogprobs(path string, promptTokens, context, topK int, steps []logprobStep, tok *tokenizer) error {
	if tok == nil {
		return fmt.Errorf("encode logprobs: tokenizer is nil")
	}
	if topK <= 0 {
		return fmt.Errorf("encode logprobs: top-k must be positive")
	}
	if topK > 128 {
		topK = 128
	}
	jsonSteps := make([]logprobJSONStep, 0, len(steps))
	for _, step := range steps {
		selected, err := logprobTokenFor(tok, step.Selected)
		if err != nil {
			return fmt.Errorf("encode logprobs step %d: %w", step.Step, err)
		}
		scores := make([]logprobJSONScore, len(step.Scores))
		for i, score := range step.Scores {
			token, err := logprobTokenFor(tok, score.ID)
			if err != nil {
				return fmt.Errorf("encode logprobs step %d score %d: %w", step.Step, i, err)
			}
			scores[i] = logprobJSONScore{Token: token, Logit: score.Logit, Logprob: score.Logprob}
		}
		jsonSteps = append(jsonSteps, logprobJSONStep{Step: step.Step, Selected: selected, TopLogprobs: scores})
	}
	dump := logprobDump{Source: "ds4-min", PromptTokens: promptTokens, Context: context, TopK: topK, Steps: jsonSteps}
	data, err := json.MarshalIndent(dump, "", "  ")
	if err != nil {
		return fmt.Errorf("encode logprobs: %w", err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o600); err != nil {
		return fmt.Errorf("write logprobs: %w", err)
	}
	return nil
}

func logprobTokenFor(tok *tokenizer, id int) (logprobToken, error) {
	if id < 0 || id >= len(tok.tokens) {
		return logprobToken{}, fmt.Errorf("token id %d out of range", id)
	}
	decoded, err := tok.Decode([]int{id})
	if err != nil {
		return logprobToken{}, err
	}
	bytes := make([]int, len(decoded))
	for i, b := range decoded {
		bytes[i] = int(b)
	}
	return logprobToken{ID: id, Text: tok.tokens[id], Bytes: bytes}, nil
}
