package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
)

// rendererCorpus is the on-disk format used by the host renderer gate. IDs are
// stored as signed little-endian int32 values when compared so that the gate
// does not depend on Go's JSON number representation.
type rendererCorpus struct {
	Version int            `json:"version"`
	Cases   []rendererCase `json:"cases"`
}

type rendererCase struct {
	Ordinal         int            `json:"ordinal"`
	Name            string         `json:"name"`
	OracleCaseID    string         `json:"oracle_case_id"`
	CandidateCaseID string         `json:"candidate_case_id"`
	Kind            string         `json:"kind"` // chat or prefix
	System          string         `json:"system,omitempty"`
	Prompt          string         `json:"prompt,omitempty"`
	Mode            string         `json:"mode"`
	Context         int            `json:"context,omitempty"`
	Turns           []rendererTurn `json:"turns,omitempty"`
	Want            []int          `json:"token_ids"`
	InputSHA256     string         `json:"rendered_input_sha256"`
	TokenSHA256     string         `json:"token_ids_sha256"`
}

type rendererTurn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (c rendererCorpus) render(t *tokenizer) ([][]int, error) {
	if c.Version != 1 {
		return nil, fmt.Errorf("renderer corpus: unsupported version %d", c.Version)
	}
	out := make([][]int, len(c.Cases))
	seen := make(map[string]bool, len(c.Cases))
	for i, tc := range c.Cases {
		if tc.Name == "" || seen[tc.Name] {
			return nil, fmt.Errorf("renderer corpus: duplicate or empty case name %q", tc.Name)
		}
		seen[tc.Name] = true
		mode, err := parseRendererMode(tc.Mode)
		if err != nil {
			return nil, fmt.Errorf("renderer corpus case %q: %w", tc.Name, err)
		}
		switch tc.Kind {
		case "chat":
			out[i], err = t.RenderChat(tc.System, tc.Prompt, mode, tc.Context)
		case "prefix":
			turns := make([]prefixTurn, len(tc.Turns))
			for j, turn := range tc.Turns {
				role := prefixUser
				switch turn.Role {
				case "USER":
				case "ASSISTANT":
					role = prefixAssistant
				default:
					return nil, fmt.Errorf("renderer corpus case %q: unknown role %q", tc.Name, turn.Role)
				}
				turns[j] = prefixTurn{role: role, content: turn.Content}
			}
			out[i], err = t.RenderPrefix(tc.System, turns, mode)
		default:
			return nil, fmt.Errorf("renderer corpus case %q: unknown kind %q", tc.Name, tc.Kind)
		}
		if err != nil {
			return nil, fmt.Errorf("renderer corpus case %q: %w", tc.Name, err)
		}
	}
	return out, nil
}

func parseRendererMode(s string) (ThinkMode, error) {
	switch s {
	case "none":
		return ThinkNone, nil
	case "high":
		return ThinkHigh, nil
	case "max":
		return ThinkMax, nil
	default:
		return 0, fmt.Errorf("unknown mode %q", s)
	}
}

func (c rendererCorpus) Validate(t *tokenizer) error {
	if len(c.Cases) == 0 {
		return fmt.Errorf("renderer corpus: no cases")
	}
	got, err := c.render(t)
	if err != nil {
		return err
	}
	for i, want := range c.Cases {
		if want.Ordinal != i {
			return fmt.Errorf("renderer corpus case %q: ordinal %d, want %d", want.Name, want.Ordinal, i)
		}
		if want.OracleCaseID == "" || want.CandidateCaseID == "" {
			return fmt.Errorf("renderer corpus case %q: missing oracle or candidate case ID", want.Name)
		}
		if want.OracleCaseID != want.CandidateCaseID {
			return fmt.Errorf("renderer corpus case %q: oracle/candidate case IDs differ", want.Name)
		}
		inputHash, err := rendererInputHash(want)
		if err != nil {
			return fmt.Errorf("renderer corpus case %q: input hash: %w", want.Name, err)
		}
		if want.InputSHA256 == "" || want.InputSHA256 != inputHash {
			return fmt.Errorf("renderer corpus case %q: rendered-input hash mismatch", want.Name)
		}
		tokenHash := rendererTokenHash(got[i])
		if want.TokenSHA256 == "" || want.TokenSHA256 != tokenHash {
			return fmt.Errorf("renderer corpus case %q: token hash mismatch", want.Name)
		}
		if !bytes.Equal(rendererIDs(got[i]), rendererIDs(want.Want)) {
			return fmt.Errorf("renderer corpus case %q: token IDs differ", want.Name)
		}
	}
	return nil
}

func rendererInputHash(c rendererCase) (string, error) {
	b, err := json.Marshal(struct {
		Name    string         `json:"name"`
		Kind    string         `json:"kind"`
		System  string         `json:"system,omitempty"`
		Prompt  string         `json:"prompt,omitempty"`
		Mode    string         `json:"mode"`
		Context int            `json:"context,omitempty"`
		Turns   []rendererTurn `json:"turns,omitempty"`
	}{c.Name, c.Kind, c.System, c.Prompt, c.Mode, c.Context, c.Turns})
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(b)
	return fmt.Sprintf("%x", h[:]), nil
}

func rendererTokenHash(ids []int) string {
	h := sha256.Sum256(rendererIDs(ids))
	return fmt.Sprintf("%x", h[:])
}

func rendererIDs(ids []int) []byte {
	b := make([]byte, 4*len(ids))
	for i, id := range ids {
		binary.LittleEndian.PutUint32(b[i*4:], uint32(int32(id)))
	}
	return b
}

func readRendererCorpus(name string) (rendererCorpus, error) {
	data, err := os.ReadFile(name)
	if err != nil {
		return rendererCorpus{}, fmt.Errorf("read renderer corpus: %w", err)
	}
	var c rendererCorpus
	if err := json.Unmarshal(data, &c); err != nil {
		return rendererCorpus{}, fmt.Errorf("decode renderer corpus: %w", err)
	}
	return c, nil
}

func writeRendererCorpus(name string, c rendererCorpus) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("encode renderer corpus: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(name, data, 0o644); err != nil {
		return fmt.Errorf("write renderer corpus: %w", err)
	}
	return nil
}
