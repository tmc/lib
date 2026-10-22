package dsv4eval

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"math"
	"os"
	"path/filepath"
)

// DecisionArtifact records a real selection before the next model operation.
// LogitsSHA256 hashes the entire float32 little-endian vector before Eval.
// Accepted is false for a stop token that was selected but never consumed.
type DecisionArtifact struct {
	Step                int    `json:"step"`
	Greedy              int    `json:"greedy"`
	Chosen              int    `json:"chosen"`
	Forced              string `json:"forced"`
	Stop                bool   `json:"stop"`
	Accepted            bool   `json:"accepted"`
	BeforeHistorySHA256 string `json:"before_history_sha256"`
	LogitsSHA256        string `json:"logits_sha256"`
}

// CaseArtifacts is a diagnostic transcript, not a numerical parity verdict.
// Prompt and Generated contain only tokens actually installed or accepted.
// TerminalLogits describe their complete History, not a selected stop token.
type CaseArtifacts struct {
	Index           int                `json:"index"`
	Source          string             `json:"source"`
	ID              string             `json:"id"`
	Build           string             `json:"build"`
	ModelSHA256     string             `json:"model_sha256"`
	Context         int                `json:"ctx"`
	Budget          int                `json:"generation_budget"`
	Prompt          []int              `json:"prompt_ids"`
	Generated       []int              `json:"generated_ids"`
	History         []int              `json:"history_ids"`
	HistorySHA256   string             `json:"history_sha256"`
	Decisions       []DecisionArtifact `json:"decisions"`
	Status          string             `json:"status"`
	ThinkClose      string             `json:"think_close"`
	StopDisposition string             `json:"stop_disposition"`
	TerminalLogits  []float32          `json:"-"`
}

func tokenHashWrite(h hash.Hash, id int) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(int32(id)))
	h.Write(b[:])
}

func logitsHash(logits []float32) string {
	h := sha256.New()
	var b [4]byte
	for _, v := range logits {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		h.Write(b[:])
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

// ArtifactRecorder writes exclusive per-case manifests and terminal logits.
// Dir must be a caller-created, empty external output directory. Existing
// case directories are errors, never overwritten or interpreted as results.
// The terminal rows are diagnostics; decision hashes do not replace raw
// per-step vectors required for a numerical cross-implementation comparison.
type ArtifactRecorder struct{ Dir string }

// Record retains actual case artifacts even when its terminal status is
// INCOMPLETE. A write error prevents the caller from publishing a full trace.
func (r ArtifactRecorder) Record(c CaseArtifacts) error {
	if r.Dir == "" {
		return fmt.Errorf("record eval artifacts: missing directory or case data")
	}
	if err := validateArtifacts(c); err != nil {
		return fmt.Errorf("record eval artifacts: %w", err)
	}
	dir := filepath.Join(r.Dir, fmt.Sprintf("case-%02d", c.Index))
	if err := os.Mkdir(dir, 0700); err != nil {
		return fmt.Errorf("record eval artifacts: %w", err)
	}
	path := filepath.Join(dir, "terminal.f32")
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	for _, v := range c.TerminalLogits {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		if _, err := w.Write(b[:]); err != nil {
			return errors.Join(err, f.Close())
		}
	}
	if err := w.Flush(); err != nil {
		return errors.Join(err, f.Close())
	}
	if err := f.Close(); err != nil {
		return err
	}
	f, err = os.OpenFile(filepath.Join(dir, "manifest.json"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	manifest := struct {
		CaseArtifacts
		TerminalLogitsSHA256 string `json:"terminal_logits_sha256"`
		Vocab                int    `json:"vocab"`
		Temperature          int    `json:"temperature"`
		Think                string `json:"think"`
		TokenEncoding        string `json:"token_encoding"`
	}{c, logitsHash(c.TerminalLogits), len(c.TerminalLogits), 0, "HIGH", "int32-le"}
	err = json.NewEncoder(f).Encode(manifest)
	closeErr := f.Close()
	return errors.Join(err, closeErr)
}

func validateArtifacts(c CaseArtifacts) error {
	if c.Index < 1 || c.Index > 12 || !hardSmokeCases[c.Source+"\x00"+c.ID] || c.Build == "" || !validSHA(c.ModelSHA256) || c.Context <= 0 || c.Budget <= 0 || len(c.Prompt) == 0 || len(c.Prompt)+c.Budget > c.Context || len(c.Generated) > c.Budget || len(c.TerminalLogits) == 0 {
		return fmt.Errorf("invalid case provenance or shape")
	}
	if _, err := greedyFinite(c.TerminalLogits); err != nil {
		return err
	}
	if len(c.History) != len(c.Prompt)+len(c.Generated) {
		return fmt.Errorf("history length differs")
	}
	h := sha256.New()
	for i, id := range append(append([]int(nil), c.Prompt...), c.Generated...) {
		if id < 0 || id >= len(c.TerminalLogits) || c.History[i] != id {
			return fmt.Errorf("history tokens differ or outside vocabulary")
		}
		tokenHashWrite(h, id)
	}
	if c.HistorySHA256 != fmt.Sprintf("%x", h.Sum(nil)) {
		return fmt.Errorf("history hash differs")
	}
	h = sha256.New()
	for _, id := range c.Prompt {
		tokenHashWrite(h, id)
	}
	accepted, stopped := 0, false
	for i, d := range c.Decisions {
		if d.Step != i+1 || d.Greedy < 0 || d.Greedy >= len(c.TerminalLogits) || d.Chosen < 0 || d.Chosen >= len(c.TerminalLogits) || !validSHA(d.LogitsSHA256) || d.BeforeHistorySHA256 != fmt.Sprintf("%x", h.Sum(nil)) {
			return fmt.Errorf("invalid decision provenance")
		}
		if d.Forced != "none" && d.Forced != "soft_forced" && d.Forced != "hard_forced" {
			return fmt.Errorf("invalid forced kind")
		}
		if d.Stop {
			if d.Accepted || i != len(c.Decisions)-1 {
				return fmt.Errorf("stop accepted or not terminal")
			}
			stopped = true
			continue
		}
		if !d.Accepted || accepted >= len(c.Generated) || c.Generated[accepted] != d.Chosen {
			return fmt.Errorf("accepted decisions differ")
		}
		accepted++
		tokenHashWrite(h, d.Chosen)
	}
	if accepted != len(c.Generated) {
		return fmt.Errorf("accepted token count differs")
	}
	if c.Status != "PASSED" && c.Status != "FAILED" && c.Status != "INCOMPLETE" {
		return fmt.Errorf("invalid terminal status")
	}
	if c.ThinkClose != "none" && c.ThinkClose != "natural" && c.ThinkClose != "hard_forced" && c.ThinkClose != "soft_forced" {
		return fmt.Errorf("invalid thinking close")
	}
	if stopped {
		if c.StopDisposition != "selected_stop_not_consumed" {
			return fmt.Errorf("stop disposition differs")
		}
	} else if c.StopDisposition != "budget_exhausted" || accepted != c.Budget {
		return fmt.Errorf("budget disposition differs")
	}
	return nil
}
