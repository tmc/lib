package dsv4bench

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
)

// FrontierLogits is the JSON record written for one context frontier.
type FrontierLogits struct {
	Source         string    `json:"source"`
	Model          string    `json:"model"`
	Backend        string    `json:"backend"`
	Quality        string    `json:"quality"`
	QuantBits      int       `json:"quant_bits"`
	PromptTokens   int       `json:"prompt_tokens"`
	FrontierTokens int       `json:"frontier_tokens"`
	Context        int       `json:"ctx"`
	Vocab          int       `json:"vocab"`
	ArgmaxID       int       `json:"argmax_id"`
	ArgmaxLogit    float32   `json:"argmax_logit"`
	Logits         []float32 `json:"logits"`
}

// WriteFrontierJSON validates and writes one deterministic JSON record.
func WriteFrontierJSON(w io.Writer, f FrontierLogits) error {
	if err := validateFrontierLogits(f); err != nil {
		return err
	}
	data, err := json.Marshal(f)
	if err != nil {
		return fmt.Errorf("marshal frontier logits: %w", err)
	}
	if _, err := w.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("write frontier logits: %w", err)
	}
	return nil
}

// ReadFrontierJSON reads one or more newline-delimited frontier records.
// Records are validated individually and extra JSON fields are rejected so a
// comparison cannot silently ignore a changed artifact schema.
func ReadFrontierJSON(r io.Reader) ([]FrontierLogits, error) {
	if r == nil {
		return nil, fmt.Errorf("read frontier logits: nil reader")
	}
	s := bufio.NewScanner(r)
	// Logit rows can be large; keep a bounded but generous line limit.
	s.Buffer(make([]byte, 64<<10), 128<<20)
	var rows []FrontierLogits
	line := 0
	for s.Scan() {
		line++
		if len(s.Bytes()) == 0 {
			return nil, fmt.Errorf("read frontier logits line %d: empty record", line)
		}
		dec := json.NewDecoder(bytes.NewReader(s.Bytes()))
		dec.DisallowUnknownFields()
		var row FrontierLogits
		if err := dec.Decode(&row); err != nil {
			return nil, fmt.Errorf("read frontier logits line %d: %w", line, err)
		}
		var extra any
		if err := dec.Decode(&extra); err != io.EOF {
			if err == nil {
				return nil, fmt.Errorf("read frontier logits line %d: trailing JSON", line)
			}
			return nil, fmt.Errorf("read frontier logits line %d: trailing data: %w", line, err)
		}
		if err := validateFrontierLogits(row); err != nil {
			return nil, fmt.Errorf("read frontier logits line %d: %w", line, err)
		}
		rows = append(rows, row)
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("read frontier logits: %w", err)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("read frontier logits: no records")
	}
	return rows, nil
}

// FrontierTolerance controls the F5-style finite-logit comparison.
// A value is within tolerance when abs(delta) <= Abs + Rel*max(abs(a),abs(b)).
type FrontierTolerance struct {
	Abs float64
	Rel float64
}

// FrontierRecordComparison summarizes one aligned frontier.
type FrontierRecordComparison struct {
	FrontierTokens int     `json:"frontier_tokens"`
	MaxAbs         float64 `json:"max_abs"`
	MaxRel         float64 `json:"max_rel"`
	Top1Match      bool    `json:"top1_match"`
	Within         bool    `json:"within"`
}

// FrontierComparison summarizes a cross-implementation frontier comparison.
// Source and backend labels may differ; model, quantization, prompt, context,
// vocabulary, and frontier identity must match exactly.
type FrontierComparison struct {
	Records         []FrontierRecordComparison `json:"records"`
	Compared        int                        `json:"compared"`
	Top1Matches     int                        `json:"top1_matches"`
	WithinTolerance int                        `json:"within_tolerance"`
	FirstFailure    int                        `json:"first_failure"`
	MaxAbs          float64                    `json:"max_abs"`
	MaxRel          float64                    `json:"max_rel"`
	Tolerance       FrontierTolerance          `json:"tolerance"`
}

// Pass reports whether every aligned frontier has matching argmax and finite
// logits within the supplied tolerance.
func (c FrontierComparison) Pass() bool {
	return c.Compared > 0 && c.Top1Matches == c.Compared && c.WithinTolerance == c.Compared
}

// CompareFrontier compares two validated frontier captures. It rejects
// reordered, missing, duplicated, or differently shaped frontiers instead of
// silently pairing records by position.
func CompareFrontier(base, target []FrontierLogits, tolerance FrontierTolerance) (FrontierComparison, error) {
	if tolerance.Abs < 0 || tolerance.Rel < 0 || math.IsNaN(tolerance.Abs) || math.IsNaN(tolerance.Rel) {
		return FrontierComparison{}, fmt.Errorf("compare frontier: invalid tolerance")
	}
	if len(base) == 0 || len(target) == 0 {
		return FrontierComparison{}, fmt.Errorf("compare frontier: empty input")
	}
	if len(base) != len(target) {
		return FrontierComparison{}, fmt.Errorf("compare frontier: %d records versus %d", len(base), len(target))
	}
	comparison := FrontierComparison{FirstFailure: -1, Tolerance: tolerance}
	for i := range base {
		if err := validateFrontierLogits(base[i]); err != nil {
			return FrontierComparison{}, fmt.Errorf("compare frontier base row %d: %w", i, err)
		}
		if err := validateFrontierLogits(target[i]); err != nil {
			return FrontierComparison{}, fmt.Errorf("compare frontier target row %d: %w", i, err)
		}
		if !sameFrontierWorkload(base[i], target[i]) {
			return FrontierComparison{}, fmt.Errorf("compare frontier row %d: workload differs", i)
		}
		row := FrontierRecordComparison{FrontierTokens: base[i].FrontierTokens, Within: true}
		row.Top1Match = base[i].ArgmaxID == target[i].ArgmaxID
		if row.Top1Match {
			comparison.Top1Matches++
		}
		if len(base[i].Logits) != len(target[i].Logits) {
			return FrontierComparison{}, fmt.Errorf("compare frontier row %d: vocabulary differs", i)
		}
		for j, left := range base[i].Logits {
			right := target[i].Logits[j]
			delta := math.Abs(float64(left) - float64(right))
			if delta > row.MaxAbs {
				row.MaxAbs = delta
			}
			denom := math.Max(math.Abs(float64(left)), math.Abs(float64(right)))
			if denom == 0 {
				denom = 1
			}
			if rel := delta / denom; rel > row.MaxRel {
				row.MaxRel = rel
			}
			if delta > tolerance.Abs+tolerance.Rel*denom {
				row.Within = false
			}
		}
		comparison.Compared++
		if row.Within {
			comparison.WithinTolerance++
		} else if comparison.FirstFailure < 0 {
			comparison.FirstFailure = i
		}
		if row.MaxAbs > comparison.MaxAbs {
			comparison.MaxAbs = row.MaxAbs
		}
		if row.MaxRel > comparison.MaxRel {
			comparison.MaxRel = row.MaxRel
		}
		comparison.Records = append(comparison.Records, row)
	}
	return comparison, nil
}

func sameFrontierWorkload(a, b FrontierLogits) bool {
	return a.Model == b.Model && a.Quality == b.Quality && a.QuantBits == b.QuantBits &&
		a.PromptTokens == b.PromptTokens && a.FrontierTokens == b.FrontierTokens &&
		a.Context == b.Context && a.Vocab == b.Vocab
}

func validateFrontierLogits(f FrontierLogits) error {
	if f.Source == "" || f.Model == "" || f.Backend == "" || f.Quality == "" {
		return fmt.Errorf("frontier logits: source, model, backend, and quality are required")
	}
	if f.QuantBits <= 0 || f.PromptTokens <= 0 || f.FrontierTokens <= 0 || f.Context <= 0 {
		return fmt.Errorf("frontier logits: quant_bits and token/context counts must be positive")
	}
	if f.Vocab <= 0 || len(f.Logits) != f.Vocab {
		return fmt.Errorf("frontier logits: vocab %d does not match logits length %d", f.Vocab, len(f.Logits))
	}
	if f.ArgmaxID < 0 || f.ArgmaxID >= f.Vocab || !finite(float64(f.ArgmaxLogit)) {
		return fmt.Errorf("frontier logits: invalid argmax")
	}
	for i, v := range f.Logits {
		if !finite(float64(v)) {
			return fmt.Errorf("frontier logits: non-finite logit at index %d", i)
		}
		if v > f.Logits[f.ArgmaxID] || (v == f.Logits[f.ArgmaxID] && i < f.ArgmaxID) {
			return fmt.Errorf("frontier logits: argmax_id %d is not canonical", f.ArgmaxID)
		}
	}
	if f.Logits[f.ArgmaxID] != f.ArgmaxLogit {
		return fmt.Errorf("frontier logits: argmax_logit does not match logits")
	}
	return nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
