package dsv4eval

import (
	"crypto/sha256"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// Runtime supplies the model and tokenizer operations needed by the evaluator.
// Sync installs a complete prompt, and Eval accepts one generated token.
// Logits must describe the next token after the most recent successful call.
type Runtime interface {
	Render(system, question string) ([]int, error)
	Sync([]int) error
	Eval(int) error
	Logits() []float32
	Text(int) (string, error)
	IsStop(int) bool
	ThinkClose() []int
}

// RunOptions pins a greedy hard-smoke execution. A zero value is invalid.
// Budget overrides case budgets when positive. Soft and Hard reserve answer
// tokens, following ds4_eval.c's soft-rank and hard-close controller.
type RunOptions struct {
	Build, ModelSHA256                string
	Context, Budget, Soft, Hard, Rank int
	// Capture receives actual per-case data before its trace entry is written.
	// Nil disables artifact capture without changing selection or model work.
	Capture func(CaseArtifacts) error
}

// EvalSystemPrompt is the pinned upstream evaluation system instruction.
const EvalSystemPrompt = "You are solving a hard benchmark question. Reason carefully. The final answer must follow the requested format exactly."

// QuestionPrompt returns the upstream answer-format instruction and question.
func QuestionPrompt(c Case) string {
	s := c.Question + "\n"
	if c.Kind == Choice {
		s += "\nChoices:\n"
		for i, choice := range c.Choices {
			s += fmt.Sprintf("%c. %s\n", 'A'+i, choice)
		}
	}
	var instruction string
	switch c.Kind {
	case Choice:
		instruction = "Solve the question. At the end, write exactly one final line in this format and do not write anything after it:\nAnswer: <letter>"
	case LineSet:
		instruction = "At the end, write exactly one final line in this format and do not write anything after it:\nAnswer: <line number or comma-separated line numbers>"
	case Integer:
		instruction = "Solve the problem. At the end, write exactly one final line in this format and do not write anything after it:\nAnswer: <integer>"
	case Rational:
		instruction = "Solve the problem. Reduce the result. At the end, write exactly one final line in this format and do not write anything after it:\nAnswer: <integer or reduced fraction>"
	case OrderedSequence:
		instruction = "Solve the problem. At the end, write exactly one final line containing the answers in the requested order, separated by commas, and do not write anything after it:\nAnswer: <ordered answers>"
	default:
		instruction = "Solve the problem. At the end, write exactly one final line in this format and do not write anything after it:\nAnswer: <exact answer>"
	}
	return s + "\n" + instruction
}

// RunSmoke executes all twelve supplied hard-smoke questions and writes a
// complete trace only when every model operation succeeds. It never infers
// a runtime result from a pre-existing trace or supplied status. Questions and
// answers must come from a licensed source matching the embedded case index.
func RunSmoke(w io.Writer, model Runtime, cases []Case, opt RunOptions) error {
	if model == nil {
		return fmt.Errorf("run smoke: missing runtime")
	}
	if err := validateRuntimeCases(cases, opt); err != nil {
		return err
	}
	writer := NewTraceWriter(w, TraceHeader{Fields: map[string]string{
		"suite": "hard-smoke", "build": opt.Build, "model_sha256": opt.ModelSHA256,
		"ctx": strconv.Itoa(opt.Context), "temp": "0", "think": "HIGH",
	}})
	counts := map[string]int{}
	for i, c := range cases {
		question := QuestionPrompt(c)
		ids, err := model.Render(EvalSystemPrompt, question)
		if err != nil {
			return fmt.Errorf("run smoke %s: render: %w", c.ID, err)
		}
		budget := c.MaxTokens
		if opt.Budget > 0 {
			budget = opt.Budget
		}
		if len(ids) == 0 || len(ids)+budget > opt.Context {
			return fmt.Errorf("run smoke %s: prompt and budget exceed context", c.ID)
		}
		if err := model.Sync(ids); err != nil {
			return fmt.Errorf("run smoke %s: sync: %w", c.ID, err)
		}
		closeIDs := model.ThinkClose()
		if len(closeIDs) == 0 {
			return fmt.Errorf("run smoke %s: missing think-close tokens", c.ID)
		}
		var output strings.Builder
		var decisions []DecisionArtifact
		var accepted []int
		historyHash := sha256.New()
		for _, id := range ids {
			tokenHashWrite(historyHash, id)
		}
		stopDisposition := "budget_exhausted"
		kind, generated, forced := "none", 0, 0
		closeIndex, closeRemaining, closeRank := 0, 0, 0
		thinking := true
		for generated < budget {
			logits := model.Logits()
			token, err := greedyFinite(logits)
			if err != nil {
				return fmt.Errorf("run smoke %s: %w", c.ID, err)
			}
			remaining := budget - generated
			greedy, forcedKind := token, "none"
			if thinking {
				if forced > 0 {
					forcedKind = kind
					token = closeIDs[forced]
					forced++
					if forced == len(closeIDs) {
						forced = 0
					}
				} else if remaining <= opt.Hard {
					forcedKind = "hard_forced"
					token = closeIDs[0]
					kind = "hard_forced"
					closeIndex, closeRemaining = generated+1, remaining
					if len(closeIDs) > 1 {
						forced = 1
					}
				} else if remaining <= opt.Soft && len(closeIDs) == 1 && tokenRank(logits, closeIDs[0]) <= opt.Rank {
					forcedKind = "soft_forced"
					token = closeIDs[0]
					kind = "soft_forced"
					closeIndex, closeRemaining, closeRank = generated+1, remaining, tokenRank(logits, token)
				}
			}
			if token < 0 || token >= len(logits) {
				return fmt.Errorf("run smoke %s: invalid token %d", c.ID, token)
			}
			decision := DecisionArtifact{Step: generated + 1, Greedy: greedy, Chosen: token, Forced: forcedKind, Stop: model.IsStop(token), BeforeHistorySHA256: fmt.Sprintf("%x", historyHash.Sum(nil))}
			if opt.Capture != nil {
				decision.LogitsSHA256 = logitsHash(logits)
			}
			if decision.Stop {
				stopDisposition = "selected_stop_not_consumed"
				if opt.Capture != nil {
					decisions = append(decisions, decision)
				}
				break
			}
			if err := model.Eval(token); err != nil {
				return fmt.Errorf("run smoke %s: eval: %w", c.ID, err)
			}
			if opt.Capture != nil {
				decision.Accepted = true
				decisions = append(decisions, decision)
				accepted = append(accepted, token)
			}
			tokenHashWrite(historyHash, token)
			text, err := model.Text(token)
			if err != nil {
				return fmt.Errorf("run smoke %s: token text: %w", c.ID, err)
			}
			output.WriteString(text)
			generated++
			if thinking && strings.Contains(output.String(), "</think>") {
				thinking = false
				if kind == "none" {
					kind = "natural"
					closeIndex, closeRemaining = generated, remaining
				}
			}
		}
		picked, complete, pass := Grade(c, output.String())
		status := "INCOMPLETE"
		if complete {
			status = "FAILED"
			if pass {
				status = "PASSED"
			}
		}
		counts[status]++
		if opt.Capture != nil {
			terminal := model.Logits()
			if _, err := greedyFinite(terminal); err != nil {
				return fmt.Errorf("run smoke %s: terminal logits: %w", c.ID, err)
			}
			history := append(append([]int(nil), ids...), accepted...)
			capture := CaseArtifacts{Index: i + 1, Source: c.Source, ID: c.ID, Build: opt.Build, ModelSHA256: opt.ModelSHA256, Context: opt.Context, Budget: budget, Prompt: append([]int(nil), ids...), Generated: accepted, History: history, HistorySHA256: fmt.Sprintf("%x", historyHash.Sum(nil)), Decisions: decisions, Status: status, ThinkClose: kind, StopDisposition: stopDisposition, TerminalLogits: append([]float32(nil), terminal...)}
			if err := opt.Capture(capture); err != nil {
				return fmt.Errorf("run smoke %s: capture: %w", c.ID, err)
			}
		}
		fields := map[string]string{"status": status, "picked": picked, "expected": c.Answer, "think_close": kind, "prompt_tokens": strconv.Itoa(len(ids)), "generated_tokens": strconv.Itoa(generated), "generation_budget": strconv.Itoa(budget), "answer_kind": kindName(c.Kind)}
		fields["think_close_token_index"] = strconv.Itoa(closeIndex)
		fields["think_close_remaining_budget"] = strconv.Itoa(closeRemaining)
		fields["think_close_rank"] = strconv.Itoa(closeRank)
		if err := writer.Case(TraceCase{Index: i + 1, Total: 12, Source: c.Source, ID: c.ID, Fields: fields, Blocks: map[string]string{"SYSTEM_PROMPT": EvalSystemPrompt + "\n", "QUESTION_PROMPT": question + "\n", "MODEL_OUTPUT": output.String() + "\n"}}); err != nil {
			return err
		}
	}
	if err := writer.Summary(map[string]string{"total": "12", "passed": strconv.Itoa(counts["PASSED"]), "failed": strconv.Itoa(counts["FAILED"]), "incomplete": strconv.Itoa(counts["INCOMPLETE"])}); err != nil {
		return err
	}
	if counts["FAILED"] != 0 || counts["INCOMPLETE"] != 0 {
		return fmt.Errorf("run smoke: failed=%d incomplete=%d", counts["FAILED"], counts["INCOMPLETE"])
	}
	return nil
}

func validateRuntimeCases(cases []Case, opt RunOptions) error {
	if opt.Build == "" || len(opt.ModelSHA256) != 64 || opt.Context <= 0 || opt.Hard <= 0 || opt.Soft < opt.Hard || opt.Rank <= 0 || opt.Budget < 0 {
		return fmt.Errorf("run smoke: invalid provenance or budget options")
	}
	for _, r := range opt.ModelSHA256 {
		if !(r >= '0' && r <= '9' || r >= 'a' && r <= 'f') {
			return fmt.Errorf("run smoke: invalid model hash")
		}
	}
	if len(cases) != 12 {
		return fmt.Errorf("run smoke: got %d cases, want 12", len(cases))
	}
	seen := map[string]bool{}
	for _, c := range cases {
		key := c.Source + "\x00" + c.ID
		if !hardSmokeCases[key] || seen[key] || strings.TrimSpace(c.Question) == "" || strings.TrimSpace(c.Answer) == "" {
			return fmt.Errorf("run smoke: missing, duplicate or incomplete case %s/%s", c.Source, c.ID)
		}
		seen[key] = true
		budget := c.MaxTokens
		if opt.Budget > 0 {
			budget = opt.Budget
		}
		if budget <= opt.Hard {
			return fmt.Errorf("run smoke: budget must exceed hard reserve")
		}
	}
	return nil
}

func greedyFinite(logits []float32) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("empty logits")
	}
	top := 0
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, fmt.Errorf("non-finite logits at %d", i)
		}
		if v > logits[top] {
			top = i
		}
	}
	return top, nil
}

func tokenRank(logits []float32, token int) int {
	if token < 0 || token >= len(logits) {
		return len(logits) + 1
	}
	rank := 1
	for i, v := range logits {
		// Upstream top-logprobs inserts tokens in ascending ID order and
		// replaces only on strictly greater logits, so ties retain that order.
		if v > logits[token] || v == logits[token] && i < token {
			rank++
		}
	}
	return rank
}
