package dsv4eval

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// TeacherState is one forced-token logits snapshot.
type TeacherState struct {
	AcceptedDecodeCount int       `json:"accepted_decode_count"`
	ForcedTokensSHA256  string    `json:"forced_tokens_sha256"`
	TokenIDsSHA256      string    `json:"token_ids_sha256"`
	Logits              []float32 `json:"logits"`
}

// TeacherStep compares one matching forced-token state.
type TeacherStep struct {
	Step        int     `json:"step"`
	Top1Left    int     `json:"top1_left"`
	Top1Right   int     `json:"top1_right"`
	Top1Agree   bool    `json:"top1_agree"`
	MaxAbs      float64 `json:"max_abs_logit"`
	KLLeftRight float64 `json:"kl_left_to_right"`
}

// TeacherReport is a deterministic comparison of two state directories.
type TeacherReport struct {
	ForcedTokensSHA256 string        `json:"forced_tokens_sha256"`
	TokenIDsSHA256     string        `json:"token_ids_sha256"`
	Steps              []TeacherStep `json:"steps"`
}

type teacherProvenance struct {
	forced string
	tokens string
}

// ReadTeacherStates reads state-*.json files and rejects missing or
// inconsistent token-stream provenance. The returned hash is the forced-token
// hash; CompareTeacherStates also checks the full token-ID hash.
func ReadTeacherStates(dir string) (map[int]TeacherState, string, error) {
	states, provenance, err := readTeacherStates(dir)
	return states, provenance.forced, err
}

func readTeacherStates(dir string) (map[int]TeacherState, teacherProvenance, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, teacherProvenance{}, fmt.Errorf("read teacher states: %w", err)
	}
	states := make(map[int]TeacherState)
	var hash, tokenHash string
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasPrefix(name, "state-") || !strings.HasSuffix(name, ".json") {
			continue
		}
		stepText := strings.TrimSuffix(strings.TrimPrefix(name, "state-"), ".json")
		step, err := strconv.Atoi(stepText)
		if err != nil || step < 0 {
			return nil, teacherProvenance{}, fmt.Errorf("read teacher states: invalid filename %q", name)
		}
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return nil, teacherProvenance{}, fmt.Errorf("read teacher state %s: %w", name, err)
		}
		var state TeacherState
		if err := json.Unmarshal(data, &state); err != nil {
			return nil, teacherProvenance{}, fmt.Errorf("read teacher state %s: %w", name, err)
		}
		if state.AcceptedDecodeCount != step || len(state.Logits) == 0 {
			return nil, teacherProvenance{}, fmt.Errorf("read teacher state %s: invalid step or logits", name)
		}
		if err := validateTeacherHash(state.ForcedTokensSHA256, "forced token", name); err != nil {
			return nil, teacherProvenance{}, err
		}
		if err := validateTeacherHash(state.TokenIDsSHA256, "token ID", name); err != nil {
			return nil, teacherProvenance{}, err
		}
		if hash == "" {
			hash = state.ForcedTokensSHA256
		} else if hash != state.ForcedTokensSHA256 {
			return nil, teacherProvenance{}, fmt.Errorf("read teacher states: forced token hashes differ")
		}
		if tokenHash == "" {
			tokenHash = state.TokenIDsSHA256
		} else if tokenHash != state.TokenIDsSHA256 {
			return nil, teacherProvenance{}, fmt.Errorf("read teacher states: token id hashes differ")
		}
		for i, value := range state.Logits {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, teacherProvenance{}, fmt.Errorf("read teacher state %s: non-finite logit %d", name, i)
			}
		}
		if _, exists := states[step]; exists {
			return nil, teacherProvenance{}, fmt.Errorf("read teacher states: duplicate step %d", step)
		}
		states[step] = state
	}
	if len(states) == 0 {
		return nil, teacherProvenance{}, fmt.Errorf("read teacher states: no state files")
	}
	for step := 0; step < len(states); step++ {
		if _, ok := states[step]; !ok {
			return nil, teacherProvenance{}, fmt.Errorf("read teacher states: missing step %d", step)
		}
	}
	return states, teacherProvenance{forced: hash, tokens: tokenHash}, nil
}

func validateTeacherHash(value, kind, name string) error {
	if value == "" {
		return fmt.Errorf("read teacher state %s: missing %s hash", name, kind)
	}
	if len(value) != 64 {
		return fmt.Errorf("read teacher state %s: %s hash is not 64 hexadecimal characters", name, kind)
	}
	if _, err := hex.DecodeString(value); err != nil {
		return fmt.Errorf("read teacher state %s: invalid %s hash: %w", name, kind, err)
	}
	return nil
}

// CompareTeacherStates compares matching steps and refuses missing steps or
// mismatched forced-token provenance.
func CompareTeacherStates(leftDir, rightDir string) (TeacherReport, error) {
	left, leftProvenance, err := readTeacherStates(leftDir)
	if err != nil {
		return TeacherReport{}, err
	}
	right, rightProvenance, err := readTeacherStates(rightDir)
	if err != nil {
		return TeacherReport{}, err
	}
	if leftProvenance != rightProvenance {
		if leftProvenance.forced != rightProvenance.forced {
			return TeacherReport{}, fmt.Errorf("teacher states: forced token hashes differ")
		}
		return TeacherReport{}, fmt.Errorf("teacher states: token id hashes differ")
	}
	if len(left) != len(right) {
		return TeacherReport{}, fmt.Errorf("teacher states: step sets differ")
	}
	steps := make([]int, 0, len(left))
	for step := range left {
		if _, ok := right[step]; !ok {
			return TeacherReport{}, fmt.Errorf("teacher states: missing step %d", step)
		}
		steps = append(steps, step)
	}
	sort.Ints(steps)
	report := TeacherReport{ForcedTokensSHA256: leftProvenance.forced, TokenIDsSHA256: leftProvenance.tokens, Steps: make([]TeacherStep, 0, len(steps))}
	for _, step := range steps {
		l, r := left[step].Logits, right[step].Logits
		if len(l) != len(r) {
			return TeacherReport{}, fmt.Errorf("teacher states: step %d vocab differs", step)
		}
		li, ri := argmax(l), argmax(r)
		var maxAbs float64
		for i := range l {
			if d := math.Abs(float64(l[i]) - float64(r[i])); d > maxAbs {
				maxAbs = d
			}
		}
		report.Steps = append(report.Steps, TeacherStep{Step: step, Top1Left: li, Top1Right: ri, Top1Agree: li == ri, MaxAbs: maxAbs, KLLeftRight: logitsKL(l, r)})
	}
	return report, nil
}

func logitsKL(left, right []float32) float64 {
	maxLeft, maxRight := float64(left[0]), float64(right[0])
	for i := 1; i < len(left); i++ {
		maxLeft = math.Max(maxLeft, float64(left[i]))
		maxRight = math.Max(maxRight, float64(right[i]))
	}
	var sumLeft, sumRight float64
	for i := range left {
		sumLeft += math.Exp(float64(left[i]) - maxLeft)
		sumRight += math.Exp(float64(right[i]) - maxRight)
	}
	logZLeft, logZRight := maxLeft+math.Log(sumLeft), maxRight+math.Log(sumRight)
	var kl float64
	for i := range left {
		logP := float64(left[i]) - logZLeft
		p := math.Exp(logP)
		kl += p * (logP - (float64(right[i]) - logZRight))
	}
	return kl
}
