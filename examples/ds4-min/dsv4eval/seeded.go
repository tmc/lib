package dsv4eval

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// SeededDecodeMetadata identifies one end-to-end seeded decode run. The
// comparator requires matching metadata because a token trace without its
// sampling and model conditions is not a parity artifact.
type SeededDecodeMetadata struct {
	Build       string  `json:"build"`
	ModelSHA256 string  `json:"model_sha256"`
	Context     int     `json:"context"`
	Temperature float32 `json:"temperature"`
	Seed        uint64  `json:"seed"`
	MTPDisabled bool    `json:"mtp_disabled"`
	// LogitsDir names the directory containing step-%04d.f32 raw float32
	// vectors.  It is deliberately metadata rather than an implicit sidecar:
	// a digest-only trace must say that it is artifact-only.
	LogitsDir    string `json:"logits_dir,omitempty"`
	ArtifactOnly bool   `json:"artifact_only,omitempty"`
}

// SeededDecodeStep is one post-prefill decode observation. LogitsSHA256 is
// the SHA-256 of the exact little-endian float32 logits vector at Step.
type SeededDecodeStep struct {
	Step         int    `json:"step"`
	Token        int    `json:"token"`
	LogitsSHA256 string `json:"logits_sha256"`
}

// SeededDecodeTrace is the bounded, host-side representation of a seeded
// decode capture. It intentionally carries digests rather than pretending
// that token equality proves numerical equality.
type SeededDecodeTrace struct {
	Metadata SeededDecodeMetadata `json:"metadata"`
	Steps    []SeededDecodeStep   `json:"steps"`
}

// SeededDecodeArtifactOptions describes the raw files emitted by ds4 and
// ds4-min.  The two implementations intentionally share this small artifact
// contract: logprobs JSON supplies the selected token, while step-N.f32
// supplies the exact little-endian float32 logit vector.
type SeededDecodeArtifactOptions struct {
	LogitsDir    string
	LogprobsPath string
	Build        string
	ModelSHA256  string
	Context      int
	Temperature  float32
	Seed         uint64
	MTPDisabled  bool
}

type seededLogprobDump struct {
	Source       string              `json:"source"`
	PromptTokens int                 `json:"prompt_tokens"`
	Context      int                 `json:"ctx"`
	TopK         int                 `json:"top_k"`
	Steps        []seededLogprobStep `json:"steps"`
}

type seededLogprobStep struct {
	Step        int                  `json:"step"`
	Selected    seededLogprobToken   `json:"selected"`
	TopLogprobs []seededLogprobScore `json:"top_logprobs"`
}

type seededLogprobToken struct {
	ID    int    `json:"id"`
	Text  string `json:"text"`
	Bytes []int  `json:"bytes"`
}

type seededLogprobScore struct {
	Token   seededLogprobToken `json:"token"`
	Logit   float32            `json:"logit"`
	Logprob float32            `json:"logprob"`
}

// ConvertSeededDecodeArtifacts converts one raw per-step dump into the
// strict trace consumed by CompareSeededDecode.  It hashes the bytes as
// written, so no host float conversion can change the evidence.  Both the
// selected-token stream and every logits file must be complete and ordered.
func ConvertSeededDecodeArtifacts(opts SeededDecodeArtifactOptions) (SeededDecodeTrace, error) {
	if opts.LogitsDir == "" || opts.LogprobsPath == "" {
		return SeededDecodeTrace{}, fmt.Errorf("seeded decode: logits directory and logprobs path are required")
	}
	metadata := SeededDecodeMetadata{
		Build: opts.Build, ModelSHA256: opts.ModelSHA256,
		Context: opts.Context, Temperature: opts.Temperature,
		Seed: opts.Seed, MTPDisabled: opts.MTPDisabled,
		LogitsDir: opts.LogitsDir,
	}
	if err := validateSeededMetadata(metadata, "converted"); err != nil {
		return SeededDecodeTrace{}, err
	}
	data, err := os.ReadFile(opts.LogprobsPath)
	if err != nil {
		return SeededDecodeTrace{}, fmt.Errorf("seeded decode: read logprobs: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var dump seededLogprobDump
	if err := dec.Decode(&dump); err != nil {
		return SeededDecodeTrace{}, fmt.Errorf("seeded decode: decode logprobs: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return SeededDecodeTrace{}, fmt.Errorf("seeded decode: logprobs has trailing JSON")
		}
		return SeededDecodeTrace{}, fmt.Errorf("seeded decode: logprobs trailing data: %w", err)
	}
	if len(dump.Steps) == 0 {
		return SeededDecodeTrace{}, fmt.Errorf("seeded decode: logprobs has no steps")
	}
	entries, err := os.ReadDir(opts.LogitsDir)
	if err != nil {
		return SeededDecodeTrace{}, fmt.Errorf("seeded decode: read logits directory: %w", err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasPrefix(name, "step-") || !strings.HasSuffix(name, ".f32") {
			continue
		}
		if entry.IsDir() {
			return SeededDecodeTrace{}, fmt.Errorf("seeded decode: logits step is a directory: %s", name)
		}
		matched := false
		for i := range dump.Steps {
			if name == fmt.Sprintf("step-%04d.f32", i) {
				matched = true
				break
			}
		}
		if !matched {
			return SeededDecodeTrace{}, fmt.Errorf("seeded decode: unexpected logits row %s", name)
		}
	}
	steps := make([]SeededDecodeStep, len(dump.Steps))
	for i, raw := range dump.Steps {
		if raw.Step != i {
			return SeededDecodeTrace{}, fmt.Errorf("seeded decode: logprob step %d is %d", i, raw.Step)
		}
		if raw.Selected.ID < 0 {
			return SeededDecodeTrace{}, fmt.Errorf("seeded decode: selected token at step %d is negative", i)
		}
		path := filepath.Join(opts.LogitsDir, fmt.Sprintf("step-%04d.f32", i))
		logits, err := os.ReadFile(path)
		if err != nil {
			return SeededDecodeTrace{}, fmt.Errorf("seeded decode: read logits step %d: %w", i, err)
		}
		if len(logits) == 0 || len(logits)%4 != 0 {
			return SeededDecodeTrace{}, fmt.Errorf("seeded decode: logits step %d has invalid byte length %d", i, len(logits))
		}
		digest := sha256.Sum256(logits)
		steps[i] = SeededDecodeStep{
			Step: i, Token: raw.Selected.ID, LogitsSHA256: hex.EncodeToString(digest[:]),
		}
	}
	return SeededDecodeTrace{Metadata: metadata, Steps: steps}, nil
}

// ReadSeededDecodeTrace reads one complete JSON trace. The decoder rejects
// unknown fields and trailing values so a receipt cannot silently omit or
// append provenance. Validation is performed by CompareSeededDecode, which
// also checks the trace against its peer.
func ReadSeededDecodeTrace(r io.Reader) (SeededDecodeTrace, error) {
	if r == nil {
		return SeededDecodeTrace{}, fmt.Errorf("seeded decode: nil trace reader")
	}
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var trace SeededDecodeTrace
	if err := dec.Decode(&trace); err != nil {
		return SeededDecodeTrace{}, fmt.Errorf("seeded decode: read trace: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return SeededDecodeTrace{}, fmt.Errorf("seeded decode: trailing JSON value")
		}
		return SeededDecodeTrace{}, fmt.Errorf("seeded decode: trailing data: %w", err)
	}
	return trace, nil
}

// SeededDecodeReport records the first point at which the two runs no longer
// have identical logits. A token mismatch before that point is a sampler or
// control failure; a mismatch at or after it is attributable to the changed
// logits and is reported rather than silently accepted.
type SeededDecodeReport struct {
	Steps                  int     `json:"steps"`
	ComparedSteps          int     `json:"compared_steps"`
	TokenMatches           int     `json:"token_matches"`
	LogitDigestMatches     int     `json:"logit_digest_matches"`
	FirstLogitDifference   int     `json:"first_logit_difference"`
	FirstTokenDifference   int     `json:"first_token_difference"`
	NumericChecked         bool    `json:"numeric_checked"`
	NumericPass            bool    `json:"numeric_pass"`
	MaxAbs                 float64 `json:"max_abs"`
	MaxRel                 float64 `json:"max_rel"`
	FirstNumericDifference int     `json:"first_numeric_difference"`
}

// Pass reports whether every token before the first differing logit digest
// matched. A zero-step report never passes, and an identical run is a valid
// pass with FirstLogitDifference set to -1. The token at the first differing
// logit step is still required to match; only later token differences are
// attributable to changed logits. Raw-vector tolerance is diagnostic and is
// exposed separately in NumericPass; it must not hide this attribution rule.
func (r SeededDecodeReport) Pass() bool {
	return r.Steps > 0 && r.ComparedSteps == r.Steps &&
		(r.FirstTokenDifference < 0 || (r.FirstLogitDifference >= 0 && r.FirstTokenDifference > r.FirstLogitDifference))
}

// CompareSeededDecode compares two traces and refuses incomplete or
// incompatible artifacts. Both runs must use the same model, context,
// temperature, seed, and non-speculative mode. MTP must be disabled because
// its extra random draws make the registered seeded stream incomparable.
func CompareSeededDecode(left, right SeededDecodeTrace) (SeededDecodeReport, error) {
	if err := validateSeededMetadata(left.Metadata, "left"); err != nil {
		return SeededDecodeReport{}, err
	}
	if err := validateSeededMetadata(right.Metadata, "right"); err != nil {
		return SeededDecodeReport{}, err
	}
	if left.Metadata.ModelSHA256 != right.Metadata.ModelSHA256 {
		return SeededDecodeReport{}, fmt.Errorf("seeded decode: model hashes differ")
	}
	if left.Metadata.Context != right.Metadata.Context {
		return SeededDecodeReport{}, fmt.Errorf("seeded decode: contexts differ")
	}
	if left.Metadata.Temperature != right.Metadata.Temperature {
		return SeededDecodeReport{}, fmt.Errorf("seeded decode: temperatures differ")
	}
	if left.Metadata.Seed != right.Metadata.Seed {
		return SeededDecodeReport{}, fmt.Errorf("seeded decode: seeds differ")
	}
	if len(left.Steps) == 0 || len(right.Steps) == 0 {
		return SeededDecodeReport{}, fmt.Errorf("seeded decode: no steps")
	}
	if len(left.Steps) != len(right.Steps) {
		return SeededDecodeReport{}, fmt.Errorf("seeded decode: step counts differ: %d versus %d", len(left.Steps), len(right.Steps))
	}

	report := SeededDecodeReport{
		Steps: len(left.Steps), FirstLogitDifference: -1, FirstTokenDifference: -1,
		FirstNumericDifference: -1,
	}
	leftRaw, rightRaw := left.Metadata.LogitsDir != "", right.Metadata.LogitsDir != ""
	if leftRaw != rightRaw {
		return SeededDecodeReport{}, fmt.Errorf("seeded decode: raw logits directory is missing on one side")
	}
	report.NumericChecked = leftRaw && rightRaw
	report.NumericPass = !report.NumericChecked
	for i := range left.Steps {
		l, r := left.Steps[i], right.Steps[i]
		if l.Step != i || r.Step != i {
			return SeededDecodeReport{}, fmt.Errorf("seeded decode: non-contiguous step at index %d: %d versus %d", i, l.Step, r.Step)
		}
		if err := validateSeededStep(l, "left"); err != nil {
			return SeededDecodeReport{}, err
		}
		if err := validateSeededStep(r, "right"); err != nil {
			return SeededDecodeReport{}, err
		}
		report.ComparedSteps++
		if l.LogitsSHA256 == r.LogitsSHA256 {
			report.LogitDigestMatches++
		} else if report.FirstLogitDifference < 0 {
			report.FirstLogitDifference = i
		}
		if l.Token == r.Token {
			report.TokenMatches++
		} else if report.FirstTokenDifference < 0 {
			report.FirstTokenDifference = i
		}
		if report.NumericChecked {
			maxAbs, maxRel, within, err := compareSeededLogitStep(left.Metadata.LogitsDir, right.Metadata.LogitsDir, i)
			if err != nil {
				return SeededDecodeReport{}, err
			}
			if maxAbs > report.MaxAbs {
				report.MaxAbs = maxAbs
			}
			if maxRel > report.MaxRel {
				report.MaxRel = maxRel
			}
			if !within && report.FirstNumericDifference < 0 {
				report.FirstNumericDifference = i
			}
		}
	}
	if report.NumericChecked {
		report.NumericPass = report.FirstNumericDifference < 0
	}
	if report.FirstTokenDifference >= 0 {
		firstLogit := report.FirstLogitDifference
		if firstLogit < 0 || report.FirstTokenDifference < firstLogit {
			return report, fmt.Errorf("seeded decode: token differs at step %d before the first differing logits", report.FirstTokenDifference)
		}
	}
	return report, nil
}

func validateSeededMetadata(m SeededDecodeMetadata, side string) error {
	if m.Build == "" {
		return fmt.Errorf("seeded decode: %s build is missing", side)
	}
	if len(m.ModelSHA256) != 64 {
		return fmt.Errorf("seeded decode: %s model hash is not 64 hexadecimal characters", side)
	}
	if _, err := hex.DecodeString(m.ModelSHA256); err != nil {
		return fmt.Errorf("seeded decode: %s model hash: %w", side, err)
	}
	if m.Context <= 0 {
		return fmt.Errorf("seeded decode: %s context is invalid", side)
	}
	if math.IsNaN(float64(m.Temperature)) || math.IsInf(float64(m.Temperature), 0) || m.Temperature <= 0 {
		return fmt.Errorf("seeded decode: %s temperature is invalid", side)
	}
	if !m.MTPDisabled {
		return fmt.Errorf("seeded decode: %s MTP is enabled", side)
	}
	if m.LogitsDir == "" && !m.ArtifactOnly {
		return fmt.Errorf("seeded decode: %s trace is missing logits_dir and artifact_only", side)
	}
	if m.LogitsDir != "" && m.ArtifactOnly {
		return fmt.Errorf("seeded decode: %s trace marks raw logits artifact-only", side)
	}
	return nil
}

const (
	seededLogitAbsTolerance = 1e-4
	seededLogitRelTolerance = 1e-4
)

func compareSeededLogitStep(leftDir, rightDir string, step int) (maxAbs, maxRel float64, within bool, err error) {
	name := fmt.Sprintf("step-%04d.f32", step)
	left, err := readSeededLogitRow(filepath.Join(leftDir, name), "left", step)
	if err != nil {
		return 0, 0, false, err
	}
	right, err := readSeededLogitRow(filepath.Join(rightDir, name), "right", step)
	if err != nil {
		return 0, 0, false, err
	}
	if len(left) != len(right) {
		return 0, 0, false, fmt.Errorf("seeded decode: logits step %d length differs: %d versus %d", step, len(left), len(right))
	}
	within = true
	for i := range left {
		l, r := left[i], right[i]
		if !isFinite32(l) || !isFinite32(r) {
			return 0, 0, false, fmt.Errorf("seeded decode: non-finite logits at step %d index %d", step, i)
		}
		d := math.Abs(float64(l) - float64(r))
		if d > maxAbs {
			maxAbs = d
		}
		denom := math.Max(math.Abs(float64(l)), math.Abs(float64(r)))
		if denom == 0 {
			denom = 1
		}
		rel := d / denom
		if rel > maxRel {
			maxRel = rel
		}
		if d > seededLogitAbsTolerance+seededLogitRelTolerance*math.Abs(float64(l)) {
			within = false
		}
	}
	return maxAbs, maxRel, within, nil
}

func readSeededLogitRow(path, side string, step int) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("seeded decode: read %s logits step %d: %w", side, step, err)
	}
	if len(b) == 0 || len(b)%4 != 0 {
		return nil, fmt.Errorf("seeded decode: %s logits step %d has invalid byte length %d", side, step, len(b))
	}
	values := make([]float32, len(b)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return values, nil
}

func isFinite32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func validateSeededStep(s SeededDecodeStep, side string) error {
	if s.Step < 0 {
		return fmt.Errorf("seeded decode: %s step is negative", side)
	}
	if s.Token < 0 {
		return fmt.Errorf("seeded decode: %s token at step %d is negative", side, s.Step)
	}
	if len(s.LogitsSHA256) != 64 {
		return fmt.Errorf("seeded decode: %s logits hash at step %d is not 64 hexadecimal characters", side, s.Step)
	}
	if _, err := hex.DecodeString(s.LogitsSHA256); err != nil {
		return fmt.Errorf("seeded decode: %s logits hash at step %d: %w", side, s.Step, err)
	}
	return nil
}
