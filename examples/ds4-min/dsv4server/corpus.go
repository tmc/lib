package dsv4server

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
)

const maxCorpusArtifactBytes = 8 << 20

// CorpusProvenance identifies the upstream runtime used to record a case.
// A comparison must bind all fields before treating a transcript as an oracle.
type CorpusProvenance struct {
	SourceRevision   string
	BinarySHA256     string
	ModelSHA256      string
	Backend          string
	Context          uint32
	Temperature      float64
	ThinkMode        string
	ThinkModeDefault string
	ThinkModeByStep  map[string]string
}

// CorpusCase is one owner-recorded upstream server case. Large KVC files are
// intentionally not loaded; terminal contains only their recorded hashes.
type CorpusCase struct {
	CaseID       string
	Step         string
	Method       string
	Endpoint     string
	Status       string
	Request      []byte
	Response     []byte
	PromptTokens []int
	Tokens       []int
	Metadata     CorpusProvenance
	Terminal     json.RawMessage
}

// CorpusObservation is one response collected from a server under test.
// Request and Response contain the bytes sent on the wire. Status uses the
// same HTTP status-line form as CorpusCase.Status.
type CorpusObservation struct {
	Status   string
	Request  []byte
	Response []byte
	Tokens   []int
}

// CompareCorpusCase compares an observed response with one finalized corpus
// case. JSON objects are compared canonically with generated identities
// masked; SSE data records use the same normalization one event at a time.
// Token traces and status lines remain exact. A missing or malformed observed
// artifact is an error, never an omitted case.
func CompareCorpusCase(want CorpusCase, got CorpusObservation) error {
	if err := CompareCorpusHTTP(want, got); err != nil {
		return err
	}
	if !reflect.DeepEqual(want.Tokens, got.Tokens) {
		return fmt.Errorf("server corpus: token trace differs")
	}
	return nil
}

// CompareCorpusHTTP compares the wire-visible portion of a corpus case.
// It deliberately does not compare Tokens: the HTTP server contract does
// not expose generated token IDs. Callers must report that token identity is
// UNREAD rather than treating a wire match as full model parity.
func CompareCorpusHTTP(want CorpusCase, got CorpusObservation) error {
	if want.Status != got.Status {
		return fmt.Errorf("server corpus: status differs: got %q want %q", got.Status, want.Status)
	}
	if err := compareCorpusRequest(want.Request, got.Request); err != nil {
		return err
	}
	wantResponse, err := normalizeCorpusResponse(want.Response)
	if err != nil {
		return fmt.Errorf("server corpus: expected response: %w", err)
	}
	gotResponse, err := normalizeCorpusResponse(got.Response)
	if err != nil {
		return fmt.Errorf("server corpus: observed response: %w", err)
	}
	if wantResponse != gotResponse {
		return errors.New("server corpus: response differs")
	}
	return nil
}

func compareCorpusRequest(want, got []byte) error {
	if len(want) == 0 || len(got) == 0 {
		if bytes.Equal(want, got) {
			return nil
		}
		return errors.New("server corpus: request differs")
	}
	wantJSON, wantErr := canonicalJSON(want)
	gotJSON, gotErr := canonicalJSON(got)
	if wantErr == nil && gotErr == nil {
		if bytes.Equal(wantJSON, gotJSON) {
			return nil
		}
	} else if bytes.Equal(want, got) {
		return nil
	}
	return errors.New("server corpus: request differs")
}

func normalizeCorpusResponse(data []byte) (string, error) {
	if len(data) == 0 {
		return "", errors.New("empty response")
	}
	if !bytes.HasPrefix(bytes.TrimSpace(data), []byte("data:")) {
		canonical, err := canonicalMaskedJSON(data)
		if err != nil {
			return "", err
		}
		return string(canonical), nil
	}
	var out []string
	for _, line := range strings.Split(strings.TrimSuffix(string(data), "\n"), "\n") {
		if line == "" || line == "data: [DONE]" {
			if line == "data: [DONE]" {
				out = append(out, line)
			}
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			return "", fmt.Errorf("malformed SSE line %q", line)
		}
		canonical, err := canonicalMaskedJSON([]byte(strings.TrimPrefix(line, "data: ")))
		if err != nil {
			return "", err
		}
		out = append(out, "data: "+string(canonical))
	}
	if len(out) == 0 {
		return "", errors.New("empty SSE response")
	}
	return strings.Join(out, "\n"), nil
}

// Corpus is a finalized owner-recorded G1-G9 corpus. The generated files stay
// outside the repository; this value contains only bounded JSON and token
// artifacts needed by a comparison gate.
type Corpus struct {
	Provenance CorpusProvenance
	Cases      map[string][]CorpusCase
}

// Compare compares one observation for every finalized case. The key is the
// stable case-id/step pair, not a filesystem path; this keeps a moved corpus
// equivalent while rejecting missing or extra observations.
func (c Corpus) Compare(observations map[string]CorpusObservation) error {
	if len(observations) == 0 {
		return errors.New("server corpus: no observations")
	}
	expected := make(map[string]CorpusCase)
	for group, cases := range c.Cases {
		for _, want := range cases {
			key := corpusCaseKey(group, want.Step)
			if _, exists := expected[key]; exists {
				return fmt.Errorf("server corpus: duplicate case key %q", key)
			}
			expected[key] = want
		}
	}
	if len(expected) != len(observations) {
		return fmt.Errorf("server corpus: observations=%d, want=%d", len(observations), len(expected))
	}
	for key, want := range expected {
		got, ok := observations[key]
		if !ok {
			return fmt.Errorf("server corpus: missing observation %q", key)
		}
		if err := CompareCorpusCase(want, got); err != nil {
			return fmt.Errorf("server corpus: %s: %w", key, err)
		}
	}
	for key := range observations {
		if _, ok := expected[key]; !ok {
			return fmt.Errorf("server corpus: unexpected observation %q", key)
		}
	}
	return nil
}

func corpusCaseKey(group, step string) string { return group + "/" + step }

// LoadCorpus requires a finalized corpus root. It rejects partial recording,
// missing G1-G9 cases, and build/provenance drift before loading any case.
func LoadCorpus(root string) (Corpus, error) {
	if root == "" {
		return Corpus{}, errors.New("server corpus: missing root")
	}
	corpusDir := filepath.Join(root, "corpus")
	manifestBytes, err := readCorpusFile(filepath.Join(corpusDir, "manifest.json"), true)
	if err != nil {
		return Corpus{}, err
	}
	var manifest struct {
		Cases map[string]struct {
			Dir     string `json:"dir"`
			Verdict string `json:"verdict"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		return Corpus{}, fmt.Errorf("server corpus manifest: %w", err)
	}
	determinismBytes, err := readCorpusFile(filepath.Join(corpusDir, "determinism.json"), true)
	if err != nil {
		return Corpus{}, err
	}
	var determinism struct {
		Cases map[string]struct {
			Verdict string `json:"verdict"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(determinismBytes, &determinism); err != nil {
		return Corpus{}, fmt.Errorf("server corpus determinism: %w", err)
	}
	prov, err := loadCorpusProvenance(filepath.Join(root, "provenance.json"))
	if err != nil {
		return Corpus{}, err
	}
	usedThinkModes := make(map[string]struct{}, len(prov.ThinkModeByStep))
	for key := range prov.ThinkModeByStep {
		usedThinkModes[key] = struct{}{}
	}
	out := Corpus{Provenance: prov, Cases: make(map[string][]CorpusCase, 9)}
	for i := 1; i <= 9; i++ {
		id := fmt.Sprintf("G%d", i)
		entry, ok := manifest.Cases[id]
		if !ok || entry.Dir == "" || entry.Verdict != "PASS" {
			return Corpus{}, fmt.Errorf("server corpus: %s is missing or not PASS", id)
		}
		if d, ok := determinism.Cases[id]; !ok || d.Verdict != "PASS" {
			return Corpus{}, fmt.Errorf("server corpus: %s determinism is missing or not PASS", id)
		}
		caseRoot, err := safeCorpusCaseDir(corpusDir, entry.Dir)
		if err != nil {
			return Corpus{}, err
		}
		var cases []CorpusCase
		if err := filepath.WalkDir(caseRoot, func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() || entry.Name() != "metadata.json" {
				return nil
			}
			c, err := LoadCorpusCase(filepath.Dir(path))
			if err != nil {
				return err
			}
			if c.CaseID != id {
				return fmt.Errorf("server corpus: case %s contains %s", id, c.CaseID)
			}
			relativeStepDir, err := filepath.Rel(corpusDir, filepath.Dir(path))
			if err != nil {
				return fmt.Errorf("server corpus: relative case path: %w", err)
			}
			if err := validateCorpusBuild(c.Metadata, prov, filepath.ToSlash(relativeStepDir)); err != nil {
				return err
			}
			if prov.ThinkMode == "per-case" {
				delete(usedThinkModes, filepath.ToSlash(relativeStepDir))
			}
			cases = append(cases, c)
			return nil
		}); err != nil {
			return Corpus{}, err
		}
		if len(cases) == 0 {
			return Corpus{}, fmt.Errorf("server corpus: %s has no case artifacts", id)
		}
		out.Cases[id] = cases
	}
	if len(usedThinkModes) != 0 {
		return Corpus{}, fmt.Errorf("server corpus: think_mode_by_step has unused entries")
	}
	return out, nil
}

func safeCorpusCaseDir(root, relative string) (string, error) {
	if relative == "" || filepath.IsAbs(relative) {
		return "", errors.New("server corpus: case directory is not relative")
	}
	clean := filepath.Clean(relative)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", errors.New("server corpus: case directory escapes root")
	}
	return filepath.Join(root, clean), nil
}

// LoadCorpusCase reads one finalized corpus case directory. Optional request,
// prompt-token, token, and terminal files are accepted because error cases do
// not produce model tokens; metadata and response remain mandatory.
func LoadCorpusCase(dir string) (CorpusCase, error) {
	if dir == "" {
		return CorpusCase{}, errors.New("server corpus: missing case directory")
	}
	metaBytes, err := readCorpusFile(filepath.Join(dir, "metadata.json"), true)
	if err != nil {
		return CorpusCase{}, err
	}
	var raw struct {
		CaseID       string   `json:"case_id"`
		Step         string   `json:"step"`
		Method       string   `json:"method"`
		Endpoint     string   `json:"endpoint"`
		Status       string   `json:"status"`
		SourceRev    string   `json:"source_rev"`
		BinarySHA256 string   `json:"binary_sha256"`
		ModelSHA256  string   `json:"model_sha256"`
		Backend      string   `json:"backend"`
		Context      uint32   `json:"context"`
		Temperature  *float64 `json:"temperature"`
		ThinkMode    string   `json:"think_mode"`
	}
	if err := json.Unmarshal(metaBytes, &raw); err != nil {
		return CorpusCase{}, fmt.Errorf("server corpus metadata: %w", err)
	}
	if raw.CaseID == "" || raw.Step == "" || raw.Method == "" || raw.Endpoint == "" || raw.Status == "" ||
		raw.SourceRev == "" || !isSHA256(raw.BinarySHA256) || !isSHA256(raw.ModelSHA256) ||
		raw.Backend == "" || raw.Context == 0 {
		return CorpusCase{}, errors.New("server corpus: incomplete provenance metadata")
	}
	statusCode, ok := parseCorpusStatus(raw.Status)
	if !ok {
		return CorpusCase{}, errors.New("server corpus: invalid HTTP status")
	}
	response, responseJSON, err := readCorpusResponse(dir)
	if err != nil {
		return CorpusCase{}, err
	}
	if responseJSON {
		if _, err := canonicalJSON(response); err != nil {
			return CorpusCase{}, fmt.Errorf("server corpus response: %w", err)
		}
	}
	request, err := readCorpusFile(filepath.Join(dir, "request.json"), false)
	if err != nil {
		return CorpusCase{}, err
	}
	if len(request) != 0 {
		if _, err := canonicalJSON(request); err != nil {
			if statusCode >= 200 && statusCode < 300 {
				return CorpusCase{}, fmt.Errorf("server corpus request: %w", err)
			}
		}
	}
	promptTokens, err := readCorpusInts(filepath.Join(dir, "prompt_tokens.txt"))
	if err != nil {
		return CorpusCase{}, err
	}
	tokens, err := readCorpusInts(filepath.Join(dir, "tokens.txt"))
	if err != nil {
		return CorpusCase{}, err
	}
	terminal, err := readCorpusFile(filepath.Join(dir, "terminal.json"), false)
	if err != nil {
		return CorpusCase{}, err
	}
	if len(terminal) != 0 {
		if _, err := canonicalJSON(terminal); err != nil {
			return CorpusCase{}, fmt.Errorf("server corpus terminal: %w", err)
		}
	}
	return CorpusCase{
		CaseID: raw.CaseID, Step: raw.Step, Method: raw.Method, Endpoint: raw.Endpoint, Status: raw.Status,
		Request: request, Response: response, PromptTokens: promptTokens, Tokens: tokens,
		Metadata: CorpusProvenance{SourceRevision: raw.SourceRev, BinarySHA256: raw.BinarySHA256,
			ModelSHA256: raw.ModelSHA256, Backend: raw.Backend, Context: raw.Context,
			Temperature: valueOrZero(raw.Temperature), ThinkMode: raw.ThinkMode},
		Terminal: terminal,
	}, nil
}

func valueOrZero(value *float64) float64 {
	if value == nil {
		return 0
	}
	return *value
}

func readCorpusResponse(dir string) ([]byte, bool, error) {
	response, err := readCorpusFile(filepath.Join(dir, "response.json"), false)
	if err != nil {
		return nil, false, err
	}
	if len(response) != 0 {
		return response, true, nil
	}
	response, err = readCorpusFile(filepath.Join(dir, "response.sse"), true)
	if err != nil {
		return nil, false, err
	}
	if len(response) == 0 {
		return nil, false, errors.New("server corpus: empty response")
	}
	return response, false, nil
}

func loadCorpusProvenance(path string) (CorpusProvenance, error) {
	data, err := readCorpusFile(path, true)
	if err != nil {
		return CorpusProvenance{}, err
	}
	var raw struct {
		SourceRev        string            `json:"source_rev"`
		BinarySHA        string            `json:"binary_sha256"`
		ModelSHA         string            `json:"model_sha256"`
		Backend          string            `json:"backend"`
		Context          uint32            `json:"ctx"`
		Temperature      *float64          `json:"temperature"`
		ThinkMode        string            `json:"think_mode"`
		ThinkModeDefault string            `json:"think_mode_default"`
		ThinkModeByStep  map[string]string `json:"think_mode_by_step"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return CorpusProvenance{}, fmt.Errorf("server corpus provenance: %w", err)
	}
	if raw.SourceRev == "" || !isSHA256(raw.BinarySHA) || !isSHA256(raw.ModelSHA) || raw.Backend == "" || raw.Context == 0 || raw.Temperature == nil || raw.ThinkMode == "" {
		return CorpusProvenance{}, errors.New("server corpus: incomplete root provenance")
	}
	if raw.ThinkMode == "per-case" {
		if raw.ThinkModeDefault == "" || len(raw.ThinkModeByStep) == 0 {
			return CorpusProvenance{}, errors.New("server corpus: incomplete per-case think mode")
		}
		for step, mode := range raw.ThinkModeByStep {
			if step == "" || mode == "" {
				return CorpusProvenance{}, errors.New("server corpus: invalid per-case think mode")
			}
		}
	}
	return CorpusProvenance{SourceRevision: raw.SourceRev, BinarySHA256: raw.BinarySHA, ModelSHA256: raw.ModelSHA, Backend: raw.Backend, Context: raw.Context, Temperature: *raw.Temperature, ThinkMode: raw.ThinkMode, ThinkModeDefault: raw.ThinkModeDefault, ThinkModeByStep: raw.ThinkModeByStep}, nil
}

func validateCorpusBuild(got, root CorpusProvenance, step string) error {
	if got.SourceRevision != root.SourceRevision || got.BinarySHA256 != root.BinarySHA256 ||
		got.ModelSHA256 != root.ModelSHA256 || got.Backend != root.Backend || got.Context != root.Context ||
		got.Temperature != root.Temperature {
		return errors.New("server corpus: case build provenance differs from root")
	}
	if root.ThinkMode == "per-case" {
		want, ok := root.ThinkModeByStep[step]
		if !ok || got.ThinkMode != want {
			return fmt.Errorf("server corpus: case think mode differs at %s", step)
		}
	} else if got.ThinkMode != "" && got.ThinkMode != root.ThinkMode {
		return errors.New("server corpus: case build provenance differs from root")
	}
	return nil
}

// ValidateCorpusProvenance binds a loaded case to an expected oracle. It
// rejects mismatches rather than silently comparing different model builds.
func ValidateCorpusProvenance(got, want CorpusProvenance) error {
	if got.SourceRevision != want.SourceRevision || got.BinarySHA256 != want.BinarySHA256 ||
		got.ModelSHA256 != want.ModelSHA256 || got.Backend != want.Backend || got.Context != want.Context ||
		got.Temperature != want.Temperature || got.ThinkMode != want.ThinkMode || got.ThinkModeDefault != want.ThinkModeDefault ||
		!reflect.DeepEqual(got.ThinkModeByStep, want.ThinkModeByStep) {
		return fmt.Errorf("server corpus provenance mismatch: got=%+v want=%+v", got, want)
	}
	return nil
}

func readCorpusFile(path string, required bool) ([]byte, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) && !required {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read server corpus file %s: %w", filepath.Base(path), err)
	}
	if len(data) > maxCorpusArtifactBytes {
		return nil, fmt.Errorf("server corpus file %s exceeds %d-byte limit", filepath.Base(path), maxCorpusArtifactBytes)
	}
	return data, nil
}

func readCorpusInts(path string) ([]int, error) {
	data, err := readCorpusFile(path, false)
	if err != nil || len(data) == 0 {
		return nil, err
	}
	fields := strings.Fields(string(data))
	values := make([]int, len(fields))
	for i, field := range fields {
		value, err := strconv.Atoi(field)
		if err != nil {
			return nil, fmt.Errorf("server corpus tokens: %w", err)
		}
		values[i] = value
	}
	return values, nil
}

func isSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func parseCorpusStatus(value string) (int, bool) {
	if strings.TrimSpace(value) != value {
		return 0, false
	}
	parts := strings.SplitN(value, " ", 3)
	if len(parts) < 2 || (parts[0] != "HTTP/1.0" && parts[0] != "HTTP/1.1" && parts[0] != "HTTP/2") || len(parts[1]) != 3 {
		return 0, false
	}
	for _, digit := range parts[1] {
		if digit < '0' || digit > '9' {
			return 0, false
		}
	}
	code, err := strconv.Atoi(parts[1])
	if err != nil || code < 100 || code > 599 {
		return 0, false
	}
	if len(parts) == 3 && parts[2] == "" {
		return 0, false
	}
	return code, true
}
