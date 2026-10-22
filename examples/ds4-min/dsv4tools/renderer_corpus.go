package dsv4tools

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

const (
	// RendererCorpusSchema is the versioned JSONL record schema exchanged by
	// the upstream oracle and the candidate renderer.
	RendererCorpusSchema = "ds4-renderer/v1"
	// RendererCorpusCases is the registered 24-prompt comparison size.
	RendererCorpusCases = 24
	maxRendererLine     = 1 << 20
	maxRendererRecords  = 256
)

// RendererRecord is one oracle or candidate renderer result. RenderedInput is
// retained so the input hash can be independently recomputed; TokenIDs uses
// signed int32 little-endian encoding for the token hash.
type RendererRecord struct {
	Schema              string `json:"schema"`
	Ordinal             int    `json:"ordinal"`
	CaseID              string `json:"case_id"`
	RenderedInput       string `json:"rendered_input"`
	RenderedInputSHA256 string `json:"rendered_input_sha256"`
	TokenIDs            []int  `json:"token_ids"`
	TokenIDsSHA256      string `json:"token_ids_sha256"`
}

// RendererReceipt is the fail-closed result of comparing two JSONL corpora.
// UNMEASURED is returned when a named corpus is unavailable; it is never a
// successful comparison.
type RendererReceipt struct {
	Schema         string `json:"schema"`
	Status         string `json:"status"`
	Reason         string `json:"reason"`
	ExpectedCases  int    `json:"expected_cases"`
	OracleCases    int    `json:"oracle_cases"`
	CandidateCases int    `json:"candidate_cases"`
	ComparedCases  int    `json:"compared_cases"`
	Mismatches     int    `json:"mismatches"`
	OraclePath     string `json:"oracle_path,omitempty"`
	CandidatePath  string `json:"candidate_path,omitempty"`
}

// CompareRendererJSONL compares a registered oracle and candidate corpus.
// Both files must contain exactly RendererCorpusCases records. Missing files
// produce an UNMEASURED receipt; malformed or mismatching files produce FAIL.
func CompareRendererJSONL(oraclePath, candidatePath string) (RendererReceipt, error) {
	receipt := RendererReceipt{Schema: RendererCorpusSchema, ExpectedCases: RendererCorpusCases, OraclePath: oraclePath, CandidatePath: candidatePath}
	oracle, err := readRendererJSONL(oraclePath)
	if errors.Is(err, os.ErrNotExist) {
		receipt.Status, receipt.Reason = "UNMEASURED", "oracle corpus is absent"
		return receipt, nil
	}
	if err != nil {
		receipt.Status, receipt.Reason = "FAIL", "oracle corpus is malformed"
		return receipt, fmt.Errorf("read oracle corpus: %w", err)
	}
	receipt.OracleCases = len(oracle)
	candidate, err := readRendererJSONL(candidatePath)
	if errors.Is(err, os.ErrNotExist) {
		receipt.Status, receipt.Reason = "UNMEASURED", "candidate corpus is absent"
		return receipt, nil
	}
	if err != nil {
		receipt.Status, receipt.Reason = "FAIL", "candidate corpus is malformed"
		return receipt, fmt.Errorf("read candidate corpus: %w", err)
	}
	receipt.CandidateCases = len(candidate)
	if len(oracle) != RendererCorpusCases || len(candidate) != RendererCorpusCases {
		receipt.Status, receipt.Reason = "FAIL", "corpus case count is not 24"
		return receipt, nil
	}
	for i := range oracle {
		receipt.ComparedCases++
		if !sameRendererRecord(oracle[i], candidate[i]) {
			receipt.Mismatches++
		}
	}
	if receipt.Mismatches != 0 {
		receipt.Status, receipt.Reason = "FAIL", "renderer records differ"
		return receipt, nil
	}
	receipt.Status, receipt.Reason = "PASS", "all 24 renderer records match"
	return receipt, nil
}

func readRendererJSONL(name string) ([]RendererRecord, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 4096), maxRendererLine)
	records := make([]RendererRecord, 0, RendererCorpusCases)
	caseIDs := make(map[string]struct{}, RendererCorpusCases)
	for s.Scan() {
		if len(records) == maxRendererRecords {
			return nil, errors.New("too many renderer records")
		}
		var record RendererRecord
		if err := json.Unmarshal(s.Bytes(), &record); err != nil {
			return nil, fmt.Errorf("decode record %d: %w", len(records), err)
		}
		if err := validateRendererRecord(record, len(records)); err != nil {
			return nil, fmt.Errorf("record %d: %w", len(records), err)
		}
		if _, ok := caseIDs[record.CaseID]; ok {
			return nil, fmt.Errorf("duplicate case id %q", record.CaseID)
		}
		caseIDs[record.CaseID] = struct{}{}
		records = append(records, record)
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("scan corpus: %w", err)
	}
	return records, nil
}

func validateRendererRecord(r RendererRecord, ordinal int) error {
	if r.Schema != RendererCorpusSchema {
		return fmt.Errorf("schema %q, want %q", r.Schema, RendererCorpusSchema)
	}
	if r.Ordinal != ordinal || r.CaseID == "" {
		return errors.New("invalid ordinal or case id")
	}
	if r.RenderedInputSHA256 != hashRendererInput(r.RenderedInput) {
		return errors.New("rendered input hash mismatch")
	}
	if r.TokenIDsSHA256 != hashRendererTokens(r.TokenIDs) {
		return errors.New("token ids hash mismatch")
	}
	for _, id := range r.TokenIDs {
		if int64(id) < -1<<31 || int64(id) > 1<<31-1 {
			return errors.New("token id is outside int32 range")
		}
	}
	return nil
}

func sameRendererRecord(a, b RendererRecord) bool {
	if a.Ordinal != b.Ordinal || a.CaseID != b.CaseID || a.RenderedInputSHA256 != b.RenderedInputSHA256 || a.TokenIDsSHA256 != b.TokenIDsSHA256 {
		return false
	}
	if a.RenderedInput != b.RenderedInput || len(a.TokenIDs) != len(b.TokenIDs) {
		return false
	}
	for i := range a.TokenIDs {
		if a.TokenIDs[i] != b.TokenIDs[i] {
			return false
		}
	}
	return true
}

func hashRendererInput(input string) string {
	h := sha256.Sum256([]byte(input))
	return hex.EncodeToString(h[:])
}

func hashRendererTokens(ids []int) string {
	b := make([]byte, 4*len(ids))
	for i, id := range ids {
		binary.LittleEndian.PutUint32(b[i*4:], uint32(int32(id)))
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// WriteRendererJSONL writes records in the registered JSONL schema.
func WriteRendererJSONL(w io.Writer, records []RendererRecord) error {
	if len(records) != RendererCorpusCases {
		return fmt.Errorf("renderer corpus has %d records, want %d", len(records), RendererCorpusCases)
	}
	enc := json.NewEncoder(w)
	for i := range records {
		record := records[i]
		record.Schema = RendererCorpusSchema
		record.Ordinal = i
		record.RenderedInputSHA256 = hashRendererInput(record.RenderedInput)
		record.TokenIDsSHA256 = hashRendererTokens(record.TokenIDs)
		if err := enc.Encode(record); err != nil {
			return fmt.Errorf("encode record %d: %w", i, err)
		}
	}
	return nil
}
