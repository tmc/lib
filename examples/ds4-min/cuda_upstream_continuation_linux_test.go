//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4kvstore"
	"github.com/tmc/lib/gguf"
)

// The manifest is an independent receipt, not metadata derived from the KVC
// under test. Contract geometry and all four hashes must be pinned before the
// run. No tolerance defaults silently turn a diagnostic into qualification.
type cudaUpstreamGateManifest struct {
	RecordPath                string
	ContinuationPath          string
	ModelSHA256               string
	TokenIDsSHA256            string
	PayloadSHA256             string
	ContinuationPayloadSHA256 string
	Contract                  dsv4kvstore.UpstreamRestoreContract
	MaxKL                     float64
	MaxAbs                    float64
}

// TestCUDAUpstreamContinuation is a fresh-process, opt-in device gate. Invoke
// the test binary under gpu-lock with DS4_UPSTREAM_RESTORE_MANIFEST naming a
// trusted JSON receipt and DS4_GGUF_PROFILE_MODEL naming the exact model.
// It restores the cold record, verifies the uploaded logits exactly, then
// teacher-forces the one token in the upstream continuation checkpoint and
// compares against that checkpoint's retained logits.
func TestCUDAUpstreamContinuation(t *testing.T) {
	manifestPath, modelPath := os.Getenv("DS4_UPSTREAM_RESTORE_MANIFEST"), os.Getenv("DS4_GGUF_PROFILE_MODEL")
	if manifestPath == "" {
		t.Skip("qualified upstream manifest required")
	}
	if modelPath == "" {
		t.Fatal("upstream restore gate requested without model")
	}
	b, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var m cudaUpstreamGateManifest
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		t.Fatal(err)
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		t.Fatal("manifest has trailing data")
	}
	if m.MaxKL <= 0 || m.MaxAbs <= 0 || math.IsNaN(m.MaxKL) || math.IsNaN(m.MaxAbs) || math.IsInf(m.MaxKL, 0) || math.IsInf(m.MaxAbs, 0) {
		t.Fatal("explicit finite positive agreement bounds required")
	}
	decodeHash := func(s string) [32]byte {
		v, err := hex.DecodeString(s)
		if err != nil || len(v) != sha256.Size {
			t.Fatal("invalid receipt hash")
		}
		var h [32]byte
		copy(h[:], v)
		return h
	}
	m.Contract.ModelSHA256 = decodeHash(m.ModelSHA256)
	m.Contract.TokenIDsSHA256 = decodeHash(m.TokenIDsSHA256)
	read := func(path, fingerprint string) dsv4kvstore.UpstreamRecord {
		record, err := dsv4kvstore.ReadUpstreamPath(path, dsv4kvstore.UpstreamInspectOptions{MaxPayloadBytes: cudaStateMaxBytes, MaxFileBytes: cudaStateMaxBytes})
		if err != nil {
			t.Fatal(err)
		}
		if record.PayloadSHA256 != decodeHash(fingerprint) {
			t.Fatal("record payload hash does not match receipt")
		}
		return record
	}
	record := read(m.RecordPath, m.PayloadSHA256)
	next := read(m.ContinuationPath, m.ContinuationPayloadSHA256)
	layout, err := dsv4kvstore.ValidateUpstreamRecordLayout(record, m.Contract.Ratios)
	if err != nil {
		t.Fatal(err)
	}
	if err := dsv4kvstore.ValidateUpstreamRestoreContract(layout, m.Contract); err != nil {
		t.Fatal(err)
	}
	nextLayout, err := dsv4kvstore.ValidateUpstreamRecordLayout(next, m.Contract.Ratios)
	if err != nil {
		t.Fatal(err)
	}
	geometry := nextLayout.Metadata
	geometry.Tokens = layout.Metadata.Tokens
	geometry.RawLive = layout.Metadata.RawLive
	if nextLayout.Metadata.Tokens != layout.Metadata.Tokens+1 || geometry != layout.Metadata || next.Metadata.QuantBits != record.Metadata.QuantBits || next.Metadata.PayloadABI != record.Metadata.PayloadABI || next.Metadata.ModelID != record.Metadata.ModelID || len(next.Trailer) != 0 {
		t.Fatal("continuation geometry mismatch")
	}
	end := 52 + int(layout.Metadata.Tokens)*4
	if !bytes.Equal(record.Payload[52:end], next.Payload[52:end]) {
		t.Fatal("continuation history mismatch")
	}
	token := int(binary.LittleEndian.Uint32(next.Payload[end:]))
	f, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		t.Fatal(err)
	}
	g, err := gguf.Read(f, info.Size())
	if err == nil {
		_, err = ReadConfig(g)
	}
	closeErr := f.Close()
	if err != nil {
		t.Fatalf("model metadata admission: %v", err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	r, err := newCUDARuntime(modelPath, int(m.Contract.Context))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	}()
	s := &cudaRequestState{runtime: r, owner: &cudaStateAdapter{runtime: r}}
	if err := s.LoadUpstream(record, m.Contract); err != nil {
		t.Fatal(err)
	}
	if err := verifyCUDAUpstreamImport(record.Payload, layout, &r.state); err != nil {
		t.Fatal(err)
	}
	t.Logf("imported_state_sections=%d exact_bits=PASS", len(layout.Sections))
	want, err := upstreamFloatValues(record.Payload[end : end+int(layout.Metadata.Vocab)*4])
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range r.logits {
		if math.Float32bits(v) != math.Float32bits(want[i]) {
			t.Fatalf("restored logits differ at %d", i)
		}
	}
	deviceLogits := make([]float32, len(want))
	if err := r.state.logits.CopyToHost(deviceLogits); err != nil {
		t.Fatal(err)
	}
	for i, v := range deviceLogits {
		if math.Float32bits(v) != math.Float32bits(want[i]) {
			t.Fatalf("device logits differ at %d", i)
		}
	}
	// A malformed receipt must fail without changing the live frontier/logits.
	bad := m.Contract
	bad.TokenIDsSHA256[0] ^= 1
	if err := s.LoadUpstream(record, bad); err == nil {
		t.Fatal("mutant token hash accepted")
	}
	if r.pos != int(layout.Metadata.Tokens) {
		t.Fatal("failed load changed position")
	}
	for i, v := range r.logits {
		if math.Float32bits(v) != math.Float32bits(want[i]) {
			t.Fatal("failed load changed logits")
		}
	}
	if err := s.eval(token, int(layout.Metadata.Tokens)); err != nil {
		t.Fatal(err)
	}
	start := 52 + int(nextLayout.Metadata.Tokens)*4
	want, err = upstreamFloatValues(next.Payload[start : start+int(nextLayout.Metadata.Vocab)*4])
	if err != nil {
		t.Fatal(err)
	}
	kl, maxAbs := cudaUpstreamAgreement(want, r.logits)
	if path := os.Getenv("DS4_UPSTREAM_RESTORE_LOGITS"); path != "" {
		data := make([]byte, 4*len(r.logits))
		for i, v := range r.logits {
			binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(v))
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		t.Logf("continuation_logits=%s sha256=%x", path, sha256.Sum256(data))
	}
	t.Logf("restored_tokens=%d forced_token=%d KL=%.9g max_abs=%.9g bounds=%.9g/%.9g", layout.Metadata.Tokens, token, kl, maxAbs, m.MaxKL, m.MaxAbs)
	if math.IsNaN(kl) || math.IsInf(kl, 0) || kl > m.MaxKL || maxAbs > m.MaxAbs {
		t.Fatal("upstream continuation agreement fails")
	}
}

func cudaUpstreamAgreement(want, got []float32) (float64, float64) {
	if len(want) != len(got) || len(want) == 0 {
		return math.Inf(1), math.Inf(1)
	}
	a, b := math.Inf(-1), math.Inf(-1)
	maxAbs := 0.0
	for i, x := range want {
		y := float64(got[i])
		v := float64(x)
		if math.IsNaN(v) || math.IsNaN(y) || math.IsInf(v, 0) || math.IsInf(y, 0) {
			return math.Inf(1), math.Inf(1)
		}
		a = math.Max(a, v)
		b = math.Max(b, y)
		maxAbs = math.Max(maxAbs, math.Abs(v-y))
	}
	sa, sb := 0.0, 0.0
	for i, x := range want {
		sa += math.Exp(float64(x) - a)
		sb += math.Exp(float64(got[i]) - b)
	}
	kl := 0.0
	la, lb := a+math.Log(sa), b+math.Log(sb)
	for i, x := range want {
		lp := float64(x) - la
		lq := float64(got[i]) - lb
		kl += math.Exp(lp) * (lp - lq)
	}
	return math.Max(0, kl), maxAbs
}

func TestCUDAUpstreamAgreement(t *testing.T) {
	want := []float32{1, 2, 3}
	kl, maxAbs := cudaUpstreamAgreement(want, want)
	if kl != 0 || maxAbs != 0 {
		t.Fatal("self comparison fails")
	}
	kl, maxAbs = cudaUpstreamAgreement(want, []float32{1, 2, 5})
	if kl <= 0 || maxAbs != 2 {
		t.Fatal("mutant not detected")
	}
}
