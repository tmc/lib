//go:build darwin && arm64

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
)

type metalUpstreamGateManifest struct {
	RecordPath, ContinuationPath                                          string
	ModelSHA256, TokenIDsSHA256, PayloadSHA256, ContinuationPayloadSHA256 string
	OracleBuild, OracleBinarySHA256                                       string
	Contract                                                              dsv4kvstore.UpstreamRestoreContract
	MaxKL, MaxAbs                                                         float64
}

// TestMetalUpstreamContinuation needs a hash-pinned model and independent
// cold/one-token continuation checkpoints. A host or CUDA receipt is not a
// Metal device qualification; run this opt-in test under the hardware lock.
func TestMetalUpstreamContinuation(t *testing.T) {
	manifestPath := os.Getenv("DS4_METAL_UPSTREAM_RESTORE_MANIFEST")
	if manifestPath == "" {
		t.Skip("qualified Metal restore manifest required")
	}
	modelPath := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	if modelPath == "" {
		t.Fatal("model path required")
	}
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var m metalUpstreamGateManifest
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&m); err != nil {
		t.Fatal(err)
	}
	if decoder.Decode(new(any)) != io.EOF {
		t.Fatal("manifest trailing data")
	}
	if m.MaxKL <= 0 || m.MaxAbs <= 0 || math.IsNaN(m.MaxKL) || math.IsNaN(m.MaxAbs) || math.IsInf(m.MaxKL, 0) || math.IsInf(m.MaxAbs, 0) || m.OracleBuild == "" {
		t.Fatal("explicit oracle and finite positive bounds required")
	}
	decodeHash := func(text string) [32]byte {
		data, err := hex.DecodeString(text)
		if err != nil || len(data) != sha256.Size {
			t.Fatal("full receipt hashes required")
		}
		var h [32]byte
		copy(h[:], data)
		return h
	}
	_ = decodeHash(m.OracleBinarySHA256)
	m.Contract.ModelSHA256, m.Contract.TokenIDsSHA256 = decodeHash(m.ModelSHA256), decodeHash(m.TokenIDsSHA256)
	read := func(path, fingerprint string) dsv4kvstore.UpstreamRecord {
		r, err := dsv4kvstore.ReadUpstreamPath(path, dsv4kvstore.UpstreamInspectOptions{MaxPayloadBytes: uint64(metalStateMaxBytes), MaxFileBytes: uint64(metalStateMaxBytes)})
		if err != nil {
			t.Fatal(err)
		}
		if r.PayloadSHA256 != decodeHash(fingerprint) {
			t.Fatal("checkpoint fingerprint mismatch")
		}
		return r
	}
	record, next := read(m.RecordPath, m.PayloadSHA256), read(m.ContinuationPath, m.ContinuationPayloadSHA256)
	layout, err := dsv4kvstore.ValidateUpstreamRecordLayout(record, m.Contract.Ratios)
	if err != nil {
		t.Fatal(err)
	}
	nextLayout, err := dsv4kvstore.ValidateUpstreamRecordLayout(next, m.Contract.Ratios)
	if err != nil {
		t.Fatal(err)
	}
	geometry := nextLayout.Metadata
	geometry.Tokens, geometry.RawLive = layout.Metadata.Tokens, layout.Metadata.RawLive
	if nextLayout.Metadata.Tokens != layout.Metadata.Tokens+1 || geometry != layout.Metadata || next.Metadata.ModelID != record.Metadata.ModelID || next.Metadata.QuantBits != record.Metadata.QuantBits || next.Metadata.PayloadABI != record.Metadata.PayloadABI || len(next.Trailer) != 0 {
		t.Fatal("continuation geometry mismatch")
	}
	end := 52 + int(layout.Metadata.Tokens)*4
	if !bytes.Equal(record.Payload[52:end], next.Payload[52:end]) {
		t.Fatal("continuation history mismatch")
	}
	token := int(binary.LittleEndian.Uint32(next.Payload[end:]))
	r, err := newMetalRuntime(modelPath, int(m.Contract.Context))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	s := &metalRequestState{runtime: r, owner: &metalStateAdapter{runtime: r}}
	if err := s.LoadUpstream(record, m.Contract); err != nil {
		t.Fatal(err)
	}
	before := append([]float32(nil), r.Logits()...)
	if len(before) != int(layout.Metadata.Vocab) || len(before) == 0 {
		t.Fatal("uploaded logits shape mismatch")
	}
	for i, v := range before {
		if math.Float32bits(v) != binary.LittleEndian.Uint32(record.Payload[end+4*i:]) {
			t.Fatalf("uploaded logit %d differs", i)
		}
	}
	var saved bytes.Buffer
	if err := s.Save(&saved); err != nil {
		t.Fatal(err)
	}
	bad := m.Contract
	bad.TokenIDsSHA256[0] ^= 1
	if s.LoadUpstream(record, bad) == nil {
		t.Fatal("mutant token provenance accepted")
	}
	for i, v := range r.Logits() {
		if math.Float32bits(v) != math.Float32bits(before[i]) {
			t.Fatal("rejected restore altered live logits")
		}
	}
	if r.pos != int(layout.Metadata.Tokens) {
		t.Fatal("rejected restore altered position")
	}
	var after bytes.Buffer
	if err := s.Save(&after); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(saved.Bytes(), after.Bytes()) {
		t.Fatal("rejected restore altered serialized state")
	}
	if err := s.eval(token, int(layout.Metadata.Tokens)); err != nil {
		t.Fatal(err)
	}
	start := 52 + int(nextLayout.Metadata.Tokens)*4
	want := make([]float32, int(nextLayout.Metadata.Vocab))
	for i := range want {
		want[i] = math.Float32frombits(binary.LittleEndian.Uint32(next.Payload[start+4*i:]))
	}
	kl, maxAbs := metalUpstreamAgreement(want, r.Logits())
	t.Logf("oracle=%s restored_tokens=%d forced_token=%d KL=%g max_abs=%g bounds=%g/%g", m.OracleBuild, layout.Metadata.Tokens, token, kl, maxAbs, m.MaxKL, m.MaxAbs)
	if math.IsNaN(kl) || math.IsInf(kl, 0) || kl > m.MaxKL || maxAbs > m.MaxAbs {
		t.Fatal("Metal upstream continuation FAIL")
	}
}

func metalUpstreamAgreement(want, got []float32) (float64, float64) {
	if len(want) == 0 || len(want) != len(got) {
		return math.Inf(1), math.Inf(1)
	}
	a, b, maxAbs := math.Inf(-1), math.Inf(-1), 0.0
	for i, v := range want {
		x, y := float64(v), float64(got[i])
		if math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) {
			return math.Inf(1), math.Inf(1)
		}
		a, b, maxAbs = math.Max(a, x), math.Max(b, y), math.Max(maxAbs, math.Abs(x-y))
	}
	sa, sb := 0.0, 0.0
	for i, v := range want {
		sa += math.Exp(float64(v) - a)
		sb += math.Exp(float64(got[i]) - b)
	}
	la, lb := a+math.Log(sa), b+math.Log(sb)
	kl := 0.0
	for i, v := range want {
		lp, lq := float64(v)-la, float64(got[i])-lb
		kl += math.Exp(lp) * (lp - lq)
	}
	return math.Max(0, kl), maxAbs
}

func TestMetalUpstreamAgreementMutation(t *testing.T) {
	want := []float32{1, 2, 3}
	if kl, m := metalUpstreamAgreement(want, want); kl != 0 || m != 0 {
		t.Fatal("self comparison failed")
	}
	if kl, m := metalUpstreamAgreement(want, []float32{1, 2, 5}); kl <= 0 || m != 2 {
		t.Fatal("arithmetic mutant undetected")
	}
}
