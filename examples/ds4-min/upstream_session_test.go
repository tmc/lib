package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4kvstore"
)

type upstreamFakeBackend struct {
	fakeSessionBackend
	calls int
	err   error
}

func (b *upstreamFakeBackend) LoadUpstream(dsv4kvstore.UpstreamRecord, dsv4kvstore.UpstreamRestoreContract) error {
	b.calls++
	return b.err
}

func upstreamSessionFixture() (dsv4kvstore.UpstreamRecord, upstreamSessionContract) {
	p := make([]byte, 76)
	for i, v := range []uint32{0x34565344, 2, 4, 4, 1, 1, 4, 1, 1, 1, 1, 2, 1} {
		binary.LittleEndian.PutUint32(p[4*i:], v)
	}
	binary.LittleEndian.PutUint32(p[52:], 1)
	binary.LittleEndian.PutUint32(p[56:], 0x3f800000)
	h := sha256.Sum256(p)
	tokens := sha256.Sum256(p[52:56])
	model := sha256.Sum256([]byte("model"))
	c := upstreamSessionContract{ModelSHA256: hex.EncodeToString(model[:]), TokenIDsSHA256: hex.EncodeToString(tokens[:]), PayloadSHA256: hex.EncodeToString(h[:]), Contract: dsv4kvstore.UpstreamRestoreContract{ModelSHA256: model, TokenIDsSHA256: tokens, PayloadABI: 2, QuantBits: 2, Context: 4, PrefillCapacity: 4, CompressedCap: 4, Tokens: 1, Layers: 1, HeadDim: 1, IndexerHeadDim: 1, Vocab: 2, UpstreamRawCapacity: 1, TargetRawCapacity: 1, RawWindow: 1, Ratios: []uint32{0}}}
	r := dsv4kvstore.UpstreamRecord{Metadata: dsv4kvstore.UpstreamMetadata{PayloadABI: 2, QuantBits: 2, Context: 4, Tokens: 1, PayloadBytes: uint64(len(p))}, Payload: p, PayloadSHA256: h}
	return r, c
}

func TestLoadUpstreamSession(t *testing.T) {
	for _, tc := range []struct {
		name    string
		fail    bool
		badHash bool
	}{{"load", false, false}, {"backend fails", true, false}, {"mutated hash", false, true}} {
		t.Run(tc.name, func(t *testing.T) {
			r, c := upstreamSessionFixture()
			b := &upstreamFakeBackend{}
			if tc.fail {
				b.err = errors.New("upload failed")
			}
			if tc.badHash {
				c.Contract.TokenIDsSHA256[0] ^= 1
			}
			s := &session{backend: b, context: 4, tokens: []int{9}, logits: []float32{7}}
			err := loadUpstreamSession(s, r, c)
			if tc.fail || tc.badHash {
				if err == nil {
					t.Fatal("failure accepted")
				}
				if !reflect.DeepEqual(s.tokens, []int{9}) || !reflect.DeepEqual(s.logits, []float32{7}) {
					t.Fatal("failed load changed host state")
				}
			} else if err != nil || !reflect.DeepEqual(s.tokens, []int{1}) || !reflect.DeepEqual(s.logits, []float32{1, 0}) {
				t.Fatalf("load: %v tokens=%v logits=%v", err, s.tokens, s.logits)
			}
			if tc.badHash && b.calls != 0 {
				t.Fatal("malformed receipt reached backend")
			}
		})
	}
}

func TestReadUpstreamSessionContract(t *testing.T) {
	_, c := upstreamSessionFixture()
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := readUpstreamSessionContract(bytes.NewReader(b)); err != nil {
		t.Fatal(err)
	}
	for _, v := range [][]byte{append(append([]byte(nil), b...), []byte("{}")...), []byte(`{"ModelSHA256":"short"}`), bytes.Repeat([]byte(" "), (1<<20)+1)} {
		if _, err := readUpstreamSessionContract(bytes.NewReader(v)); err == nil {
			t.Fatal("invalid manifest accepted")
		}
	}
}

func TestKVLoadContractRequiresPath(t *testing.T) {
	if _, err := parseConfig([]string{"-model", "model.gguf", "-kv-load-contract", "receipt.json"}, bytes.NewReader(nil), io.Discard); err == nil || !strings.Contains(err.Error(), "requires kv-load") {
		t.Fatal("contract without checkpoint accepted")
	}
	c, err := parseConfig([]string{"-model", "model.gguf", "-inspect", "-kv-load-contract", "receipt.json", "-kv-load", "cold.kv"}, bytes.NewReader(nil), io.Discard)
	if err != nil || c.kvLoad != "cold.kv" || c.kvLoadContract != "receipt.json" {
		t.Fatalf("paired flags: %v %+v", err, c)
	}
}

func TestLoadUpstreamUnsupportedPreservesHistory(t *testing.T) {
	r, c := upstreamSessionFixture()
	s := &session{backend: &fakeSessionBackend{}, context: 4, tokens: []int{9}, logits: []float32{7}}
	if err := loadUpstreamSession(s, r, c); err == nil {
		t.Fatal("unsupported backend accepted")
	}
	if !reflect.DeepEqual(s.tokens, []int{9}) || !reflect.DeepEqual(s.logits, []float32{7}) {
		t.Fatal("unsupported backend changed history")
	}
}

func TestLoadUpstreamNonfiniteRejectsBeforeUpload(t *testing.T) {
	r, c := upstreamSessionFixture()
	binary.LittleEndian.PutUint32(r.Payload[56:], 0x7fc00000)
	r.PayloadSHA256 = sha256.Sum256(r.Payload)
	c.PayloadSHA256 = hex.EncodeToString(r.PayloadSHA256[:])
	b := &upstreamFakeBackend{}
	s := &session{backend: b, context: 4, tokens: []int{9}, logits: []float32{7}}
	if err := loadUpstreamSession(s, r, c); err == nil || b.calls != 0 {
		t.Fatal("nonfinite logits reached upload")
	}
	if !reflect.DeepEqual(s.tokens, []int{9}) || !reflect.DeepEqual(s.logits, []float32{7}) {
		t.Fatal("failed admission changed history")
	}
}

func TestLoadUpstreamRawMutationRejectsBeforeUpload(t *testing.T) {
	r, c := upstreamSessionFixture()
	// Mutate a non-token raw tensor byte while leaving the recorded digest
	// and independent receipt untouched. Layout validation must rehash it.
	r.Payload[len(r.Payload)-1] ^= 1
	b := &upstreamFakeBackend{}
	s := &session{backend: b, context: 4, tokens: []int{9}, logits: []float32{7}}
	if err := loadUpstreamSession(s, r, c); err == nil || b.calls != 0 {
		t.Fatal("stale fingerprint reached upload")
	}
	if !reflect.DeepEqual(s.tokens, []int{9}) || !reflect.DeepEqual(s.logits, []float32{7}) {
		t.Fatal("rejected payload changed history")
	}
}
