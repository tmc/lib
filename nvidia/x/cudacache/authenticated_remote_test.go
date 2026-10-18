package cudacache_test

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
)

func TestAuthenticatedRemoteTLS(t *testing.T) {
	for _, kind := range []string{cudacache.RemoteArtifact, cudacache.RemoteReceipt} {
		for _, mode := range []string{"signed", "origins-only", "immutable-config", "body-tamper", "signature-tamper", "wrong-key", "wrong-kind", "unknown-signer", "unknown-origin", "missing-proof", "proof-oversize", "proof-trailing", "proof-duplicate", "proof-null", "payload-oversize"} {
			t.Run(kind+"/"+mode, func(t *testing.T) {
				private := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{7}, ed25519.SeedSize))
				public := private.Public().(ed25519.PublicKey)
				key := "signed-key"
				policy := cudacache.SearchPolicyID{Strategy: "synthetic-host-fixture"}
				var body []byte
				if kind == cudacache.RemoteArtifact {
					var err error
					body, err = cudacache.MarshalArtifact(newTestArtifact(t, "signed_fixture", "sm_90"))
					if err != nil {
						t.Fatal(err)
					}
				} else {
					key = cudacache.SearchReceiptKey(policy, "workload")
					body, _ = json.Marshal(cudacache.SearchReceipt{PolicyID: policy, WorkloadKey: "workload", CandidateID: "synthetic", ArtifactKey: "artifact", Epoch: "old-epoch"})
				}
				proofKind, proofKey, origin, signer := kind, key, "producer-A", "key-A"
				switch mode {
				case "wrong-key":
					proofKey = "another-key"
				case "wrong-kind":
					if kind == cudacache.RemoteArtifact {
						proofKind = cudacache.RemoteReceipt
					} else {
						proofKind = cudacache.RemoteArtifact
					}
				case "unknown-signer":
					signer = "unknown"
				case "unknown-origin":
					origin = "producer-B"
				}
				proof, err := cudacache.SignRemoteProof(proofKind, proofKey, origin, signer, body, private)
				if err != nil {
					t.Fatal(err)
				}
				switch mode {
				case "body-tamper":
					body = append(append([]byte(nil), body...), '\n')
				case "signature-tamper":
					var object map[string]json.RawMessage
					json.Unmarshal(proof, &object)
					object["signature"], _ = json.Marshal(make([]byte, ed25519.SignatureSize))
					proof, _ = json.Marshal(object)
				case "proof-oversize":
					proof = bytes.Repeat([]byte("x"), cudacache.MaxRemoteProofBytes+1)
				case "proof-trailing":
					proof = append(proof, '\n')
				case "proof-duplicate":
					proof = append(append([]byte{}, proof[:len(proof)-1]...), []byte(`,"signature":null}`)...)
				case "proof-null":
					proof = []byte(`{"message":null,"signature":null}`)
				}
				server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Method != http.MethodGet {
						t.Errorf("unexpected method %s", r.Method)
						http.Error(w, "method", 405)
						return
					}
					if r.URL.Path == "/proofs/"+kind+"/"+key {
						if mode == "missing-proof" {
							http.NotFound(w, r)
							return
						}
						// Flush headers to exercise unknown Content-Length proof bounds.
						w.(http.Flusher).Flush()
						w.Write(proof)
						return
					}
					prefix := "/artifacts/"
					if kind == cudacache.RemoteReceipt {
						prefix = "/receipts/"
					}
					if r.URL.Path != prefix+key {
						t.Errorf("unexpected path %s", r.URL.Path)
						http.NotFound(w, r)
						return
					}
					w.(http.Flusher).Flush()
					w.Write(body)
				}))
				defer server.Close()
				root := t.TempDir()
				trust := cudacache.TrustPolicy{RequireSigned: mode != "origins-only", AllowedOrigins: []string{"producer-A"}, Signers: map[string]ed25519.PublicKey{"key-A": public}}
				limit := int64(1 << 20)
				if mode == "payload-oversize" {
					limit = int64(len(body) - 1)
				}
				cache, err := cudacache.New(cudacache.Config{RootDir: root, Remote: cudacache.NewHTTPRemoteSource(server.URL, server.Client()), Trust: trust, MaxRemoteDownloadBytes: limit})
				if err != nil {
					t.Fatal(err)
				}
				if mode == "immutable-config" {
					trust.AllowedOrigins[0] = "producer-B"
					public[0] ^= 1
					delete(trust.Signers, "key-A")
				}
				if kind == cudacache.RemoteArtifact {
					_, err = cache.GetArtifact(context.Background(), key)
				} else {
					var receipt *cudacache.SearchReceipt
					receipt, err = cache.GetReceipt(context.Background(), policy, "workload")
					if err == nil && !receipt.IsStale("current-epoch") {
						t.Fatal("stored receipt became current qualification")
					}
				}
				positive := mode == "signed" || mode == "origins-only" || mode == "immutable-config"
				if positive {
					if err != nil {
						t.Fatal(err)
					}
					return
				}
				want := cudacache.ErrUntrustedSource
				if mode == "payload-oversize" {
					want = cudacache.ErrDownloadLimitExceeded
				}
				if !errors.Is(err, want) {
					t.Fatalf("got %v want %v", err, want)
				}
				for _, dir := range []string{"artifacts", "receipts", "quarantine"} {
					entries, err := os.ReadDir(filepath.Join(root, dir))
					if err != nil || len(entries) != 0 {
						t.Fatalf("unverified payload published/decoded in %s: %v %v", dir, entries, err)
					}
				}
			})
		}
	}
}

type proofRemote struct {
	*cudacache.MemoryRegistry
	proof io.ReadCloser
	err   error
}

func (r proofRemote) FetchProof(context.Context, string, string) (io.ReadCloser, error) {
	return r.proof, r.err
}

type closeProof struct {
	io.Reader
	closes   int
	closeErr error
}

func (r *closeProof) Close() error { r.closes++; return r.closeErr }

func TestAuthenticatedRemoteProofOwnership(t *testing.T) {
	for _, mode := range []string{"close-error", "oversize", "invalid", "canceled"} {
		t.Run(mode, func(t *testing.T) {
			private := ed25519.NewKeyFromSeed(make([]byte, ed25519.SeedSize))
			registry := cudacache.NewMemoryRegistry()
			body, _ := cudacache.MarshalArtifact(newTestArtifact(t, "proof_ownership", "sm_90"))
			registry.PublishArtifact(context.Background(), "key", body)
			raw, _ := cudacache.SignRemoteProof(cudacache.RemoteArtifact, "key", "producer", "signer", body, private)
			if mode == "oversize" {
				raw = bytes.Repeat([]byte("x"), cudacache.MaxRemoteProofBytes+1)
			}
			if mode == "invalid" {
				raw = []byte("not-json")
			}
			stream := &closeProof{Reader: bytes.NewReader(raw)}
			if mode == "close-error" {
				stream.closeErr = errors.New("synthetic close failure")
			}
			cache, err := cudacache.New(cudacache.Config{RootDir: t.TempDir(), Remote: proofRemote{MemoryRegistry: registry, proof: stream}, Trust: cudacache.TrustPolicy{RequireSigned: true, Signers: map[string]ed25519.PublicKey{"signer": private.Public().(ed25519.PublicKey)}}})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if mode == "canceled" {
				cancel()
			}
			_, err = cache.GetArtifact(ctx, "key")
			want := cudacache.ErrUntrustedSource
			closes := 1
			if mode == "canceled" {
				want = context.Canceled
				closes = 0
			}
			if !errors.Is(err, want) || stream.closes != closes {
				t.Fatalf("err=%v closes=%d", err, stream.closes)
			}
		})
	}
}

func TestAuthenticatedRemoteConstructionRefusals(t *testing.T) {
	for _, mode := range []string{"no-keys", "short-key", "no-proof-interface"} {
		t.Run(mode, func(t *testing.T) {
			trust := cudacache.TrustPolicy{RequireSigned: true}
			var remote cudacache.RemoteSource = cudacache.NewHTTPRemoteSource("https://unused.invalid", nil)
			if mode == "short-key" {
				trust.Signers = map[string]ed25519.PublicKey{"key": {1}}
			}
			if mode == "no-proof-interface" {
				trust.Signers = map[string]ed25519.PublicKey{"key": make([]byte, ed25519.PublicKeySize)}
				remote = cudacache.NewMemoryRegistry()
			}
			_, err := cudacache.New(cudacache.Config{RootDir: t.TempDir(), Remote: remote, Trust: trust})
			if !errors.Is(err, cudacache.ErrUntrustedSource) {
				t.Fatal(err)
			}
		})
	}
}

func TestAuthenticatedRemoteSigningRefusals(t *testing.T) {
	private := ed25519.NewKeyFromSeed(make([]byte, ed25519.SeedSize))
	for _, kind := range []string{"unknown", cudacache.RemoteArtifact} {
		key := "key"
		if kind == cudacache.RemoteArtifact {
			key = "../escape"
		}
		if _, err := cudacache.SignRemoteProof(kind, key, "producer", "signer", nil, private); !errors.Is(err, cudacache.ErrUntrustedSource) {
			t.Fatal(err)
		}
	}
	if _, err := cudacache.SignRemoteProof(cudacache.RemoteArtifact, "key", strings.Repeat("x", 513), "signer", nil, private); !errors.Is(err, cudacache.ErrUntrustedSource) {
		t.Fatal(err)
	}
}

func TestAuthenticatedRemoteProofCancellation(t *testing.T) {
	body, _ := cudacache.MarshalArtifact(newTestArtifact(t, "cancel_proof", "sm_90"))
	started := make(chan struct{})
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/proofs/") {
			w.(http.Flusher).Flush()
			close(started)
			<-r.Context().Done()
			return
		}
		w.Write(body)
	}))
	defer server.Close()
	private := ed25519.NewKeyFromSeed(make([]byte, ed25519.SeedSize))
	cache, err := cudacache.New(cudacache.Config{RootDir: t.TempDir(), Remote: cudacache.NewHTTPRemoteSource(server.URL, server.Client()), Trust: cudacache.TrustPolicy{RequireSigned: true, Signers: map[string]ed25519.PublicKey{"key": private.Public().(ed25519.PublicKey)}}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan error, 1)
	go func() { _, err := cache.GetArtifact(ctx, "key"); result <- err }()
	<-started
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestAuthenticatedRemoteLocalOwnerRoot(t *testing.T) {
	// RootDir is the caller's local trust boundary. Local entries are not
	// retroactively upgraded to signed provenance or current qualification.
	private := ed25519.NewKeyFromSeed(make([]byte, ed25519.SeedSize))
	cache, err := cudacache.New(cudacache.Config{RootDir: t.TempDir(), Remote: proofRemote{MemoryRegistry: cudacache.NewMemoryRegistry(), err: errors.New("must not fetch")}, Trust: cudacache.TrustPolicy{RequireSigned: true, Signers: map[string]ed25519.PublicKey{"key": private.Public().(ed25519.PublicKey)}}})
	if err != nil {
		t.Fatal(err)
	}
	art := newTestArtifact(t, "local_owner", "sm_90")
	if err := cache.PutArtifact(context.Background(), "local", art); err != nil {
		t.Fatal(err)
	}
	if _, err := cache.GetArtifact(context.Background(), "local"); err != nil {
		t.Fatal(err)
	}
}

// proofWireSize is an independent size oracle for the documented v1 JSON
// fields, not a separate cryptographic/signature implementation.
func proofWireSize(t testing.TB, kind, key, origin, signer string) int {
	t.Helper()
	message := struct {
		Domain  string `json:"domain"`
		Version int    `json:"version"`
		Kind    string `json:"kind"`
		Key     string `json:"key"`
		Origin  string `json:"origin"`
		KeyID   string `json:"key_id"`
		SHA256  string `json:"sha256"`
	}{"cudacache.remote-entry", 1, kind, key, origin, signer, strings.Repeat("0", 64)}
	wire := struct {
		Message   any    `json:"message"`
		Signature []byte `json:"signature"`
	}{message, make([]byte, ed25519.SignatureSize)}
	data, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	return len(data)
}

func TestAuthenticatedRemoteSigningContract(t *testing.T) {
	private := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{9}, ed25519.SeedSize))
	public := private.Public().(ed25519.PublicKey)
	for _, kind := range []string{cudacache.RemoteArtifact, cudacache.RemoteReceipt} {
		for _, name := range []string{"normal-utf8", "escaped-oversize", "invalid-origin", "invalid-keyid", "boundary-minus1", "boundary", "boundary-plus1"} {
			t.Run(kind+"/"+name, func(t *testing.T) {
				policy := cudacache.SearchPolicyID{Strategy: "synthetic-signing-boundary"}
				key := strings.Repeat("k", 200)
				var body []byte
				if kind == cudacache.RemoteArtifact {
					body, _ = cudacache.MarshalArtifact(newTestArtifact(t, "wire_boundary", "sm_90"))
				} else {
					key = cudacache.SearchReceiptKey(policy, "workload")
					body, _ = json.Marshal(cudacache.SearchReceipt{PolicyID: policy, WorkloadKey: "workload", CandidateID: "fixture", ArtifactKey: "artifact"})
				}
				origin, signer := "生成者<東京>", "署名鍵-é"
				refused := false
				target := 0
				switch name {
				case "escaped-oversize":
					origin = strings.Repeat("<", 512)
					signer = strings.Repeat("<", 128)
					refused = true
				case "invalid-origin":
					origin = string([]byte{0xff})
					refused = true
				case "invalid-keyid":
					signer = string([]byte{0xff})
					refused = true
				case "boundary-minus1":
					target = cudacache.MaxRemoteProofBytes - 1
				case "boundary":
					target = cudacache.MaxRemoteProofBytes
				case "boundary-plus1":
					target = cudacache.MaxRemoteProofBytes + 1
					refused = true
				}
				if target != 0 {
					signer = strings.Repeat("<", 128)
					fixed := proofWireSize(t, kind, key, "", signer)
					found := false
					for n := 1; n <= 512; n++ {
						expansion := target - fixed - n
						if expansion < 0 || expansion%5 != 0 || expansion/5 > n {
							continue
						}
						escaped := expansion / 5
						origin = strings.Repeat("<", escaped) + strings.Repeat("o", n-escaped)
						found = true
						break
					}
					if !found || proofWireSize(t, kind, key, origin, signer) != target {
						t.Fatal("exact wire boundary not constructed")
					}
				}
				proof, err := cudacache.SignRemoteProof(kind, key, origin, signer, body, private)
				if refused {
					if !errors.Is(err, cudacache.ErrUntrustedSource) || len(proof) != 0 {
						t.Fatalf("refusal proof=%d err=%v", len(proof), err)
					}
					return
				}
				if err != nil || len(proof) > cudacache.MaxRemoteProofBytes || target != 0 && len(proof) != target {
					t.Fatalf("successful proof=%d target=%d err=%v", len(proof), target, err)
				}
				registry := cudacache.NewMemoryRegistry()
				if kind == cudacache.RemoteArtifact {
					err = registry.PublishArtifact(context.Background(), key, body)
				} else {
					err = registry.PublishReceipt(context.Background(), key, body)
				}
				if err != nil {
					t.Fatal(err)
				}
				stream := &closeProof{Reader: bytes.NewReader(proof)}
				cache, err := cudacache.New(cudacache.Config{RootDir: t.TempDir(), Remote: proofRemote{MemoryRegistry: registry, proof: stream}, Trust: cudacache.TrustPolicy{RequireSigned: true, AllowedOrigins: []string{origin}, Signers: map[string]ed25519.PublicKey{signer: public}}})
				if err != nil {
					t.Fatal(err)
				}
				if kind == cudacache.RemoteArtifact {
					_, err = cache.GetArtifact(context.Background(), key)
				} else {
					_, err = cache.GetReceipt(context.Background(), policy, "workload")
				}
				if err != nil || stream.closes != 1 {
					t.Fatalf("public signer/receiver roundtrip: err=%v closes=%d", err, stream.closes)
				}
			})
		}
	}
}

func TestAuthenticatedRemoteUTF8TrustRefusal(t *testing.T) {
	private := ed25519.NewKeyFromSeed(make([]byte, ed25519.SeedSize))
	for _, field := range []string{"origin", "signer"} {
		t.Run(field, func(t *testing.T) {
			origin, signer := "producer", "key"
			if field == "origin" {
				origin = string([]byte{0xff})
			} else {
				signer = string([]byte{0xff})
			}
			_, err := cudacache.New(cudacache.Config{RootDir: t.TempDir(), Remote: proofRemote{MemoryRegistry: cudacache.NewMemoryRegistry()}, Trust: cudacache.TrustPolicy{AllowedOrigins: []string{origin}, Signers: map[string]ed25519.PublicKey{signer: private.Public().(ed25519.PublicKey)}}})
			if !errors.Is(err, cudacache.ErrUntrustedSource) {
				t.Fatal(err)
			}
		})
	}
}
