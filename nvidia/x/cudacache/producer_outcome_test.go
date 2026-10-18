package cudacache_test

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
)

type producerTransport func(*http.Request) (*http.Response, error)

func (f producerTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.Body != nil {
		r.Body.Close()
	}
	return f(r)
}

func producerHTTP(t *testing.T, body *outcomeReader, status int, size int64, before func()) *cudacache.HTTPRemoteSource {
	t.Helper()
	return cudacache.NewHTTPRemoteSource("https://in-process.invalid", &http.Client{Transport: producerTransport(func(*http.Request) (*http.Response, error) {
		if before != nil {
			before()
		}
		return &http.Response{StatusCode: status, Body: body, ContentLength: size, Header: make(http.Header)}, nil
	})})
}

func TestProducerHTTPResponseOwnership(t *testing.T) {
	readErr, closeErr := errors.New("read cause"), errors.New("close cause")
	for _, method := range []string{"artifact", "receipt", "proof", "publish artifact", "publish receipt"} {
		for _, mode := range []string{"success", "created", "close", "missing", "status", "cancel", "read", "read cancel", "close cancel", "limit"} {
			if mode == "created" && !strings.HasPrefix(method, "publish") || (mode == "read" || mode == "read cancel") && method != "receipt" || mode == "close cancel" && (method == "artifact" || method == "proof") || mode == "limit" && method != "receipt" && method != "proof" {
				continue
			}
			t.Run(method+"/"+mode, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				r := &outcomeReader{Reader: strings.NewReader("payload")}
				status, size := http.StatusOK, int64(7)
				var before func()
				var want []error
				rejected := false
				switch mode {
				case "created":
					status = http.StatusCreated
				case "close":
					r.closeErr = closeErr
					want = []error{closeErr}
					rejected = true
				case "missing":
					status = http.StatusNotFound
					rejected = true
					if method == "artifact" || method == "receipt" {
						want = []error{cudacache.ErrRemoteNotFound}
					} else if method == "proof" {
						want = []error{cudacache.ErrUntrustedSource}
					}
					r.closeErr = closeErr
					want = append(want, closeErr)
				case "status":
					status = http.StatusServiceUnavailable
					r.closeErr = closeErr
					want = []error{closeErr}
					rejected = true
					if method == "proof" {
						want = append(want, cudacache.ErrUntrustedSource)
					}
				case "cancel":
					before = cancel
					r.closeErr = closeErr
					want = []error{context.Canceled, closeErr}
					rejected = true
				case "read", "read cancel":
					r.readErr = readErr
					r.closeErr = closeErr
					want = []error{readErr, closeErr}
					rejected = true
					if mode == "read cancel" {
						r.read = cancel
						want = append(want, context.Canceled)
					}
				case "close cancel":
					r.close = cancel
					r.closeErr = closeErr
					want = []error{context.Canceled, closeErr}
					rejected = true
				case "limit":
					size = cudacache.DefaultMaxRemoteDownloadBytes + 1
					r.closeErr = closeErr
					want = []error{closeErr}
					rejected = true
					if method == "receipt" {
						want = append(want, cudacache.ErrDownloadLimitExceeded)
					} else {
						want = append(want, cudacache.ErrUntrustedSource)
					}
				}
				h := producerHTTP(t, r, status, size, before)
				var err error
				var stream io.ReadCloser
				var data []byte
				switch method {
				case "artifact":
					stream, _, err = h.FetchArtifact(ctx, "key")
				case "receipt":
					data, err = h.FetchReceipt(ctx, "key")
				case "proof":
					stream, err = h.FetchProof(ctx, cudacache.RemoteArtifact, "key")
				case "publish artifact":
					err = h.PublishArtifact(ctx, "key", []byte("upload"))
				case "publish receipt":
					err = h.PublishReceipt(ctx, "key", []byte("upload"))
				}
				// A streaming success transfers ownership; its Close failure belongs to
				// the caller and does not make the producer's transfer an error.
				if mode == "close" && (method == "artifact" || method == "proof") {
					if err != nil || stream != r || r.closes != 0 {
						t.Fatalf("transfer: stream=%v err=%v closes=%d", stream, err, r.closes)
					}
					if !errors.Is(stream.Close(), closeErr) {
						t.Fatal("caller lost Close error")
					}
					return
				}
				if rejected {
					if err == nil || stream != nil || data != nil {
						t.Fatalf("accepted failed response: stream=%v data=%q err=%v", stream, data, err)
					}
					for _, e := range want {
						if !errors.Is(err, e) {
							t.Errorf("lost %v: %v", e, err)
						}
					}
					if errors.Is(err, cudacache.ErrRemoteUnavailable) {
						t.Errorf("producer invented outage: %v", err)
					}
				} else {
					if err != nil {
						t.Fatal(err)
					}
					if stream != nil {
						if r.closes != 0 {
							t.Fatal("producer closed transferred reader")
						}
						raw, e := io.ReadAll(stream)
						if e != nil || string(raw) != "payload" {
							t.Fatalf("payload=%q err=%v", raw, e)
						}
						if e := stream.Close(); e != nil {
							t.Fatal(e)
						}
					} else if method == "receipt" && string(data) != "payload" {
						t.Fatalf("bytes=%q", data)
					}
				}
				if r.closes != 1 {
					t.Errorf("Close calls=%d", r.closes)
				}
				if (mode == "missing" || mode == "status" || mode == "limit") && r.reads != 0 {
					t.Error("read rejected response")
				}
			})
		}
	}
}

func TestProducerHTTPClientErrorOwnership(t *testing.T) {
	cause := errors.New("redirect refused")
	for _, method := range []string{"artifact", "receipt", "proof", "publish artifact", "publish receipt"} {
		t.Run(method, func(t *testing.T) {
			body := &outcomeReader{Reader: strings.NewReader("redirect")}
			calls := 0
			client := &http.Client{Transport: producerTransport(func(*http.Request) (*http.Response, error) {
				calls++
				return &http.Response{StatusCode: http.StatusTemporaryRedirect, Body: body, ContentLength: 8, Header: http.Header{"Location": []string{"https://in-process.invalid/redirect"}}}, nil
			}), CheckRedirect: func(*http.Request, []*http.Request) error { return cause }}
			h := cudacache.NewHTTPRemoteSource("https://in-process.invalid", client)
			var err error
			switch method {
			case "artifact":
				_, _, err = h.FetchArtifact(context.Background(), "key")
			case "receipt":
				_, err = h.FetchReceipt(context.Background(), "key")
			case "proof":
				_, err = h.FetchProof(context.Background(), cudacache.RemoteArtifact, "key")
			case "publish artifact":
				err = h.PublishArtifact(context.Background(), "key", nil)
			case "publish receipt":
				err = h.PublishReceipt(context.Background(), "key", nil)
			}
			if !errors.Is(err, cause) || body.closes != 1 || calls != 1 {
				t.Fatalf("err=%v closes=%d calls=%d", err, body.closes, calls)
			}
		})
	}
}

func TestProducerReceiptFailureNotPublished(t *testing.T) {
	policy := cudacache.SearchPolicyID{Strategy: "beam", ScoringEpoch: "producer"}
	raw, err := json.Marshal(&cudacache.SearchReceipt{PolicyID: policy, WorkloadKey: "work", CandidateID: "candidate", ArtifactKey: "artifact"})
	if err != nil {
		t.Fatal(err)
	}
	cause := errors.New("response close")
	for _, mode := range []string{"close", "read", "missing", "limit", "corrupt", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			body := &outcomeReader{Reader: bytes.NewReader(raw), closeErr: cause}
			status, size := http.StatusOK, int64(len(raw))
			want := []error{cause}
			outage := false
			switch mode {
			case "close":
				outage = true
			case "read":
				body.readErr = io.ErrUnexpectedEOF
				want = append(want, io.ErrUnexpectedEOF)
				outage = true
			case "missing":
				status = http.StatusNotFound
				want = append(want, cudacache.ErrRemoteNotFound, cudacache.ErrNotFound)
			case "limit":
				size = cudacache.DefaultMaxRemoteDownloadBytes + 1
				want = append(want, cudacache.ErrDownloadLimitExceeded)
			case "corrupt":
				body.Reader = strings.NewReader("{")
				body.closeErr = nil
				want = []error{cudacache.ErrCorruptReceipt}
			case "cancel":
				body.close = cancel
				want = append(want, context.Canceled)
			}
			h := producerHTTP(t, body, status, size, nil)
			root := t.TempDir()
			c, err := cudacache.New(cudacache.Config{RootDir: root, Remote: h})
			if err != nil {
				t.Fatal(err)
			}
			got, err := c.GetReceipt(ctx, policy, "work")
			if got != nil {
				t.Fatal("failed response returned receipt")
			}
			checkOutcome(t, err, want, outage)
			if body.closes != 1 {
				t.Fatalf("Close calls=%d", body.closes)
			}
			fresh, err := cudacache.New(cudacache.Config{RootDir: root})
			if err != nil {
				t.Fatal(err)
			}
			if got, err := fresh.GetReceipt(context.Background(), policy, "work"); got != nil || !errors.Is(err, cudacache.ErrNotFound) {
				t.Fatalf("failure persisted: %v", err)
			}
		})
	}
}

func TestProducerProofCauses(t *testing.T) {
	private := ed25519.NewKeyFromSeed(make([]byte, ed25519.SeedSize))
	fetchErr, readErr, closeErr := errors.New("proof fetch"), errors.New("proof read"), errors.New("proof close")
	for _, kind := range []string{cudacache.RemoteArtifact, cudacache.RemoteReceipt} {
		for _, mode := range []string{"success", "fetch reader", "fetch nil", "read close", "close", "cancel fetch", "cancel read", "cancel close", "provider canceled", "provider deadline", "missing", "limit", "corrupt", "oversize", "invalid"} {
			t.Run(kind+"/"+mode, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				policy := cudacache.SearchPolicyID{Strategy: "beam", ScoringEpoch: "proof-producer"}
				key := "key"
				if kind == cudacache.RemoteReceipt {
					key = cudacache.SearchReceiptKey(policy, "work")
				}
				var raw []byte
				if kind == cudacache.RemoteArtifact {
					raw, _ = cudacache.MarshalArtifact(newTestArtifact(t, "proof_producer", "sm_90"))
				} else {
					raw, _ = json.Marshal(&cudacache.SearchReceipt{PolicyID: policy, WorkloadKey: "work", CandidateID: "candidate", ArtifactKey: "artifact"})
				}
				proof, err := cudacache.SignRemoteProof(kind, key, "producer", "signer", raw, private)
				if err != nil {
					t.Fatal(err)
				}
				reader := &outcomeReader{Reader: bytes.NewReader(proof)}
				var providerErr error
				want := []error{cudacache.ErrUntrustedSource}
				nilReader := false
				switch mode {
				case "success":
					want = nil
				case "fetch reader":
					providerErr = fetchErr
					reader.closeErr = closeErr
					want = append(want, fetchErr, closeErr)
				case "fetch nil":
					providerErr = fetchErr
					nilReader = true
					want = append(want, fetchErr)
				case "read close":
					reader.readErr = readErr
					reader.closeErr = closeErr
					want = append(want, readErr, closeErr)
				case "close":
					reader.closeErr = closeErr
					want = append(want, closeErr)
				case "cancel fetch":
					providerErr = fetchErr
					reader.closeErr = closeErr
					want = []error{context.Canceled, fetchErr, closeErr}
				case "cancel read":
					reader.read = cancel
					reader.readErr = readErr
					reader.closeErr = closeErr
					want = []error{context.Canceled, readErr, closeErr}
				case "cancel close":
					reader.close = cancel
					reader.closeErr = closeErr
					want = []error{context.Canceled, closeErr}
				case "provider canceled":
					providerErr = context.Canceled
					reader.closeErr = closeErr
					want = []error{context.Canceled, closeErr}
				case "provider deadline":
					reader.readErr = context.DeadlineExceeded
					reader.closeErr = closeErr
					want = []error{context.DeadlineExceeded, closeErr}
				case "missing":
					providerErr = cudacache.ErrRemoteNotFound
					reader.closeErr = closeErr
					want = append(want, cudacache.ErrRemoteNotFound, closeErr)
				case "limit":
					providerErr = cudacache.ErrDownloadLimitExceeded
					reader.closeErr = closeErr
					want = append(want, cudacache.ErrDownloadLimitExceeded, closeErr)
				case "corrupt":
					providerErr = cudacache.ErrCorruptArtifact
					reader.closeErr = closeErr
					want = append(want, cudacache.ErrCorruptArtifact, closeErr)
				case "oversize":
					reader.Reader = bytes.NewReader(bytes.Repeat([]byte("x"), cudacache.MaxRemoteProofBytes+1))
				case "invalid":
					reader.Reader = strings.NewReader("{")
				}
				source := &outcomeSource{artifact: func(context.Context) (io.ReadCloser, int64, error) {
					return io.NopCloser(bytes.NewReader(raw)), int64(len(raw)), nil
				}, receipt: func(context.Context) ([]byte, error) { return raw, nil }, proof: func(context.Context, string, string) (io.ReadCloser, error) {
					if mode == "cancel fetch" {
						cancel()
					}
					if nilReader {
						return nil, providerErr
					}
					return reader, providerErr
				}}
				root := t.TempDir()
				c, err := cudacache.New(cudacache.Config{RootDir: root, Remote: source, Trust: cudacache.TrustPolicy{RequireSigned: true, AllowedOrigins: []string{"producer"}, Signers: map[string]ed25519.PublicKey{"signer": private.Public().(ed25519.PublicKey)}}})
				if err != nil {
					t.Fatal(err)
				}
				if kind == cudacache.RemoteArtifact {
					_, err = c.GetArtifact(ctx, key)
				} else {
					_, err = c.GetReceipt(ctx, policy, "work")
				}
				if want == nil {
					if err != nil {
						t.Fatal(err)
					}
				} else {
					checkOutcome(t, err, want, false)
					if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
						if errors.Is(err, cudacache.ErrUntrustedSource) {
							t.Errorf("cancellation classified untrusted: %v", err)
						}
					}
					fresh, e := cudacache.New(cudacache.Config{RootDir: root})
					if e != nil {
						t.Fatal(e)
					}
					if kind == cudacache.RemoteArtifact {
						_, e = fresh.GetArtifact(context.Background(), key)
					} else {
						_, e = fresh.GetReceipt(context.Background(), policy, "work")
					}
					if !errors.Is(e, cudacache.ErrNotFound) {
						t.Fatalf("failed proof published: %v", e)
					}
				}
				closes := 1
				if nilReader {
					closes = 0
				}
				if reader.closes != closes {
					t.Errorf("Close calls=%d want %d", reader.closes, closes)
				}
				if providerErr != nil && reader.reads != 0 {
					t.Fatal("read stream returned with fetch error")
				}
			})
		}
	}
}

func TestProducerHTTPCacheTransfer(t *testing.T) {
	raw, err := cudacache.MarshalArtifact(newTestArtifact(t, "http_transfer", "sm_90"))
	if err != nil {
		t.Fatal(err)
	}
	for _, fail := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "close failure"}[fail], func(t *testing.T) {
			cause := errors.New("artifact body close")
			body := &outcomeReader{Reader: bytes.NewReader(raw)}
			if fail {
				body.closeErr = cause
			}
			c, err := cudacache.New(cudacache.Config{RootDir: t.TempDir(), Remote: producerHTTP(t, body, http.StatusOK, int64(len(raw)), nil)})
			if err != nil {
				t.Fatal(err)
			}
			got, err := c.GetArtifact(context.Background(), "key")
			if fail {
				if got != nil {
					t.Fatal("accepted failed close")
				}
				checkOutcome(t, err, []error{cause}, true)
			} else if err != nil || got == nil {
				t.Fatalf("got=%v err=%v", got, err)
			}
			if body.closes != 1 {
				t.Fatalf("Close calls=%d", body.closes)
			}
		})
	}
}

func TestProducerProofCloseKeepsLocalPublication(t *testing.T) {
	private := ed25519.NewKeyFromSeed(make([]byte, ed25519.SeedSize))
	raw, err := cudacache.MarshalArtifact(newTestArtifact(t, "remote_proof_close", "sm_90"))
	if err != nil {
		t.Fatal(err)
	}
	cause := errors.New("proof close")
	reader := &outcomeReader{Reader: strings.NewReader("{}"), closeErr: cause}
	source := &outcomeSource{artifact: func(context.Context) (io.ReadCloser, int64, error) {
		return io.NopCloser(bytes.NewReader(raw)), int64(len(raw)), nil
	}, proof: func(context.Context, string, string) (io.ReadCloser, error) { return reader, nil }}
	root := t.TempDir()
	c, err := cudacache.New(cudacache.Config{RootDir: root, Remote: source, Trust: cudacache.TrustPolicy{RequireSigned: true, Signers: map[string]ed25519.PublicKey{"signer": private.Public().(ed25519.PublicKey)}}})
	if err != nil {
		t.Fatal(err)
	}
	local := newTestArtifact(t, "local_proof_close", "sm_90")
	reader.close = func() {
		if err := c.PutArtifact(context.Background(), "key", local); err != nil {
			t.Fatal(err)
		}
	}
	got, err := c.GetArtifact(context.Background(), "key")
	if got != nil {
		t.Fatal("failed proof returned value")
	}
	checkOutcome(t, err, []error{cause, cudacache.ErrUntrustedSource}, false)
	if reader.closes != 1 {
		t.Fatalf("Close calls=%d", reader.closes)
	}
	got, err = c.GetArtifact(context.Background(), "key")
	if err != nil || got == nil || got.Digest != local.Digest {
		t.Fatalf("local publication lost: got=%v err=%v", got, err)
	}
	fresh, err := cudacache.New(cudacache.Config{RootDir: root})
	if err != nil {
		t.Fatal(err)
	}
	got, err = fresh.GetArtifact(context.Background(), "key")
	if err != nil || got == nil || got.Digest != local.Digest {
		t.Fatalf("local publication did not persist: got=%v err=%v", got, err)
	}
}
