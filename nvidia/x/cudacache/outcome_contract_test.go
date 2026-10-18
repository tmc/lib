package cudacache_test

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type outcomeSource struct {
	artifact func(context.Context) (io.ReadCloser, int64, error)
	receipt  func(context.Context) ([]byte, error)
	proof    func(context.Context, string, string) (io.ReadCloser, error)
}

func (s *outcomeSource) FetchArtifact(ctx context.Context, _ string) (io.ReadCloser, int64, error) {
	if s.artifact == nil {
		return nil, 0, cudacache.ErrRemoteNotFound
	}
	return s.artifact(ctx)
}
func (s *outcomeSource) FetchReceipt(ctx context.Context, _ string) ([]byte, error) {
	if s.receipt == nil {
		return nil, cudacache.ErrRemoteNotFound
	}
	return s.receipt(ctx)
}
func (*outcomeSource) PublishArtifact(context.Context, string, []byte) error { return nil }
func (*outcomeSource) PublishReceipt(context.Context, string, []byte) error  { return nil }
func (s *outcomeSource) FetchProof(ctx context.Context, kind, key string) (io.ReadCloser, error) {
	if s.proof != nil {
		return s.proof(ctx, kind, key)
	}
	return nil, cudacache.ErrUntrustedSource
}

type outcomeReader struct {
	io.Reader
	readErr, closeErr error
	read, close       func()
	reads, closes     int
}

func (r *outcomeReader) Read(p []byte) (int, error) {
	r.reads++
	if r.read != nil {
		r.read()
	}
	if r.readErr != nil {
		return 0, r.readErr
	}
	return r.Reader.Read(p)
}
func (r *outcomeReader) Close() error {
	r.closes++
	if r.close != nil {
		r.close()
	}
	return r.closeErr
}

func checkOutcome(t *testing.T, err error, want []error, outage bool) {
	t.Helper()
	if err == nil {
		t.Fatal("failed operation returned no error")
	}
	for _, target := range want {
		if !errors.Is(err, target) {
			t.Errorf("error %v does not retain %v", err, target)
		}
	}
	if errors.Is(err, cudacache.ErrRemoteUnavailable) != outage {
		t.Errorf("outage=%t, want %t: %v", errors.Is(err, cudacache.ErrRemoteUnavailable), outage, err)
	}
}

func TestOutcomeCompilerTarget(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	c, err := cudacache.New(cudacache.Config{RootDir: root})
	if err != nil {
		t.Fatal(err)
	}
	a := cudacache.NewNVRTCAdapter(c)
	id := cudacache.CompilerIdentity{TargetArch: "sm_90", OutputKind: nvrtcx.OutputPTX}
	art := newTestArtifact(t, "k", "sm_100")
	checkOutcome(t, a.Store(ctx, "input", id, art), []error{cudacache.ErrIncompatibleTarget}, false)
	got, err := a.GetOrCompile(ctx, "input", id, func(context.Context) (*cudakernel.Artifact, cudacache.ToolProvenance, error) {
		return art, cudacache.ToolProvenance{}, nil
	})
	if got != nil {
		t.Fatal("incompatible compile returned an artifact")
	}
	checkOutcome(t, err, []error{cudacache.ErrIncompatibleTarget}, false)
	key := cudacache.CompilerCacheKey("input", id)
	if _, err := os.Stat(filepath.Join(root, "manifests", key+".manifest")); !os.IsNotExist(err) {
		t.Fatalf("incompatible Store published: %v", err)
	}

	art = newTestArtifact(t, "k", id.TargetArch)
	if err := a.Store(ctx, "input", id, art); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "artifacts", key+"-"+art.Digest+".artifact")
	copy := *art
	copy.Target = "sm_100"
	data, err := cudacache.MarshalArtifact(&copy)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	fresh, err := cudacache.New(cudacache.Config{RootDir: root})
	if err != nil {
		t.Fatal(err)
	}
	got, hit, err := cudacache.NewNVRTCAdapter(fresh).Lookup(ctx, "input", id)
	if got != nil || hit {
		t.Fatal("incompatible persisted metadata admitted")
	}
	checkOutcome(t, err, []error{cudacache.ErrCorruptArtifact, cudacache.ErrIncompatibleTarget}, false)
}

func TestOutcomeRemoteArtifact(t *testing.T) {
	transport, cleanup := errors.New("transport fixture"), errors.New("close fixture")
	for _, tc := range []struct {
		name                                                                                               string
		fetch, read, close                                                                                 error
		nilReader, advertisedLimit, streamedLimit, cancelRead, cancelClose, deadline, malformed, untrusted bool
		want                                                                                               []error
		outage                                                                                             bool
	}{
		{name: "fetch", fetch: transport, nilReader: true, want: []error{transport}, outage: true},
		{name: "fetch reader and close", fetch: transport, close: cleanup, want: []error{transport, cleanup}, outage: true},
		{name: "read", read: transport, want: []error{transport}, outage: true},
		{name: "truncated transport", read: io.ErrUnexpectedEOF, want: []error{io.ErrUnexpectedEOF}, outage: true},
		{name: "close", close: cleanup, want: []error{cleanup}, outage: true},
		{name: "read and close", read: transport, close: cleanup, want: []error{transport, cleanup}, outage: true},
		{name: "nil stream", nilReader: true, outage: true},
		{name: "remote missing", fetch: cudacache.ErrRemoteNotFound, nilReader: true, want: []error{cudacache.ErrNotFound, cudacache.ErrRemoteNotFound}},
		{name: "provider missing", fetch: cudacache.ErrNotFound, nilReader: true, want: []error{cudacache.ErrNotFound}},
		{name: "missing and close", fetch: cudacache.ErrRemoteNotFound, close: cleanup, want: []error{cudacache.ErrNotFound, cudacache.ErrRemoteNotFound, cleanup}},
		{name: "provider limit", fetch: cudacache.ErrDownloadLimitExceeded, nilReader: true, want: []error{cudacache.ErrDownloadLimitExceeded}},
		{name: "advertised limit", advertisedLimit: true, want: []error{cudacache.ErrDownloadLimitExceeded}},
		{name: "limit and close", advertisedLimit: true, close: cleanup, want: []error{cudacache.ErrDownloadLimitExceeded, cleanup}},
		{name: "stream limit", streamedLimit: true, want: []error{cudacache.ErrDownloadLimitExceeded}},
		{name: "provider corruption", fetch: cudacache.ErrCorruptArtifact, nilReader: true, want: []error{cudacache.ErrCorruptArtifact}},
		{name: "corrupt decode", malformed: true, want: []error{cudacache.ErrCorruptArtifact}},
		{name: "provider untrusted", fetch: cudacache.ErrUntrustedSource, nilReader: true, want: []error{cudacache.ErrUntrustedSource}},
		{name: "provider incompatible", fetch: cudacache.ErrIncompatibleTarget, nilReader: true, want: []error{cudacache.ErrIncompatibleTarget}},
		{name: "provider unqualified", fetch: cudacache.ErrUnqualifiedOutcome, nilReader: true, want: []error{cudacache.ErrUnqualifiedOutcome}},
		{name: "provider unavailable", fetch: fmt.Errorf("provider: %w", cudacache.ErrRemoteUnavailable), nilReader: true, want: []error{cudacache.ErrRemoteUnavailable}, outage: true},
		{name: "proof untrusted", untrusted: true, want: []error{cudacache.ErrUntrustedSource}},
		{name: "provider cancellation", read: fmt.Errorf("provider: %w", context.Canceled), want: []error{context.Canceled}},
		{name: "cancel read retains error", read: transport, cancelRead: true, close: cleanup, want: []error{context.Canceled, transport, cleanup}},
		{name: "cancel close after bytes", cancelClose: true, want: []error{context.Canceled}},
		{name: "real deadline", deadline: true, want: []error{context.DeadlineExceeded}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tc.deadline {
				var stop context.CancelFunc
				ctx, stop = context.WithTimeout(ctx, 20*time.Millisecond)
				defer stop()
			}
			art := newTestArtifact(t, "k", "sm_90")
			data, err := cudacache.MarshalArtifact(art)
			if err != nil {
				t.Fatal(err)
			}
			limit := int64(len(data) + 1)
			if tc.malformed {
				data = []byte("{")
			}
			if tc.streamedLimit {
				data = bytes.Repeat([]byte("x"), int(limit)+1)
			}
			r := &outcomeReader{Reader: bytes.NewReader(data), readErr: tc.read, closeErr: tc.close}
			if tc.cancelRead {
				r.read = cancel
			}
			if tc.cancelClose {
				r.close = cancel
			}
			if tc.deadline {
				r.read = func() { <-ctx.Done(); r.readErr = ctx.Err() }
			}
			calls := 0
			s := &outcomeSource{artifact: func(context.Context) (io.ReadCloser, int64, error) {
				calls++
				if tc.nilReader {
					return nil, 0, tc.fetch
				}
				size := int64(-1)
				if tc.advertisedLimit {
					size = limit + 1
				}
				return r, size, tc.fetch
			}}
			cfg := cudacache.Config{RootDir: t.TempDir(), Remote: s, MaxRemoteDownloadBytes: limit}
			if tc.untrusted {
				cfg.Trust = cudacache.TrustPolicy{RequireSigned: true, Signers: map[string]ed25519.PublicKey{"fixture": bytes.Repeat([]byte{7}, ed25519.PublicKeySize)}}
			}
			c, err := cudacache.New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			got, err := c.GetArtifact(ctx, "artifact")
			if got != nil {
				t.Fatal("failed remote fetch returned an artifact")
			}
			checkOutcome(t, err, tc.want, tc.outage)
			// A real deadline can expire before GetArtifact acquires a reader.
			// In that case no reader is owned, and no Close is due.
			fetched := calls
			if fetched != 1 && !(tc.deadline && fetched == 0 && errors.Is(ctx.Err(), context.DeadlineExceeded)) {
				t.Fatalf("FetchArtifact calls=%d, want one fetch or deadline before acquisition", fetched)
			}
			wantClose := 1
			if tc.nilReader || fetched == 0 {
				wantClose = 0
			}
			if r.closes != wantClose {
				t.Errorf("Close calls=%d want %d", r.closes, wantClose)
			}
			if (tc.fetch != nil || tc.advertisedLimit) && r.reads != 0 {
				t.Error("read a rejected stream")
			}
			// A failed fetch must not populate either memory or disk. Query
			// with a fresh context and a source that now reports a miss.
			s.artifact = func(context.Context) (io.ReadCloser, int64, error) {
				calls++
				return nil, 0, cudacache.ErrRemoteNotFound
			}
			if got, err := c.GetArtifact(context.Background(), "artifact"); got != nil || !errors.Is(err, cudacache.ErrNotFound) || calls != fetched+1 {
				t.Fatalf("failed fetch was cached: got=%v err=%v calls=%d", got, err, calls)
			}
			fresh, err := cudacache.New(cudacache.Config{RootDir: cfg.RootDir})
			if err != nil {
				t.Fatal(err)
			}
			if got, err := fresh.GetArtifact(context.Background(), "artifact"); got != nil || !errors.Is(err, cudacache.ErrNotFound) {
				t.Fatalf("failed fetch persisted: %v", err)
			}
		})
	}
}

func TestOutcomeRemoteReceipt(t *testing.T) {
	cause := &os.PathError{Op: "fixture fetch", Path: "in-process", Err: errors.New("provider failed")}
	policy := cudacache.SearchPolicyID{Strategy: "beam", ScoringEpoch: "outcome"}
	receipt := &cudacache.SearchReceipt{PolicyID: policy, WorkloadKey: "work", CandidateID: "candidate", ArtifactKey: "artifact"}
	valid, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name                                    string
		err                                     error
		cancel, malformed, untrusted, oversized bool
		want                                    []error
		outage                                  bool
	}{
		{name: "fetch identity", err: cause, want: []error{cause}, outage: true},
		{name: "remote missing", err: cudacache.ErrRemoteNotFound, want: []error{cudacache.ErrNotFound, cudacache.ErrRemoteNotFound}},
		{name: "provider size", err: cudacache.ErrDownloadLimitExceeded, want: []error{cudacache.ErrDownloadLimitExceeded}},
		{name: "actual size", oversized: true, want: []error{cudacache.ErrDownloadLimitExceeded}},
		{name: "provider corruption", err: cudacache.ErrCorruptReceipt, want: []error{cudacache.ErrCorruptReceipt}},
		{name: "decode corruption", malformed: true, want: []error{cudacache.ErrCorruptReceipt}},
		{name: "provider untrusted", err: cudacache.ErrUntrustedSource, want: []error{cudacache.ErrUntrustedSource}},
		{name: "proof untrusted", untrusted: true, want: []error{cudacache.ErrUntrustedSource}},
		{name: "provider deadline", err: fmt.Errorf("provider: %w", context.DeadlineExceeded), want: []error{context.DeadlineExceeded}},
		{name: "cancel retains error", err: cause, cancel: true, want: []error{context.Canceled, cause}},
		{name: "late cancellation", cancel: true, want: []error{context.Canceled}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			s := &outcomeSource{receipt: func(context.Context) ([]byte, error) {
				calls++
				if tc.cancel {
					cancel()
				}
				if tc.malformed {
					return []byte("{"), tc.err
				}
				if tc.oversized {
					return bytes.Repeat([]byte("x"), len(valid)+2), tc.err
				}
				return valid, tc.err
			}}
			cfg := cudacache.Config{RootDir: t.TempDir(), Remote: s, MaxRemoteDownloadBytes: int64(len(valid) + 1)}
			if tc.untrusted {
				cfg.Trust = cudacache.TrustPolicy{RequireSigned: true, Signers: map[string]ed25519.PublicKey{"fixture": bytes.Repeat([]byte{7}, ed25519.PublicKeySize)}}
			}
			c, err := cudacache.New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			got, err := c.GetReceipt(ctx, policy, "work")
			if got != nil {
				t.Fatal("failed remote receipt returned a value")
			}
			checkOutcome(t, err, tc.want, tc.outage)
			if errors.Is(err, cause) {
				var pe *os.PathError
				if !errors.As(err, &pe) || pe != cause {
					t.Error("provider error type/identity lost")
				}
			}
			s.receipt = func(context.Context) ([]byte, error) { calls++; return nil, cudacache.ErrRemoteNotFound }
			if got, err := c.GetReceipt(context.Background(), policy, "work"); got != nil || !errors.Is(err, cudacache.ErrNotFound) || calls != 2 {
				t.Fatalf("failed receipt cached: got=%v err=%v calls=%d", got, err, calls)
			}
			fresh, err := cudacache.New(cudacache.Config{RootDir: cfg.RootDir})
			if err != nil {
				t.Fatal(err)
			}
			if got, err := fresh.GetReceipt(context.Background(), policy, "work"); got != nil || !errors.Is(err, cudacache.ErrNotFound) {
				t.Fatalf("failed receipt persisted: %v", err)
			}
		})
	}
}

func TestOutcomeCloseKeepsLocalPublication(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	art := newTestArtifact(t, "k", "sm_90")
	localData, err := cudacache.MarshalArtifact(art)
	if err != nil {
		t.Fatal(err)
	}
	var indented bytes.Buffer
	if err := json.Indent(&indented, localData, "", " "); err != nil {
		t.Fatal(err)
	}
	remoteData := indented.Bytes()
	closeErr := errors.New("close failed after local publication")
	var c *cudacache.Cache
	var publishErr error
	r := &outcomeReader{Reader: bytes.NewReader(remoteData), closeErr: closeErr, close: func() { publishErr = c.PutArtifact(ctx, art.Digest, art) }}
	s := &outcomeSource{artifact: func(context.Context) (io.ReadCloser, int64, error) { return r, int64(len(remoteData)), nil }}
	root := t.TempDir()
	c, err = cudacache.New(cudacache.Config{RootDir: root, Remote: s})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	var got *cudakernel.Artifact
	var fetchErr error
	go func() { defer close(done); got, fetchErr = c.GetArtifact(ctx, art.Digest) }()
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatal("Reader.Close callback blocked on publication mutex")
	}
	if got != nil || publishErr != nil || r.closes != 1 {
		t.Fatalf("got=%v publication=%v close calls=%d", got, publishErr, r.closes)
	}
	checkOutcome(t, fetchErr, []error{closeErr}, true)
	data, err := os.ReadFile(filepath.Join(root, "artifacts", art.Digest+".artifact"))
	if err != nil || !bytes.Equal(data, localData) {
		t.Fatalf("failed remote overwrote local publication: %v", err)
	}
	fresh, err := cudacache.New(cudacache.Config{RootDir: root})
	if err != nil {
		t.Fatal(err)
	}
	for _, reader := range []*cudacache.Cache{c, fresh} {
		if got, err := reader.GetArtifact(context.Background(), art.Digest); err != nil || !reflect.DeepEqual(got, art) {
			t.Fatalf("local publication lost: %v", err)
		}
	}
}

func TestOutcomeRemoteSuccess(t *testing.T) {
	for _, kind := range []string{cudacache.RemoteArtifact, cudacache.RemoteReceipt} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			art := newTestArtifact(t, "k", "sm_90")
			policy := cudacache.SearchPolicyID{Strategy: "beam", ScoringEpoch: "outcome"}
			receipt := &cudacache.SearchReceipt{PolicyID: policy, WorkloadKey: "work", CandidateID: "candidate", ArtifactKey: art.Digest}
			key := art.Digest
			var data []byte
			var err error
			if kind == cudacache.RemoteReceipt {
				key = cudacache.SearchReceiptKey(policy, "work")
				data, err = cudacache.MarshalReceipt(receipt)
			} else {
				data, err = cudacache.MarshalArtifact(art)
			}
			if err != nil {
				t.Fatal(err)
			}
			private := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{11}, ed25519.SeedSize))
			proof, err := cudacache.SignRemoteProof(kind, key, "fixture", "signer", data, private)
			if err != nil {
				t.Fatal(err)
			}
			body := &outcomeReader{Reader: bytes.NewReader(data)}
			proofBody := &outcomeReader{Reader: bytes.NewReader(proof)}
			fetches, proofs := 0, 0
			s := &outcomeSource{
				artifact: func(context.Context) (io.ReadCloser, int64, error) { fetches++; return body, int64(len(data)), nil },
				receipt:  func(context.Context) ([]byte, error) { fetches++; return data, nil },
				proof:    func(context.Context, string, string) (io.ReadCloser, error) { proofs++; return proofBody, nil },
			}
			cfg := cudacache.Config{RootDir: t.TempDir(), Remote: s, Trust: cudacache.TrustPolicy{RequireSigned: true, AllowedOrigins: []string{"fixture"}, Signers: map[string]ed25519.PublicKey{"signer": private.Public().(ed25519.PublicKey)}}}
			c, err := cudacache.New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			lookup := func(c *cudacache.Cache) {
				if kind == cudacache.RemoteArtifact {
					got, err := c.GetArtifact(ctx, key)
					if err != nil || !reflect.DeepEqual(got, art) {
						t.Fatalf("artifact positive: %v", err)
					}
				} else {
					got, err := c.GetReceipt(ctx, policy, "work")
					if err != nil || !reflect.DeepEqual(got, receipt) {
						t.Fatalf("receipt positive: %v", err)
					}
				}
			}
			lookup(c)
			lookup(c)
			if fetches != 1 || proofs != 1 || proofBody.closes != 1 {
				t.Fatalf("fetches=%d proofs=%d proof closes=%d", fetches, proofs, proofBody.closes)
			}
			if kind == cudacache.RemoteArtifact && body.closes != 1 {
				t.Fatalf("artifact closes=%d", body.closes)
			}
			fresh, err := cudacache.New(cudacache.Config{RootDir: cfg.RootDir})
			if err != nil {
				t.Fatal(err)
			}
			lookup(fresh)
		})
	}
}
