package cudacache

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"unicode/utf8"
)

const (
	// RemoteArtifact identifies an artifact in a remote proof.
	RemoteArtifact = "artifact"
	// RemoteReceipt identifies a receipt in a remote proof.
	RemoteReceipt = "receipt"
	// MaxRemoteProofBytes bounds a remote proof independently of its payload.
	MaxRemoteProofBytes = 4096
)

// RemoteProofSource supplies canonical signed proofs separately from payloads.
// The cache closes each returned reader and reads at most MaxRemoteProofBytes+1
// bytes. Implementations must honor ctx, including while a read is blocked.
// The payload and proof need not be fetched atomically: the signature binds the
// bytes actually downloaded, so a mismatched pair refuses.
type RemoteProofSource interface {
	FetchProof(ctx context.Context, kind, key string) (io.ReadCloser, error)
}

type remoteProofMessage struct {
	Domain  string `json:"domain"`
	Version int    `json:"version"`
	Kind    string `json:"kind"`
	Key     string `json:"key"`
	Origin  string `json:"origin"`
	KeyID   string `json:"key_id"`
	SHA256  string `json:"sha256"`
}

type remoteProof struct {
	Message   remoteProofMessage `json:"message"`
	Signature []byte             `json:"signature"`
}

func validRemoteIdentity(identity string, limit int) bool {
	return identity != "" && len(identity) <= limit && utf8.ValidString(identity)
}

func validProofMessage(m remoteProofMessage) bool {
	return m.Domain == "cudacache.remote-entry" && m.Version == 1 &&
		(m.Kind == RemoteArtifact || m.Kind == RemoteReceipt) && validateKey(m.Key) == nil &&
		validRemoteIdentity(m.Origin, 512) && validRemoteIdentity(m.KeyID, 128)
}

// SignRemoteProof returns at most MaxRemoteProofBytes canonical v1 proof bytes
// for exactly data. Invalid UTF-8 identities are refused, not normalized. Origin is
// the signer's asserted producer identity, not a physical host or HTTPS peer.
// A signature establishes neither artifact safety nor current qualification.
// The private key belongs to the producer; caches receive only pinned public keys.
func SignRemoteProof(kind, key, origin, keyID string, data []byte, privateKey ed25519.PrivateKey) ([]byte, error) {
	hash := sha256.Sum256(data)
	m := remoteProofMessage{"cudacache.remote-entry", 1, kind, key, origin, keyID, hex.EncodeToString(hash[:])}
	if !validProofMessage(m) || len(privateKey) != ed25519.PrivateKeySize {
		return nil, fmt.Errorf("%w: invalid signing identity", ErrUntrustedSource)
	}
	message, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	proof, err := json.Marshal(remoteProof{m, ed25519.Sign(privateKey, message)})
	if err != nil {
		return nil, err
	}
	if len(proof) > MaxRemoteProofBytes {
		return nil, fmt.Errorf("%w: encoded proof exceeds limit", ErrUntrustedSource)
	}
	return proof, nil
}

func (tp TrustPolicy) authenticated() bool {
	return tp.RequireSigned || len(tp.AllowedOrigins) != 0 || len(tp.Signers) != 0
}

func copyTrust(tp TrustPolicy) (TrustPolicy, error) {
	out := TrustPolicy{RequireSigned: tp.RequireSigned, AllowedOrigins: append([]string(nil), tp.AllowedOrigins...), Signers: make(map[string]ed25519.PublicKey, len(tp.Signers))}
	for _, origin := range out.AllowedOrigins {
		if !validRemoteIdentity(origin, 512) {
			return TrustPolicy{}, fmt.Errorf("%w: invalid allowed origin", ErrUntrustedSource)
		}
	}
	for id, key := range tp.Signers {
		if !validRemoteIdentity(id, 128) || len(key) != ed25519.PublicKeySize {
			return TrustPolicy{}, fmt.Errorf("%w: invalid pinned signer", ErrUntrustedSource)
		}
		out.Signers[id] = append(ed25519.PublicKey(nil), key...)
	}
	return out, nil
}

func (c *Cache) verifyRemote(ctx context.Context, kind, key string, data []byte) error {
	if !c.trust.authenticated() {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	transport, ok := c.remote.(RemoteProofSource)
	if !ok {
		return fmt.Errorf("%w: proof transport unavailable", ErrUntrustedSource)
	}
	r, err := transport.FetchProof(ctx, kind, key)
	if err != nil {
		if r != nil {
			err = errors.Join(err, r.Close())
		}
		return remoteProofError(ctx, "fetch proof", err)
	}
	if r == nil {
		return remoteProofError(ctx, "nil proof stream", ErrUntrustedSource)
	}
	raw, readErr := io.ReadAll(io.LimitReader(r, MaxRemoteProofBytes+1))
	closeErr := r.Close()
	if err := errors.Join(readErr, closeErr, ctx.Err()); err != nil {
		return remoteProofError(ctx, "read proof", err)
	}
	if len(raw) > MaxRemoteProofBytes {
		return fmt.Errorf("%w: oversized proof", ErrUntrustedSource)
	}
	var proof remoteProof
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&proof); err != nil {
		return fmt.Errorf("%w: invalid proof", ErrUntrustedSource)
	}
	canonical, err := json.Marshal(proof)
	if err != nil || !bytes.Equal(canonical, raw) || !validProofMessage(proof.Message) || len(proof.Signature) != ed25519.SignatureSize {
		return fmt.Errorf("%w: noncanonical proof", ErrUntrustedSource)
	}
	m := proof.Message
	hash := sha256.Sum256(data)
	publicKey, known := c.trust.Signers[m.KeyID]
	if !known || m.Kind != kind || m.Key != key || m.SHA256 != hex.EncodeToString(hash[:]) || c.trust.Validate(m.Origin) != nil {
		return fmt.Errorf("%w: proof identity mismatch", ErrUntrustedSource)
	}
	message, err := json.Marshal(m)
	if err != nil || !ed25519.Verify(publicKey, message, proof.Signature) {
		return fmt.Errorf("%w: invalid proof signature", ErrUntrustedSource)
	}
	return nil
}

// A required proof that cannot be fetched or read remains untrusted. Preserve
// provider and cleanup causes; cancellation is a separate outcome.
func remoteProofError(ctx context.Context, action string, err error) error {
	err = errors.Join(ctx.Err(), err)
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("%s: %w", action, err)
	}
	return fmt.Errorf("%w: %s: %w", ErrUntrustedSource, action, err)
}
