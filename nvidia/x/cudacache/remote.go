package cudacache

import (
	"context"
	"crypto/ed25519"
	"errors"
	"io"
)

var (
	// ErrRemoteNotFound is returned when an artifact or receipt is not found on the remote.
	ErrRemoteNotFound = errors.New("cudacache: remote entry not found")

	// ErrRemoteUnavailable is returned when remote network or service is unreachable.
	ErrRemoteUnavailable = errors.New("cudacache: remote unavailable")

	// ErrDownloadLimitExceeded is returned when a remote payload exceeds the configured byte limit.
	ErrDownloadLimitExceeded = errors.New("cudacache: remote download limit exceeded")

	// ErrUntrustedSource is returned when remote source or signer does not pass trust policy.
	ErrUntrustedSource = errors.New("cudacache: untrusted remote source")
)

// RemoteSource defines the contract for reading artifacts and receipts from a remote store or registry.
// Implementations MUST NOT execute downloaded code during lookup or fetching.
type RemoteSource interface {
	// FetchArtifact streams the artifact data with the given cache key or digest.
	// The returned reader must be closed by the caller.
	FetchArtifact(ctx context.Context, key string) (io.ReadCloser, int64, error)

	// FetchReceipt retrieves serialized JSON bytes for a search receipt key.
	FetchReceipt(ctx context.Context, key string) ([]byte, error)

	// PublishArtifact optionally uploads an artifact to the remote registry.
	PublishArtifact(ctx context.Context, key string, data []byte) error

	// PublishReceipt optionally uploads a search receipt to the remote registry.
	PublishReceipt(ctx context.Context, key string, data []byte) error
}

// TrustPolicy authenticates remote payloads using caller-pinned signer keys.
// AllowedOrigins names signer-attested producers, not network peers or physical
// hosts. Local entries remain trusted to the owner of Config.RootDir; a stored
// receipt does not establish current qualification. New copies the configuration.
type TrustPolicy struct {
	AllowedOrigins []string `json:"allowed_origins,omitempty"`
	RequireSigned  bool     `json:"require_signed"`
	// Signers maps signer key IDs to pinned Ed25519 public keys. Any configured
	// signer, origin restriction or RequireSigned requires a verified proof.
	Signers map[string]ed25519.PublicKey `json:"signers,omitempty"`
}

// Validate checks only origin allowlist membership. It does not authenticate
// the supplied string; Cache verifies its signed proof before calling Validate.
func (tp TrustPolicy) Validate(origin string) error {
	if len(tp.AllowedOrigins) == 0 {
		// If no origins specified, allow all by default.
		return nil
	}
	for _, allowed := range tp.AllowedOrigins {
		if allowed == "*" || allowed == origin {
			return nil
		}
	}
	return ErrUntrustedSource
}
