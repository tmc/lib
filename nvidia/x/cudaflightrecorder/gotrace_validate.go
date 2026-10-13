package cudaflightrecorder

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"unicode"
	"unicode/utf8"
)

// A GoTraceValidator decodes a Go execution trace with a decoder chosen by the
// caller and returns a short description of that decoder. Its result is the
// caller's assertion, not independent decoder authority. To report malformed
// input it returns an error wrapping ErrArtifactInvalid; any other error means
// the outcome is unknown.
//
// The validator receives a private copy of the trace bytes and must not
// modify it.
type GoTraceValidator func(ctx context.Context, data []byte) (decoder string, err error)

// GoTraceValidation reports one explicit validation of a bundle's go.trace.
type GoTraceValidation struct {
	Outcome     string // "decoded", "rejected", or "unavailable"
	Status      string // Outcome, a colon and the detail
	Authority   string // always GoTraceValidatorAuthority
	Decoder     string // validator's description, verbatim; empty unless decoded
	InputSHA256 string // sha256 of the validated bytes; empty if none were read
	InputBytes  int64
}

// GoTraceValidatorAuthority states the boundary of every GoTraceValidation.
const GoTraceValidatorAuthority = "caller-selected validator assertion; not independent decoder authority; timing and population unvalidated"

// Errors returned by ValidateGoTrace.
var (
	ErrGoTraceValidatorUnselected = errors.New("cudaflightrecorder: go trace validator not selected")
	ErrGoTraceNotRetained         = errors.New("cudaflightrecorder: go.trace not retained")
	ErrGoTraceValidatorResult     = errors.New("cudaflightrecorder: go trace validator result invalid")
)

// maxGoTraceValidationBytes matches the provider snapshot budget.
const maxGoTraceValidationBytes = 64 << 20

// ValidateGoTrace validates b.Files["go.trace"] with v. It never modifies b.
// The caller must not modify b during the call. Snapshot, FaultSnapshot,
// Handler and escalation never call it, and it changes no Manifest field.
//
// The recorder's own placeholder and magic checks run first and report
// "rejected" without calling v. The error is nil only for the "decoded"
// outcome; a cancelled ctx yields "unavailable" even if v returned.
func ValidateGoTrace(ctx context.Context, b *Bundle, v GoTraceValidator) (*GoTraceValidation, error) {
	return validateGoTraceBundle(ctx, b, v, bytes.Clone)
}

func validateGoTraceBundle(ctx context.Context, b *Bundle, v GoTraceValidator, clone func([]byte) []byte) (*GoTraceValidation, error) {
	if v == nil {
		return nil, ErrGoTraceValidatorUnselected
	}
	if b == nil {
		return nil, ErrGoTraceNotRetained
	}
	entry, ok := b.Files["go.trace"]
	if !ok {
		return nil, ErrGoTraceNotRetained
	}
	r := &GoTraceValidation{Authority: GoTraceValidatorAuthority}
	unavailable := func(err error) (*GoTraceValidation, error) {
		r.Outcome = "unavailable"
		r.Status = "unavailable: " + err.Error()
		r.Decoder = ""
		return r, err
	}
	if err := ctx.Err(); err != nil {
		return unavailable(fmt.Errorf("cudaflightrecorder: validate go trace: %w", err))
	}
	if len(entry) > maxGoTraceValidationBytes {
		return unavailable(fmt.Errorf("%w: go.trace %d bytes", ErrSnapshotBudgetExceeded, len(entry)))
	}
	private := clone(entry)
	sum := sha256.Sum256(private)
	r.InputSHA256 = hex.EncodeToString(sum[:])
	r.InputBytes = int64(len(private))
	if err := validateGoTrace(private); err != nil {
		r.Outcome = "rejected"
		r.Status = "rejected: " + err.Error()
		return r, err
	}
	if err := ctx.Err(); err != nil {
		return unavailable(fmt.Errorf("cudaflightrecorder: validate go trace: %w", err))
	}
	desc, verr := v(ctx, private)
	if sha256.Sum256(private) != sum {
		return unavailable(fmt.Errorf("%w: validator modified its input", ErrGoTraceValidatorResult))
	}
	if err := ctx.Err(); err != nil {
		return unavailable(fmt.Errorf("cudaflightrecorder: validate go trace: %w", err))
	}
	switch {
	case errors.Is(verr, ErrArtifactInvalid):
		r.Outcome = "rejected"
		r.Status = "rejected: caller assertion: " + verr.Error()
		return r, verr
	case verr != nil:
		return unavailable(verr)
	}
	if err := checkGoTraceDecoder(desc); err != nil {
		return unavailable(err)
	}
	r.Outcome = "decoded"
	r.Status = "decoded: caller assertion by " + desc + "; timing unvalidated"
	r.Decoder = desc
	return r, nil
}

func checkGoTraceDecoder(desc string) error {
	if desc == "" || len(desc) > 256 || !utf8.ValidString(desc) {
		return fmt.Errorf("%w: decoder description must be 1 to 256 bytes of UTF-8", ErrGoTraceValidatorResult)
	}
	for _, c := range desc {
		if unicode.IsControl(c) {
			return fmt.Errorf("%w: decoder description contains a control character", ErrGoTraceValidatorResult)
		}
	}
	return nil
}
