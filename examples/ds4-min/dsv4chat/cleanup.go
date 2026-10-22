package dsv4chat

import (
	"errors"
	"io"
)

// cleanupOwner transfers unresolved cleanup responsibility through an error.
// A caller can discover it using errors.As into io.Closer. Close retries every
// owner, retaining only those whose release still fails.
type cleanupOwner struct {
	cause  error
	owners []io.Closer
}

func (e *cleanupOwner) Error() string { return e.cause.Error() }
func (e *cleanupOwner) Unwrap() error { return e.cause }
func (e *cleanupOwner) Close() error {
	var remaining []io.Closer
	var errs []error
	for _, owner := range e.owners {
		if err := owner.Close(); err != nil {
			remaining = append(remaining, owner)
			errs = append(errs, err)
		}
	}
	e.owners = remaining
	return errors.Join(errs...)
}

func retainCleanup(cause error, owners ...io.Closer) error {
	if len(owners) == 0 {
		return cause
	}
	return &cleanupOwner{cause: cause, owners: owners}
}

// cleanupOwners walks every error branch instead of using errors.As, which
// would stop at the first owner in a joined error tree.
func cleanupOwners(err error) []io.Closer {
	if err == nil {
		return nil
	}
	if owner, ok := err.(*cleanupOwner); ok {
		return append([]io.Closer(nil), owner.owners...)
	}
	if owner, ok := err.(io.Closer); ok {
		return []io.Closer{owner}
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		var owners []io.Closer
		for _, part := range joined.Unwrap() {
			owners = append(owners, cleanupOwners(part)...)
		}
		return owners
	}
	if wrapped, ok := err.(interface{ Unwrap() error }); ok {
		return cleanupOwners(wrapped.Unwrap())
	}
	return nil
}
