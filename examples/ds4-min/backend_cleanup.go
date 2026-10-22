package main

import (
	"errors"
	"fmt"
	"io"
)

// backendCleanupOwner keeps the immutable foundation alive until every
// request using its resources has been released. It transfers cleanup through
// the returned error, using the standard io.Closer discovery contract.
type backendCleanupOwner struct {
	cause            error
	request, backend io.Closer
}

func (e *backendCleanupOwner) Error() string { return e.cause.Error() }
func (e *backendCleanupOwner) Unwrap() error { return e.cause }
func (e *backendCleanupOwner) Close() error {
	if e.request != nil {
		if err := e.request.Close(); err != nil {
			return err
		}
		e.request = nil
	}
	if e.backend != nil {
		if err := e.backend.Close(); err != nil {
			return fmt.Errorf("close backend: %w", err)
		}
		e.backend = nil
	}
	return nil
}

func finishBackendCleanup(err error, backend io.Closer) error {
	if requests := backendErrorOwners(err); len(requests) > 0 {
		return &backendCleanupOwner{cause: err, request: &backendCleanupGroup{requests}, backend: backend}
	}
	if closeErr := backend.Close(); closeErr != nil {
		return &backendCleanupOwner{cause: errors.Join(err, fmt.Errorf("close backend: %w", closeErr)), backend: backend}
	}
	return err
}

type backendCleanupGroup struct{ owners []io.Closer }

func (g *backendCleanupGroup) Close() error {
	var remaining []io.Closer
	var errs []error
	for _, owner := range g.owners {
		if err := owner.Close(); err != nil {
			remaining = append(remaining, owner)
			errs = append(errs, err)
		}
	}
	g.owners = remaining
	return errors.Join(errs...)
}

func backendErrorOwners(err error) []io.Closer {
	if err == nil {
		return nil
	}
	if owner, ok := err.(io.Closer); ok {
		return []io.Closer{owner}
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		var owners []io.Closer
		for _, part := range joined.Unwrap() {
			owners = append(owners, backendErrorOwners(part)...)
		}
		return owners
	}
	if wrapped, ok := err.(interface{ Unwrap() error }); ok {
		return backendErrorOwners(wrapped.Unwrap())
	}
	return nil
}
