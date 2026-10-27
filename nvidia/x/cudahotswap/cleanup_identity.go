package cudahotswap

import (
	"errors"
	"fmt"
	"io"
	"reflect"
)

// CleanupIdentityError reports retained cleanup owners whose aliasing cannot
// be determined by interface equality. Register and Push can return this error
// joined with the original failure; use errors.As to find it.
//
// No dependent owner is closed, drained or released while this error remains
// unresolved. The caller must know whether the recorded owners are aliases or
// distinct, call ResolveAliases, and retry Service.Close. Resolution is a caller
// assertion, not a check of native resource identity. If that assertion cannot
// be made, keep the service and device alive rather than guess.
//
// The zero value has no cleanup to resolve.
type CleanupIdentityError struct {
	cleanup *requestCleanup
	owners  []io.Closer
}

func (e *CleanupIdentityError) Error() string {
	return "cleanup owner identity is ambiguous; resolve aliases before retrying service close"
}

// Owners returns the ordered owner occurrences named by this error. The slice
// belongs to the caller, but the owners still belong to the service: do not
// close them directly. Adding another owner makes this error stale; resolving it
// then returns an error without changing cleanup.
func (e *CleanupIdentityError) Owners() []io.Closer {
	if e == nil {
		return nil
	}
	return append([]io.Closer(nil), e.owners...)
}

// ResolveAliases records an exhaustive partition of Owners' indices into alias
// groups. A singleton asserts a distinct owner. Indices grouped together assert
// one cleanup obligation; its earliest occurrence is retained. For example,
// []int{0, 1} asserts aliases, while []int{0}, []int{1} asserts distinct owners.
// Known unequal comparable owners and different dynamic types cannot be merged.
//
// Resolution closes nothing. Retry Service.Close to perform cleanup and release
// the request lease. A failed close retains its owner for retry. Resolution is
// serialized with cleanup; stale or already-resolved errors and invalid groups
// change nothing. This method does not change the Loader ownership contract.
func (e *CleanupIdentityError) ResolveAliases(groups ...[]int) error {
	if e == nil || e.cleanup == nil {
		return errors.New("resolve cleanup aliases: no retained cleanup")
	}
	r := e.cleanup
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.identity != e || !r.ambiguous {
		return errors.New("resolve cleanup aliases: stale or already resolved error")
	}
	seen := make([]bool, len(e.owners))
	keep := make([]bool, len(e.owners))
	for _, group := range groups {
		if len(group) == 0 {
			return errors.New("resolve cleanup aliases: empty group")
		}
		first := len(e.owners)
		for _, i := range group {
			if i < 0 || i >= len(seen) || seen[i] {
				return fmt.Errorf("resolve cleanup aliases: invalid or repeated owner index %d", i)
			}
			seen[i] = true
			if i < first {
				first = i
			}
		}
		for j, i := range group {
			for _, k := range group[:j] {
				a, b := e.owners[k], e.owners[i]
				if reflect.TypeOf(a) != reflect.TypeOf(b) ||
					reflect.ValueOf(a).Comparable() && reflect.ValueOf(b).Comparable() && a != b {
					return errors.New("resolve cleanup aliases: known distinct owners cannot be merged")
				}
			}
		}
		keep[first] = true
	}
	for _, present := range seen {
		if !present {
			return errors.New("resolve cleanup aliases: every owner index must be included")
		}
	}
	owners := make([]io.Closer, 0, len(groups))
	for i, owner := range e.owners {
		if keep[i] {
			owners = append(owners, owner)
		}
	}
	r.owners = owners
	r.ambiguous = false
	r.identity = nil
	return nil
}
