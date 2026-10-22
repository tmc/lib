package main

import "fmt"

// validateStateContext checks the per-request context requested from a
// backend. A request may use a smaller context than the model owner, but it
// cannot allocate an unusable timeline or exceed the owner's allocations.
func validateStateContext(context, max int) error {
	if context <= 1 {
		return fmt.Errorf("state context %d must exceed one token", context)
	}
	if max <= 1 {
		return fmt.Errorf("backend context %d is invalid", max)
	}
	if context > max {
		return fmt.Errorf("state context %d exceeds backend context %d", context, max)
	}
	return nil
}
