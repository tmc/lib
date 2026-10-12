//go:build !darwin && !linux

package cudadiageval

import (
	"context"
	"errors"
	"os"
)

// Structural registration IO is portable. Safe descriptor-based acquisition
// needs a platform implementation before filesystem verification is available.
type registrationRoot struct{}

func registrationOpenRoot(string) (registrationRoot, error) {
	return registrationRoot{}, errors.New("verify registration: safe filesystem acquisition unavailable on this platform")
}
func (registrationRoot) close() error { return nil }
func (registrationRoot) open(context.Context, string) (*os.File, error) {
	return nil, errors.New("verify registration: safe filesystem acquisition unavailable on this platform")
}
