//go:build linux

package main

import "github.com/tmc/lib/examples/ds4-min/dsv4"

// checkOpen permits construction but refuses new work once teardown starts.
// Close uses the device directly so retained owners remain retryable.
func (r *cudaFoundation) checkOpen() error {
	if r == nil || r.closing {
		return dsv4.ErrClosed
	}
	return nil
}

func (r *cudaFoundation) withContext(fn func() error) error {
	if err := r.checkOpen(); err != nil {
		return err
	}
	return r.device.WithContext(fn)
}
