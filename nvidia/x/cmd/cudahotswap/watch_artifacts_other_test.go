//go:build !linux && !darwin

package main

import "errors"

func watchReadArtifact(string, int64) ([]byte, error) {
	return nil, errors.New("watch artifact descriptor reading is unsupported on this platform")
}
