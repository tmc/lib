//go:build !linux && !darwin

package ncclinspector

import (
	"errors"
	"os"
)

func openOrdinary(string) (*os.File, error) {
	return nil, errors.New("inspector: file decoding unsupported on this platform")
}

func fileIdentity(os.FileInfo) (uint64, uint64, error) {
	return 0, 0, errors.New("inspector: unavailable file identity")
}
