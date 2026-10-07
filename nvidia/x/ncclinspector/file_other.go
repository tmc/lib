//go:build !linux && !darwin

package ncclinspector

import (
	"errors"
	"os"
)

func openOrdinary(string) (*os.File, error) {
	return nil, errors.New("inspector: file decoding unsupported on this platform")
}
