//go:build !linux

package nativereceipt

import (
	"errors"
	"os"
)

func newSession() (*Session, error) {
	return nil, errors.New("native receipt: Linux process mappings required")
}
func capture(*Session, string, string, string) (Stage, error) {
	return Stage{}, errors.New("native receipt: Linux process mappings required")
}

func Identity(int) (Process, error) {
	return Process{}, errors.New("native receipt: Linux process mappings required")
}
func metadataEqual(os.FileInfo, os.FileInfo) bool { return false }

func openReceipt(string) (*os.File, error) {
	return nil, errors.New("native receipt: Linux process mappings required")
}
func validateProof(Receipt) error {
	return errors.New("native receipt: Linux process mappings required")
}
