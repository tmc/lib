//go:build !darwin && !linux

package cudalock

import (
	"errors"
	"os"
)

func openGuardFile(string) (*os.File, error) {
	return nil, errors.New("advisory guard requires Linux or Darwin")
}

func guardLock(*os.File) error         { return errors.New("unsupported advisory guard") }
func guardUnlock(*os.File) error       { return errors.New("unsupported advisory guard") }
func guardBusy(error) bool             { return false }
func guardSingleLink(os.FileInfo) bool { return false }
