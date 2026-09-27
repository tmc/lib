// Package cufile is a declaration-only stand-in for the real cufile package.
package cufile

import "syscall"

type Handle uintptr

type ErrnoError struct {
	Func  string
	Errno syscall.Errno
}

func (e *ErrnoError) Error() string { return "" }
func (e *ErrnoError) Unwrap() error { return e.Errno }

func (h Handle) Read(bufPtrBase uintptr, size uint, fileOffset int64, bufPtrOffset int64) (int64, error) {
	return 0, nil
}
