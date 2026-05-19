package fuse

import "errors"

var (
	ErrInvalidArgument = errors.New("fuse: invalid argument")
	ErrUnsupported     = errors.New("fuse: unsupported")
	ErrClosed          = errors.New("fuse: closed")
	ErrNotExist        = errors.New("fuse: does not exist")
	ErrReadOnly        = errors.New("fuse: read only")
)
