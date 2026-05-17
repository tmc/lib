package kvm

import "errors"

var (
	ErrClosed          = errors.New("kvm: closed")
	ErrInvalidArgument = errors.New("kvm: invalid argument")
	ErrUnsupported     = errors.New("kvm: unsupported")
)
