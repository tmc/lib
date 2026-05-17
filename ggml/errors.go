package ggml

import (
	"errors"
	"fmt"
)

var (
	ErrClosed          = errors.New("ggml: closed")
	ErrInvalidArgument = errors.New("ggml: invalid argument")
	ErrUnsupported     = errors.New("ggml: unsupported")
)

type OperationError struct {
	Op        string
	Rank      int
	Peer      int
	Backend   string
	Retryable bool
	Err       error
}

func (e *OperationError) Error() string {
	if e == nil {
		return "<nil>"
	}
	op := e.Op
	if op == "" {
		op = "operation"
	}
	msg := op
	if e.Rank >= 0 {
		msg += fmt.Sprintf(" rank %d", e.Rank)
	}
	if e.Peer >= 0 {
		msg += fmt.Sprintf(" peer %d", e.Peer)
	}
	if e.Backend != "" {
		msg += " backend " + e.Backend
	}
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

func (e *OperationError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}
