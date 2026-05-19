package virtiofs

import (
	"errors"
	"fmt"
)

var (
	ErrInvalidArgument = errors.New("virtiofs: invalid argument")
	ErrUnsupported     = errors.New("virtiofs: unsupported")
	ErrClosed          = errors.New("virtiofs: closed")
)

type ResponseError struct {
	Code int32
}

func (e ResponseError) Error() string {
	return fmt.Sprintf("virtiofs: fuse errno %d", e.Code)
}
