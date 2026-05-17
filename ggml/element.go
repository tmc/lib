package ggml

import "fmt"

type Element interface {
	~float32 | ~int32
}

type DType int

const (
	DTypeF32 DType = iota
	DTypeI32
)

func dtypeOfElement[T Element]() (DType, error) {
	var zero T
	switch any(zero).(type) {
	case float32:
		return DTypeF32, nil
	case int32:
		return DTypeI32, nil
	default:
		return 0, fmt.Errorf("element: unsupported element type %T", zero)
	}
}
