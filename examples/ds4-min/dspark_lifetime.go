package main

import (
	"errors"
	"fmt"
	"math"
)

func dsparkDataOffset(data, tensor, within uint64) (int64, error) {
	if data > math.MaxInt64 || tensor > math.MaxInt64-data || within > math.MaxInt64-data-tensor {
		return 0, fmt.Errorf("dspark: support offset overflow")
	}
	return int64(data + tensor + within), nil
}

// Retain failed handles only. A later close must never retry a successful
// release, even if another handle in the same close failed.
func releaseDSparkHandles[T any](handles []T, free func(T) error) ([]T, error) {
	var retained []T
	var errs []error
	for _, handle := range handles {
		if err := free(handle); err != nil {
			retained = append(retained, handle)
			errs = append(errs, err)
		}
	}
	return retained, errors.Join(errs...)
}

// admitDSparkHandle never publishes an incomplete resource. Failed cleanup
// returns the still-owned handle for a later close attempt.
func admitDSparkHandle[T comparable](handle T, loadErr error, validate func(T) error, free func(T) (bool, error)) (T, []T, error) {
	var zero T
	err := loadErr
	if handle == zero && err == nil {
		return zero, nil, fmt.Errorf("dspark: missing resource handle")
	}
	if err == nil {
		err = validate(handle)
	}
	if err == nil {
		return handle, nil, nil
	}
	if handle == zero {
		return zero, nil, err
	}
	released, cleanup := free(handle)
	if !released {
		if cleanup == nil {
			cleanup = fmt.Errorf("dspark: resource release incomplete")
		}
		return zero, []T{handle}, errors.Join(err, cleanup)
	}
	return zero, nil, errors.Join(err, cleanup)
}
