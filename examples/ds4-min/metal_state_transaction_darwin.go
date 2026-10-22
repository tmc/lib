//go:build darwin && arm64

package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// metalStateLoadHooks is an internal fault-injection seam. Hooks run after
// actual field writes, never instead of them. Production passes nil.
type metalStateLoadHooks struct {
	afterWrite func(int, bool) error
}

func metalStatePoison(r *metalRuntime) error {
	if r != nil && errors.Is(r.failed, dsv4.ErrDesynchronized) {
		return r.failed
	}
	return nil
}

// replaceMetalStateFields preserves the old bytes until every replacement has
// succeeded. Metadata is published by the caller only after this returns nil.
func replaceMetalStateFields(fields []any, encoded [][]byte, hooks *metalStateLoadHooks, poison func(error)) error {
	if len(fields) != len(encoded) {
		return fmt.Errorf("Metal state load: field count mismatch")
	}
	var total int64
	for i, field := range fields {
		if err := validateMetalStateField(field, encoded[i]); err != nil {
			return fmt.Errorf("Metal state load: field %d: %w", i, err)
		}
		total += int64(len(encoded[i]))
		if total > metalStateMaxBytes {
			return fmt.Errorf("Metal state load: backup exceeds limit")
		}
	}
	backup := make([][]byte, len(fields))
	for i, field := range fields {
		var err error
		backup[i], err = encodeMetalStateField(field)
		if err != nil {
			return fmt.Errorf("Metal state load: backup field %d: %w", i, err)
		}
	}
	write := func(data [][]byte, rollback bool) error {
		for i, field := range fields {
			if err := decodeMetalStateField(field, data[i]); err != nil {
				return fmt.Errorf("field %d: %w", i, err)
			}
			if hooks != nil && hooks.afterWrite != nil {
				if err := hooks.afterWrite(i, rollback); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if err := write(encoded, false); err != nil {
		if rollbackErr := write(backup, true); rollbackErr != nil {
			joined := errors.Join(err, fmt.Errorf("Metal state rollback: %w", rollbackErr), dsv4.ErrDesynchronized)
			poison(joined)
			return joined
		}
		return err
	}
	return nil
}
