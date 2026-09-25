//go:build !linux

package main

import (
	"context"
	"errors"
)

func runTUI(ctx context.Context, src source, h *host, opts options, rec *recorder) error {
	return errors.New("the interactive interface needs Linux; use -s for a snapshot")
}
