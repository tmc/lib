// Package fuse provides a pure-Go FUSE protocol and Linux device helper.
//
// The package exposes low-level FUSE frames and a small FileSystem interface.
// It does not use cgo or libfuse. Linux helpers for /dev/fuse are build-tagged
// and root/kernel checks are manual gated.
package fuse
