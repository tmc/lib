//go:build darwin

// Package metalicb records a fixed sequence of Metal compute dispatches and
// re-encodes that sequence into a new command buffer on each launch.
//
// Despite its historical name, the package does not create or execute a Metal
// indirect command buffer. Sequence launches are CPU-encoded and synchronous.
//
// Deprecated: use a package named for CPU-encoded dispatch sequences in new
// APIs. This package remains only for compatibility while no real consumer
// justifies a replacement package.
package metalicb
