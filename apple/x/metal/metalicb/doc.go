//go:build darwin

// Package metalicb records a fixed sequence of Metal compute dispatches and
// re-encodes that sequence into a new command buffer on each launch.
//
// Despite its historical name, the package does not create or execute a Metal
// indirect command buffer. Launches are CPU-encoded and synchronous.
package metalicb
