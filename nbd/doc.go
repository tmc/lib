package nbd

// Package nbd provides a pure-Go Network Block Device protocol and Linux
// device helper.
//
// The protocol implementation is independent of cgo and libnbd. A Device is a
// small Go interface over random-access storage. Serve exports a Device over an
// existing connection, and Dial connects to an export on an existing connection.
//
// Linux helpers for /dev/nbdX are available on Linux only. They use ioctls via
// the standard syscall package. Kernel-device tests are manual gated because
// they require root privileges, a loaded nbd module, and an unused /dev/nbdX.
