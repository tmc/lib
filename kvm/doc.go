// Package kvm provides Go bindings for the Linux KVM userspace API.
//
// The package wraps Linux kernel file descriptors with explicit Close methods.
// It keeps the ioctl interface small and Go-native while exposing the UAPI
// structs needed for low-level userspace control.
//
// The generated bindings are Linux-only. On other platforms, constructors
// return ErrUnsupported.
//
// Hardware execution is not run by default; tests that open device nodes must
// remain behind an explicit manual gate.
package kvm
