// Package bpfload loads Linux BPF programs and opens pinned BPF maps.
//
// Programs are raw eight-byte BPF instructions, not ELF containers. Loading
// requires the privileges and kernel support appropriate to the program type.
// This package implements upstream Linux BPF commands; it does not implement
// vendor GPU execution extensions. Descriptors returned by this package must
// be closed by their owner.
package bpfload
